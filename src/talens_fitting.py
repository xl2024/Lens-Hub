# Copyright 2026 X. Liu
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
#
# NOTICE OF MODIFICATION:
# This file was modified from `jlens.fitting` to remotely fit talens (TaylorLens) with nnsight.

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence
import torch
from typing import Any

from jlens.protocol import LensModel
from jlens.fitting import valid_position_mask, _check_layer_indices, _atomic_save

from src.talens import TaylorLens

logger = logging.getLogger("jlens")

#: Positions before this index are excluded; early positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16


def taylor_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    pos_stride: int = 1,
    expand_at: float | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` and average residual 
    at final layer ``R_l`` for one prompt.

    Runs one forward pass on the prompt replicated ``dim_batch`` times along
    the batch axis, retains the graph, then runs ``ceil(d_model / dim_batch)``
    backward passes against it. Each backward computes ``dim_batch`` rows of
    ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
    dimension ``dim_start + b``, set at every valid target position. See the
    module docstring for the resulting estimator and how it relates to
    a strict per-position Jacobian.

    Args:
        model: The model to compute Jacobians for.
        prompt: Input text.
        source_layers: Layer indices ``l`` to compute ``J_l`` and ``R_l`` at.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer; negative indices count from the end. In some cases,
            targeting the penultimate layer can give a better-conditioned
            ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude; see :func:`valid_position_mask`.
        pos_stride: Fit Jacobians every this many positions.
        expand_at: At which the Taylor approximation is expanded.

    Returns:
        ``(last_residuals, jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
        source layer to a ``[d_model, d_model]`` fp32 CPU tensor.
    """
    nn_model = model._model
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    input_ids = nn_model.tokenizer(
        prompt, max_length=max_seq_len, truncation=True, return_tensors="pt"
    ).input_ids
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    _valid_positions = position_mask.nonzero(as_tuple=True)[0].tolist()
    valid_positions = [_valid_positions[i] for i in range(0, len(_valid_positions), pos_stride)]
    if valid_positions[-1] != _valid_positions[-1]:
        valid_positions.append(_valid_positions[-1])
    n_valid_positions = len(valid_positions)
    logger.info(f"valid_positions({n_valid_positions}): {valid_positions}")

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    last_residuals = {
        layer: torch.zeros(d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    for layer in source_layers:
        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)
            # replicated_ids = input_ids.repeat(n_dims_this_pass, 1)

            with nn_model.session(remote=True):
                rows = torch.zeros((n_dims_this_pass, n_valid_positions, d_model))
                last_residual = torch.zeros((n_valid_positions, d_model))
                for b in range(n_dims_this_pass):
                    for pos_idx, pos in enumerate(valid_positions):
                        with nn_model.trace(input_ids):
                            h_l = nn_model.model.layers[layer].output[0,pos,:]
                            new_h = torch.full_like(h_l, h_l.mean() if expand_at is None else expand_at)
                            new_h.requires_grad_(True)
                            nn_model.model.layers[layer].output[0,pos,:] = new_h

                            target_activation = nn_model.model.layers[target_layer].output[:,pos,:]  # [dim_batch, seq_len, d_model]
                            if pass_idx == 0 and b == 0:
                                last_residual[pos_idx,:] = target_activation[0,:]
                            cotangent = torch.zeros_like(target_activation)
                            cotangent[0, dim_start + b] = 1.0
                            loss = (target_activation * cotangent).sum()
                            with loss.backward():
                                grad = new_h.grad
                                rows[b,pos_idx,:] = grad
                rows_mean = rows.mean(dim=1).save()
                last_residual_mean = last_residual.mean(dim=0).save()

            jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = rows_mean.detach().float().cpu()
            del rows_mean
            if pass_idx == 0:
                last_residuals[layer] = last_residual_mean.detach().float().cpu()
                del last_residual_mean

            # if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
            logger.info(
                "    layer %d - pass %d/%d (dims %d-%d)",
                layer + 1,
                pass_idx + 1,
                n_passes,
                dim_start + 1,
                dim_start + n_dims_this_pass,
            )

    return last_residuals, jacobians, seq_len, n_valid_positions


def fit_talens(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    pos_stride: int = 1,
    expand_at: float | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> TaylorLens:
    """Fit ``R_l`` and ``J_l`` over a list of prompts and return a :class:`TaylorLens`.

    Per-prompt Jacobians from :func:`taylor_for_prompt` are accumulated as a
    running mean. If ``checkpoint_path`` is set, the running sum is written
    every ``checkpoint_every`` prompts (atomic) and resumed from on restart.

    Args:
        model: The model to fit on.
        prompts: Text prompts to average over. See the README for guidance on
            corpus size and distribution.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`taylor_for_prompt`. Defaults to the final
            layer; negative indices count from the end.
        dim_batch: See :func:`taylor_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: See :func:`taylor_for_prompt`.
        pos_stride: Run through sequence every this many positions.
        expand_at: At which the Taylor approximation is expanded.
        checkpoint_path: If set, write a resumable checkpoint here.
        checkpoint_every: Write the checkpoint every N prompts (default 1).
            ``None`` skips per-iteration writes and saves once at the end; the
            checkpoint can be large (``len(source_layers) * d_model**2 * 4``
            bytes), so raise this for large models.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.

    Returns:
        The fitted :class:`TaylorLens`.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    logger.info(
        "fit: n_layers=%d d_model=%d, fitting %d source layers "
        "(target=L%d) on %d prompts",
        n_layers,
        d_model,
        len(source_layers),
        target_layer,
        len(prompts),
    )

    # Running state: sum of per-prompt Jacobians, success count, and the list
    # index to resume from. ``next_idx`` is tracked separately from ``n_done``
    # so a too-short prompt that was skipped is not re-processed on resume.
    residual_sum: dict[int, torch.Tensor]
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    prompts_R_norms: dict[int, dict[int, float]]
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard it"
                )
        residual_sum, jacobian_sum, n_done, next_idx, prompts_R_norms = (
            state["residual_sum"],
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
            state["prompts_R_norms"],
        )
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            next_idx,
            len(prompts),
        )
        print(prompts_R_norms)
    else:
        residual_sum = {
            layer: torch.zeros(d_model, dtype=torch.float32)
            for layer in source_layers
        }
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0
        prompts_R_norms = {}

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "residual_sum": residual_sum,
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                    "prompts_R_norms": prompts_R_norms,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_res, per_prompt_J, seq_len, n_valid = taylor_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                pos_stride=pos_stride,
                expand_at=expand_at,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        # Per-prompt diagnostics, max over source layers: the prompt's own
        # Jacobian norm flags heavy-tailed outliers, and the relative shift
        # in the running mean tracks convergence (falls ~1/n once settled).
        res_norm = max(per_prompt_res[l].norm().item() for l in source_layers) / sqrt_d
        res_norm_min = min(per_prompt_res[l].norm().item() for l in source_layers) / sqrt_d
        prompts_R_norms[prompt_idx] = {l: (per_prompt_res[l].norm().item() / sqrt_d) for l in source_layers}
        if n_done > 0:
            mean_rel_change_res = max(
                (
                    (per_prompt_res[l] - residual_sum[l] / n_done).norm()
                    / ((n_done + 1) * (residual_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change_res = float("nan")

        for layer in source_layers:
            residual_sum[layer] += per_prompt_res[layer]

        prompt_norm = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_prompt_J[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]

        n_done += 1
        next_idx = prompt_idx + 1

        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  "
            "max||J||/sqrt(d)=%.3f  max_d_mean=%.2e "
            "max||h||/sqrt(d)=%.3f min||h||/sqrt(d)=%.3f  max_d_mean_res=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
            res_norm,
            res_norm_min,
            mean_rel_change_res
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    residual_mean = {layer: residual_sum[layer] / n_done for layer in source_layers}
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d prompts", n_done)
    return TaylorLens(residuals=residual_mean, jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
