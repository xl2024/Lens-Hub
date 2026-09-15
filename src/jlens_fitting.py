from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence
import torch
from typing import Any

from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.fitting import valid_position_mask, _check_layer_indices, _atomic_save

logger = logging.getLogger("jlens")

#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` for one prompt.

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
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer; negative indices count from the end. In some cases,
            targeting the penultimate layer can give a better-conditioned
            ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude; see :func:`valid_position_mask`.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
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
    valid_positions = position_mask.nonzero(as_tuple=True)[0]
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
        n_dims_this_pass = min(dim_batch, d_model - dim_start)
        # replicated_ids = input_ids.repeat(n_dims_this_pass, 1)

        with nn_model.session(remote=True):
            rows = {l: torch.zeros((n_dims_this_pass, d_model)) for l in source_layers}
            for b in range(n_dims_this_pass):
                with nn_model.trace(input_ids):
                    source_activations = {}
                    for l in source_layers:
                        h_l = nn_model.model.layers[l].output
                        h_l.requires_grad_(True)
                        source_activations[l] = h_l

                    target_activation = nn_model.model.layers[target_layer].output  # [dim_batch, seq_len, d_model]
                    cotangent = torch.zeros_like(target_activation)
                    cotangent[0, valid_positions, dim_start + b] = 1.0
                    loss = (target_activation * cotangent).sum()
                    with loss.backward():
                        for l in reversed(source_layers):
                            grad = source_activations[l].grad
                            # grad: [n_dims_this_pass, seq_len, d_model]
                            rows[l][b,:] = grad[0, valid_positions, :].mean(dim=0)
            rows.save()

        for layer in rows:
            jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = rows[layer].float().cpu()
        del rows

        # if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
        logger.info(
            "    pass %d/%d (dims %d-%d)",
            pass_idx + 1,
            n_passes,
            dim_start + 1,
            dim_start + n_dims_this_pass,
        )

    return jacobians, seq_len, n_valid_positions


def fit_jlens(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> JacobianLens:
    """Fit ``J_l`` over a list of prompts and return a :class:`JacobianLens`.

    Per-prompt Jacobians from :func:`jacobian_for_prompt` are accumulated as a
    running mean. If ``checkpoint_path`` is set, the running sum is written
    every ``checkpoint_every`` prompts (atomic) and resumed from on restart.

    Args:
        model: The model to fit on.
        prompts: Text prompts to average over. See the README for guidance on
            corpus size and distribution.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`jacobian_for_prompt`. Defaults to the final
            layer; negative indices count from the end.
        dim_batch: See :func:`jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: See :func:`jacobian_for_prompt`.
        checkpoint_path: If set, write a resumable checkpoint here.
        checkpoint_every: Write the checkpoint every N prompts (default 1).
            ``None`` skips per-iteration writes and saves once at the end; the
            checkpoint can be large (``len(source_layers) * d_model**2 * 4``
            bytes), so raise this for large models.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.

    Returns:
        The fitted :class:`JacobianLens`.
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
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
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
        jacobian_sum, n_done, next_idx = (
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
        )
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            next_idx,
            len(prompts),
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        # Per-prompt diagnostics, max over source layers: the prompt's own
        # Jacobian norm flags heavy-tailed outliers, and the relative shift
        # in the running mean tracks convergence (falls ~1/n once settled).
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
            "max||J||/sqrt(d)=%.3f  max_d_mean=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d prompts", n_done)
    return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
