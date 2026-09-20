# Copyright 2026 X. Liu
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
#
# NOTICE OF MODIFICATION:
# This file was modified from `jlens.fitting` to remotely fit lilens (LinearLens) with nnsight.

from __future__ import annotations

import logging
from collections.abc import Sequence
import torch
import math

from jlens.protocol import LensModel
from jlens.fitting import valid_position_mask, _check_layer_indices

from src.lilens import LinearLens

logger = logging.getLogger("jlens")

#: Positions before this index are excluded; early positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16


def fit_lilens(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 10,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> LinearLens:
    """Fit ``A_l`` and ``b_l`` over a list of prompts and return a :class:`LinearLens`.
    Args:
        model: The model to fit on.
        prompts: Text prompts to average over.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: Defaults to the final layer; negative indices count from the end.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: Positions before this index are excluded; early positions act as 
            attention sinks and have atypical residual statistics.

    Returns:
        The fitted :class:`LinearLens`.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    nn_model = model._model
    d_model = nn_model.config.hidden_size
    input_ids = nn_model.tokenizer(prompts, max_length=max_seq_len, truncation=True, return_tensors="pt").input_ids
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    valid_positions = [i for i,m in enumerate(position_mask) if m]
    n_valid_positions = int(position_mask.sum())
    assert len(valid_positions) == n_valid_positions
    n_prompts = len(prompts)
    n_tokens = n_prompts * n_valid_positions
    logger.info(
        "fit: n_layers=%d d_model=%d, fitting %d source layers "
        "(target=L%d) on %d prompts with %d valid positions and %d tokens",
        n_layers,
        d_model,
        len(source_layers),
        target_layer,
        n_prompts,
        n_valid_positions,
        n_tokens,
    )

    with nn_model.session(remote=True):
        mappings = {
            layer: torch.zeros(d_model+1, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        device = nn_model.model.embed_tokens.weight.device
        XtX = {
            layer: torch.zeros(d_model + 1, d_model + 1, dtype=torch.float32, device=device)
            for layer in source_layers
        }
        XtY = {
            layer: torch.zeros(d_model + 1, d_model, dtype=torch.float32, device=device)
            for layer in source_layers
        }
        
        for pass_idx, dim_start in enumerate(range(0, n_prompts, dim_batch)):
            n_dims_this_pass = min(dim_batch, n_prompts - dim_start)
            with nn_model.trace(input_ids[dim_start:dim_start+n_dims_this_pass]):
                H_final = nn_model.model.layers[target_layer].output.float()
                H_final_token = H_final[:, position_mask, :].reshape(-1, d_model)
                assert H_final_token.shape[0] == n_dims_this_pass*n_valid_positions, f"shape mismatch (target layer {target_layer}): {H_final_token.shape[0]} != {n_dims_this_pass*n_valid_positions}"

            with nn_model.trace(input_ids[dim_start:dim_start+n_dims_this_pass]):
                for layer in source_layers:
                    H_l = nn_model.model.layers[layer].output.float()
                    H_l_token = H_l[:, position_mask, :].reshape(-1, d_model)
                    assert H_l_token.shape[0] == n_dims_this_pass*n_valid_positions, f"shape mismatch (source layer {layer}): {H_l_token.shape[0]} != {n_dims_this_pass*n_valid_positions}"
                    ones = torch.ones(H_l_token.shape[0], 1, dtype=H_l_token.dtype, device=H_l_token.device)
                    X = torch.cat([H_l_token, ones], dim=-1)
                    XtX[layer] += X.T @ X
                    XtY[layer] += X.T @ H_final_token


        ridge = 1e-4
        eye = torch.eye(d_model + 1, dtype=XtX[source_layers[0]].dtype, device=XtX[source_layers[0]].device)
        eye[-1, -1] = 0.0  # does not regularize bias
        for layer in source_layers:
            W_l = torch.linalg.solve(XtX[layer] + ridge * eye, XtY[layer])
            mappings[layer] = W_l
        mappings.save()

    logger.info("fit: done")
    return LinearLens(mappings=mappings, n_prompts=len(prompts), d_model=d_model)
