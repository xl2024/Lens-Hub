
import torch
from collections.abc import Sequence
import numpy as np

import jlens
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.vis import SliceData, _meaningful_token_mask, _ranks_of

jlens.configure_logging()

from src.talens import TaylorLens
from src.lilens import LinearLens


def apply_jlens(
        lens: JacobianLens,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    if layers is None:
        layers = lens.source_layers
    out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
    if out_of_range:
        raise ValueError(
            f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
        )
    unknown = set(layers) - set(lens.source_layers)
    if use_jacobian and unknown:
        raise ValueError(
            f"layers {sorted(unknown)} not in source_layers; "
            f"fitted layers are {lens.source_layers}"
        )
    nn_model = model._model
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})

    input_ids = nn_model.tokenizer(prompt, max_length=max_seq_len, truncation=True, return_tensors="pt").input_ids
    # trace remotely with nnsight
    with nn_model.trace(prompt, remote=True):
        activations = {}
        for layer in record_at:
            activations[layer] = nn_model.model.layers[layer].output[0]
        activations.save()

    def select(layer: int) -> torch.Tensor:
        """Residuals at the requested positions: ``[n_positions, d_model]``."""
        full = activations[layer]  # [seq_len, d_model]
        return (full if positions is None else full[list(positions)]).float()

    lens_logits: dict[int, torch.Tensor] = {}
    for layer in layers:
        print(f"Processing layer {layer}...")
        residual = select(layer)
        if use_jacobian:
            J_l = lens.jacobians[layer]
            residual = torch.matmul(residual, J_l.T)
        lens_logits[layer] = model.unembed(residual)

    model_logits = model.unembed(select(final_layer))
    return lens_logits, model_logits, input_ids


def compute_slice_lens(
    model: LensModel,
    lens: JacobianLens | TaylorLens | LinearLens,
    prompt: str,
    *,
    top_n: int = 10,
    max_tracked: int | None = None,
    pinned_token_ids: set[int] | None = None,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    max_seq_len: int = 512,
    mask_display: bool = False,
) -> SliceData:
    nn_model = model._model
    tokenizer = nn_model.tokenizer
    pinned_token_ids = set(pinned_token_ids or ())
    final_layer = model.n_layers - 1

    if not lens.source_layers:
        if isinstance(lens, JacobianLens):
            raise ValueError("lens has no fitted layers (jacobians is empty)")
        elif isinstance(lens, TaylorLens):
            raise ValueError("lens has no fitted layers (taylors is empty)")
        elif isinstance(lens, LinearLens):
            raise ValueError("lens has no fitted layers (mappings is empty)")
        else:
            raise

    fitted_layers = lens.source_layers
    layers = fitted_layers[::layer_stride]
    if fitted_layers[-1] not in layers:
        layers.append(fitted_layers[-1])
    if final_layer not in layers:
        layers.append(final_layer)
    layers = sorted(set(layers))

    input_ids = tokenizer(prompt, max_length=max_seq_len, truncation=True, return_tensors="pt").input_ids
    full_len = input_ids.shape[1]
    start = 0 if last_n_tokens is None else max(0, full_len - last_n_tokens)
    seq_len = full_len - start
    context_token_ids = input_ids[0].tolist()
    context_token_strs = [
        tokenizer.decode([t], clean_up_tokenization_spaces=False)
        for t in context_token_ids
    ]

    # trace remotely with nnsight
    with nn_model.trace(prompt, remote=True):
        activations = {}
        for layer in layers:
            activations[layer] = nn_model.model.layers[layer].output[0]
        activations.save()

    def lens_logits(layer: int) -> torch.Tensor:
        residual = activations[layer][start:].float()
        if isinstance(lens, JacobianLens):
            if layer in lens.jacobians:
                J_l = lens.jacobians[layer]
                residual = torch.matmul(residual, J_l.T)
            # else: layer == final_layer, J = I -> this row is the model's output.
            return model.unembed(residual).float().detach()  # [seq_len, vocab_size]
        elif isinstance(lens, TaylorLens):
            if layer in lens.jacobians:
                residual = lens.transport(residual, layer)
            # else: layer == final_layer, J = I -> this row is the model's output.
            return model.unembed(residual).float().detach()  # [seq_len, vocab_size]
        elif isinstance(lens, LinearLens):
            if layer in lens.mappings:
                residual = lens.transport(residual, layer)
            # else: layer == final_layer, J = I -> this row is the model's output.
            return model.unembed(residual).float().detach()  # [seq_len, vocab_size]
        else:
            raise

    n_layers = len(layers)
    top_ids = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    top_ranks = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    display_mask: torch.Tensor | None = None
    vocab_size = 0

    # Pass 1: per-layer top-K. Logits are not retained across layers (they
    # would dominate memory at long seq_len x large vocab x n_layers).
    for layer_idx, layer in enumerate(layers):
        print(f"Processing layer {layer}...")
        logits = lens_logits(layer)
        vocab_size = int(logits.shape[-1])

        if not mask_display:
            top_idx = logits.topk(top_n, dim=-1).indices
            top_ids[:, layer_idx] = top_idx.cpu().numpy()
            top_ranks[:, layer_idx] = np.arange(top_n, dtype=np.int32)
        else:
            if display_mask is None:
                display_mask = _meaningful_token_mask(
                    tokenizer, vocab_size, logits.device
                )
            top_idx = (
                logits.masked_fill(~display_mask, float("-inf"))
                .topk(top_n, dim=-1)
                .indices
            )
            top_ids[:, layer_idx] = top_idx.cpu().numpy()
            top_ranks[:, layer_idx] = _ranks_of(logits, top_idx).cpu().numpy()
        del logits

    # Choose tracked tokens: pinned + most-frequently-high-ranked in the top-N grid.
    flat_ids = top_ids.ravel()
    flat_ranks = top_ranks.ravel()
    score_by_token: dict[int, float] = {}
    for token_id, rank in zip(flat_ids, flat_ranks, strict=True):
        score_by_token[int(token_id)] = score_by_token.get(int(token_id), 0.0) + 1.0 / (
            int(rank) + 1
        )
    by_score = sorted(score_by_token, key=score_by_token.__getitem__, reverse=True)
    tracked = sorted(set(by_score[:max_tracked]) | pinned_token_ids)

    # Pass 2: re-unembed per layer and compute tracked-token ranks chunked
    # (no full-seq argsort; peak memory is one layer's logits + a chunk sort).
    rank_tensor = np.full((seq_len, n_layers, len(tracked)), -1, dtype=np.int32)
    if tracked:
        tracked_tensor = torch.tensor(tracked, dtype=torch.long)
        for layer_idx, layer in enumerate(layers):
            logits = lens_logits(layer)
            rank_tensor[:, layer_idx] = (
                _ranks_of(logits, tracked_tensor.to(logits.device)).cpu().numpy()
            )
            del logits

    vocab_ids = (
        set(int(t) for t in np.unique(flat_ids)) | set(tracked) | set(context_token_ids)
    )
    vocab_fragment = {
        int(t): tokenizer.decode([int(t)], clean_up_tokenization_spaces=False)
        for t in vocab_ids
    }

    return SliceData(
        seq_len=seq_len,
        layers=layers,
        context_token_ids=context_token_ids,
        context_token_strs=context_token_strs,
        top_ids=top_ids,
        top_ranks=top_ranks,
        tracked_token_ids=tracked,
        rank_tensor=rank_tensor,
        vocab_fragment=vocab_fragment,
        vocab_size=vocab_size,
        pinned_token_ids=sorted(pinned_token_ids),
        ctx_offset=start,
    )
