# Copyright 2026 X. Liu
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
#
# NOTICE OF MODIFICATION:
# This file was modified from `jlens.lens` to apply ``A @ h + b`` to approximate all components between source layers and the final layer.

"""Applying a fitted Linear lens.

A :class:`LinearLens` holds the per-layer ``A`` and ``b`` matrices produced by
:func:`lilens_fitting.fit_lilens`. :meth:`LinearLens.apply` runs a forward pass and
reads out the requested layers; :meth:`LinearLens.transport` is the bare
`` A @ h + b`` for callers that already have residuals.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
import torch
from typing import Any

from jlens.protocol import LensModel


class LinearLens:
    """A fitted Linear lens: per-layer ``A`` and ``b`` matrices and the readout method.

    Attributes:
        mappings: ``{layer_index: Tensor[d_model+1, d_model]}``. Each ``mappings``
            maps the residual at layer ``l`` into the final-layer basis.
        source_layers: Sorted list of fitted layer indices.
        n_prompts: Number of prompts the lens was fitted over.
        d_model: Residual-stream width.
    """

    def __init__(
        self,
        mappings: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.mappings = {layer: m.float() for layer, m in mappings.items()}
        self.source_layers = sorted(self.mappings)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def __repr__(self) -> str:
        return (
            f"LinearLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers))"
        )

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save to ``path``. LinearLens are stored as ``dtype`` (default fp16:
        halves file size; entries are O(1) so the range is not a constraint
        and fp16's extra mantissa bits beat bf16 here)."""
        torch.save(
            {
                "M": {layer: m.to(dtype) for layer, m in self.mappings.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> LinearLens:
        """Load a lens previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "M" not in checkpoint:
            raise ValueError(
                f"{path} is not a LinearLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit() checkpoint?)"
            )
        return cls(
            mappings=checkpoint["M"],
            n_prompts=checkpoint["n_prompts"],
            d_model=checkpoint["d_model"],
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
    ) -> LinearLens:
        """Load a lens from a local file, a local directory, or a HuggingFace
        Hub ``repo_id``. ``filename`` is the path inside the directory or repo
        (so one Hub repo can host lenses for many models); ignored when
        ``name_or_path`` is itself a file. ``revision`` selects a Hub branch,
        tag, or commit. Deserialisation goes through :meth:`load`
        (``weights_only=True``)."""
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence[LinearLens]) -> LinearLens:
        """Combine lenses fitted on disjoint prompt subsets into one
        (``n_prompts``-weighted mean of the inputs).

        Args:
            lenses: Lenses to merge. Must agree on ``source_layers`` and
                ``d_model``.

        Raises:
            ValueError: If ``lenses`` is empty or the inputs disagree on shape.
        """
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.source_layers != first.source_layers
                or other.d_model != first.d_model
            ):
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            weighted_sum = sum(
                lens.mappings[layer] * lens.n_prompts for lens in lenses
            )
            merged[layer] = weighted_sum / n_total
        return cls(mappings=merged, n_prompts=n_total, d_model=first.d_model)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at ``layer`` into the final-layer basis: ``A @ h + b``.

        Args:
            residual: Tensor of shape ``[..., d_model]``.
            layer: Source layer index (must be in :attr:`source_layers`).
        """
        M_bar = self.mappings[layer].to(residual.device)
        ones = torch.ones_like(residual[..., :1])
        return torch.cat([residual,ones], dim=-1) @ M_bar

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_linear: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run ``model`` on ``prompt`` and return lens logits at ``positions``.

        Args:
            model: The model to read out from.
            prompt: Input text.
            layers: Layers to read out at. Defaults to all of
                :attr:`source_layers`. Must be a subset of
                :attr:`source_layers` when ``use_linear`` is ``True``.
            positions: Token positions to read out (Python indexing into the
                sequence; negative indices count from the end). ``None`` returns
                every position.
            max_seq_len: Truncate the prompt to this many tokens.
            use_linear: If ``False``, skip the ``A @ h + b`` transport (vanilla
                logit-lens baseline).

        Returns:
            A triple ``(lens_logits, model_logits, input_ids)``. ``lens_logits``
            maps each requested layer to a ``[n_positions, vocab_size]`` tensor;
            ``model_logits`` is the model's actual final-layer logits at the
            same positions (same shape). ``n_positions`` is ``len(positions)``,
            or the full sequence length when ``positions`` is ``None``.

        Raises:
            ValueError: If any requested layer is out of range for the model,
                or (with ``use_linear``) not in :attr:`source_layers`.
        """
        if layers is None:
            layers = self.source_layers
        out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
            )
        unknown = set(layers) - set(self.source_layers)
        if use_linear and unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not in source_layers; "
                f"fitted layers are {self.source_layers}"
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
            if use_linear:
                residual = self.transport(residual, layer)
            lens_logits[layer] = model.unembed(residual)

        model_logits = model.unembed(select(final_layer))
        return lens_logits, model_logits, input_ids
