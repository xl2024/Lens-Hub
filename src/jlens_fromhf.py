# Copyright 2026 X. Liu
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
#
# NOTICE OF MODIFICATION:
# This file was modified from `jlens.hf` to wrap a model with the same interfaces as jlens' HFLensModel.

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from nnsight import LanguageModel, ndif


ndif.register("src.jlens_fromhf")


def _resolve_attr_path(obj: Any, dotted_path: str) -> Any:
    return functools.reduce(getattr, dotted_path.split("."), obj)


@dataclass(frozen=True)
class Layout:
    """Where the lens-relevant submodules live inside a HuggingFace model.

    Attributes:
        path: Dotted attribute path from the ``*ForCausalLM`` to the bare text
            decoder (the module to call for a hooks-visible forward pass).
        layers: Attribute name on the text decoder for the residual blocks.
        norm: Attribute name for the final pre-unembed norm.
        embed: Attribute name for the input token embedding.
        lm_head: Attribute name on the ``*ForCausalLM`` for the unembedding.
    """

    path: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


class HFLensModel:
    """:class:`~jlens.protocol.LensModel` over a loaded HuggingFace model.

    Holds references into the caller's model; nothing is copied. The
    constructor mutates that model in place: every parameter gets
    ``requires_grad_(False)`` (the Jacobian fit needs grads only with respect
    to activations), and ``force_bos`` may set ``tokenizer.add_bos_token``.
    """

    def __init__(
        self,
        nn_model: LanguageModel,
        *,
        force_bos: bool = True,
    ) -> None:
        self._model = nn_model
        self.tokenizer = nn_model.tokenizer
        if (
            force_bos
            and getattr(self.tokenizer, "bos_token_id", None) is not None
            and hasattr(self.tokenizer, "add_bos_token")
        ):
            self.tokenizer.add_bos_token = True

        self._text_module = self._model.model
        self.layers = self._text_module.layers
        self._final_norm = self._text_module.norm
        self._embed_tokens = self._text_module.embed_tokens
        self._lm_head = self._model.lm_head

        text_config = self._model.config
        self.n_layers: int = text_config.num_hidden_layers
        self.d_model: int = text_config.hidden_size
        self._logit_softcap: float | None = getattr(
            text_config, "final_logit_softcapping", None
        )
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config.num_hidden_layers={self.n_layers} but found "
                f"{len(self.layers)} blocks."
            )
        
    def __repr__(self) -> str:
        if getattr(self.model.config, "architectures", None) is not None:
            arch_name = self.model.config.architectures[0]
        else: 
            arch_name = type(self.model).__name__
        return (
            f"HFLensModel({arch_name}, "
            f"n_layers={self.n_layers}, d_model={self.d_model})"
        )

    @property
    def input_device(self) -> torch.device:
        return torch.device("cpu")

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        return encoded.input_ids

    def forward(self, input_ids: torch.Tensor) -> Any:
        with self._model.trace(input_ids, remote=True):
            output = self._model.output.save()
        return output.float().cpu()

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        residual = residual.to(device=torch.device("cuda:0"), dtype=self._model.dtype)
        with self._model.trace(" ", remote=True):
            normed = self._final_norm(residual)
            logits = self._lm_head(normed)
            if self._logit_softcap is not None:
                logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
            saved_logits = logits.save()
        return saved_logits.float().cpu()


def from_hf(
    nn_model: LanguageModel,
    *,
    layout: Layout | None = None,
    text_module: str | None = None,
    force_bos: bool = True,
) -> HFLensModel:
    """Wrap a loaded HuggingFace model as a :class:`~jlens.protocol.LensModel`.

    Args:
        nn_model: A loaded ``*LanguageModel`` already on the target device and dtype.
        layout: Where the residual blocks / final norm / embedding / LM head
            live inside ``hf_model``. Auto-detected for the common HF families;
            pass explicitly only for unusual layouts.
        text_module: Deprecated alias for ``layout=Layout(path=text_module)``.
        force_bos: Some instruction-tuned checkpoints ship with
            ``add_bos_token=False``; raw-text prompts are degraded without an
            attention-sink BOS, so this sets it ``True`` by default. The
            attribute may have no effect for some fast-tokenizer
            configurations.
    """
    if text_module is not None:
        if layout is not None:
            raise TypeError("pass at most one of layout= / text_module=")
        layout = Layout(path=text_module)
    return HFLensModel(
        nn_model, force_bos=force_bos
    )
