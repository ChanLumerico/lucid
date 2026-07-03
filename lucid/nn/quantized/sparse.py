"""Quantized ``Embedding`` — int8 table storage, dequantize-on-lookup.

The embedding table is stored as int8 with a per-row (per-token) scale, which
is the dominant memory win for large-vocabulary embeddings; the forward
dequantizes the table and performs the ordinary lookup.
"""

from typing import TYPE_CHECKING, Protocol, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import quantize_weight
from lucid.quantization._functional import dequantize
from lucid.quantization._qscheme import qint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    class _FloatEmbedding(Protocol):
        num_embeddings: int
        embedding_dim: int
        weight: Tensor
        padding_idx: int | None
        qconfig: object


class Embedding(nn.Module):
    """int8 quantized embedding lookup table (built via :meth:`from_float`)."""

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight_ch_axis = 0
        self.register_buffer(
            "weight_int8",
            lucid.zeros((num_embeddings, embedding_dim), dtype=lucid.int8),
        )
        self.register_buffer("weight_scale", lucid.ones(num_embeddings))
        self.register_buffer("weight_zero_point", lucid.zeros(num_embeddings))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # index → embeddings
        """Dequantize the table and look up ``x``."""
        weight = dequantize(
            self.weight_int8, self.weight_scale, self.weight_zero_point, ch_axis=0
        )
        return F.embedding(x, weight, self.padding_idx)

    @classmethod
    def from_float(cls, mod: nn.Module) -> Embedding:
        """Quantize a float :class:`~lucid.nn.Embedding`'s table (per-row int8)."""
        f = cast("_FloatEmbedding", mod)
        qmod = cls(f.num_embeddings, f.embedding_dim, padding_idx=f.padding_idx)
        # Per-row weight quantization (ch_axis 0 = per token) via the shared helper,
        # falling back to a default per-channel qint8 observer if no qconfig is set.
        if getattr(mod, "qconfig", None) is None:
            from lucid.quantization.observer import PerChannelMinMaxObserver
            from lucid.quantization._functional import quantize
            from lucid.quantization._qscheme import per_channel_symmetric

            obs = PerChannelMinMaxObserver(
                ch_axis=0, qscheme=per_channel_symmetric, qdtype=qint8
            )
            obs(f.weight)
            scale, zero_point = obs.calculate_qparams()
            codes = quantize(f.weight, scale, zero_point, qint8, ch_axis=0)
        else:
            codes, scale, zero_point, _ = quantize_weight(mod)
        qmod.register_buffer("weight_int8", codes)
        qmod.register_buffer("weight_scale", scale)
        qmod.register_buffer("weight_zero_point", zero_point)
        return qmod

    @override
    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        )
