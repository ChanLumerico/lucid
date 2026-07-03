"""``QuantizedLinearMLX`` — real int4/int8 GEMM Linear (Metal only).

Unlike :class:`~lucid.nn.quantized.Linear` (which dequantizes to float and runs
a float matmul), this layer stores the weight in MLX's group-wise packed format
and runs the genuine low-precision kernel (``quantized_matmul``) — the actual
compute + memory win.  It is Metal-only (the kernel is GPU-only) and is built
from a float ``Linear`` via :meth:`from_float`, choosing ``bits`` (4 or 8).
"""

from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
from lucid.quantization._qgemm import quantize, quantized_matmul

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class QuantizedLinearMLX(nn.Module):
    """int4/int8-weight linear backed by MLX ``quantized_matmul`` (Metal only)."""

    packed_weight: Tensor
    scales: Tensor
    biases: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        group_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        # Placeholders — real shapes are filled by from_float (packed layout
        # depends on bits/group_size).
        self.register_buffer("packed_weight", lucid.zeros((1,), dtype=lucid.int32))
        self.register_buffer("scales", lucid.zeros((1,)))
        self.register_buffer("biases", lucid.zeros((1,)))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_features))
        else:
            self.bias = None

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary layer
        """Run the MLX low-precision GEMM ``x @ packed_wᵀ`` (+ bias)."""
        y = quantized_matmul(
            x,
            self.packed_weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if self.bias is not None:
            y = y + self.bias
        return y

    @classmethod
    def from_float(
        cls, mod: nn.Module, bits: int = 8, group_size: int = 64
    ) -> QuantizedLinearMLX:
        """Quantize a float ``Linear``'s weight into MLX packed form (on Metal)."""
        lin = cast("nn.Linear", mod)
        m = cls(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            bits=bits,
            group_size=group_size,
        )
        weight = lin.weight.to("metal")  # (out, in)
        packed, scales, biases = quantize(weight, group_size=group_size, bits=bits)
        m.register_buffer("packed_weight", packed)
        m.register_buffer("scales", scales)
        m.register_buffer("biases", biases)
        if lin.bias is not None:
            m.register_buffer("bias", lin.bias.detach().to("metal"))
        return m

    @override
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, backend=mlx"
        )
