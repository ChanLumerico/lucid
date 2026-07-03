"""Python surface over the engine's MLX low-precision GEMM ops (GPU only).

Thin ``_wrap`` / ``_unwrap`` adapters around ``_C_engine.quantized.*`` (built
in Phase 6, E1).  These are the *real* int4/int8 kernels — MLX group-wise
affine quantize + quantized_matmul — as opposed to the dequantize-to-float
path.  Available only when the engine was compiled with the quantized ops
(``is_available()``) and only for Metal tensors.
"""

from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ``_C_engine.quantized`` is a C++ submodule present only when the engine was
# built with the E1 ops and is not declared in engine.pyi — hence the
# ``attr-defined`` ignores below.  Accessing it inside the functions (not at
# import) keeps this module importable against an engine that lacks it.


def is_available() -> bool:
    """True if the engine exposes the MLX quantized-GEMM ops."""
    return hasattr(_C_engine, "quantized")


def quantize(
    w: Tensor, group_size: int = 64, bits: int = 8
) -> tuple[Tensor, Tensor, Tensor]:
    """Group-wise quantize a weight → ``(packed_weight, scales, biases)``."""
    wq, scales, biases = _C_engine.quantized.quantize(  # type: ignore[attr-defined]
        _unwrap(w), group_size, bits
    )
    return _wrap(wq), _wrap(scales), _wrap(biases)


def dequantize(
    wq: Tensor, scales: Tensor, biases: Tensor, group_size: int = 64, bits: int = 8
) -> Tensor:
    """Reconstruct the float weight from its packed form."""
    out = _C_engine.quantized.dequantize(  # type: ignore[attr-defined]
        _unwrap(wq), _unwrap(scales), _unwrap(biases), group_size, bits
    )
    return _wrap(out)


def quantized_matmul(
    x: Tensor,
    wq: Tensor,
    scales: Tensor,
    biases: Tensor,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 8,
) -> Tensor:
    """``x @ packed_w`` via MLX's low-precision GEMM (Metal only)."""
    out = _C_engine.quantized.quantized_matmul(  # type: ignore[attr-defined]
        _unwrap(x),
        _unwrap(wq),
        _unwrap(scales),
        _unwrap(biases),
        transpose,
        group_size,
        bits,
    )
    return _wrap(out)
