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

# ``_C_engine.quantized`` is an optional C++ submodule — present only when the
# engine was built with the E1 ops.  It is declared in ``engine.pyi`` (so the
# calls type-check), but accessed inside the functions (not at import) so this
# module stays importable against an engine that lacks it.


def is_available() -> bool:
    """Report whether the engine exposes the MLX quantized-GEMM ops (Metal only).

    The group-wise low-precision GEMM kernels are an optional C++ submodule
    (``_C_engine.quantized``) compiled only when the engine is built with the
    Phase 6 quantized ops.  Callers gate the real int4/int8 fast path on this
    before dispatching :func:`quantize` / :func:`quantized_matmul`, falling back
    to the dequantize-to-float path when it is absent.

    Returns
    -------
    bool
        ``True`` if the quantized submodule is present, ``False`` otherwise.
    """
    return hasattr(_C_engine, "quantized")


def quantize(
    w: Tensor, group_size: int = 64, bits: int = 8
) -> tuple[Tensor, Tensor, Tensor]:
    """Group-wise quantize a weight to MLX's packed int4/int8 form (Metal only).

    Wraps the engine's MLX group-wise affine quantizer: contiguous columns along
    the quantized axis are grouped, each group gets its own ``(scale, bias)`` pair,
    and the codes are bit-packed into a uint32 tensor.  The result feeds
    :func:`quantized_matmul` or is reconstructed by :func:`dequantize`.

    Parameters
    ----------
    w : Tensor
        Float weight residing on a Metal device.
    group_size : int, default 64
        Number of columns along the quantized axis sharing one ``(scale, bias)``.
    bits : int, default 8
        Bit-width of the packed codes; either ``4`` or ``8``.

    Returns
    -------
    tuple of Tensor
        ``(packed_weight, scales, biases)`` — the bit-packed uint32 (I32-tagged)
        weight, the per-group scales, and the per-group biases.
    """
    wq, scales, biases = _C_engine.quantized.quantize(_unwrap(w), group_size, bits)
    return _wrap(wq), _wrap(scales), _wrap(biases)


def dequantize(
    wq: Tensor, scales: Tensor, biases: Tensor, group_size: int = 64, bits: int = 8
) -> Tensor:
    """Reconstruct a float weight from its packed group-wise form (Metal only).

    Inverse of :func:`quantize`: each packed code is unpacked and mapped back to
    float via its group's ``scale`` and ``bias``, yielding an approximation of the
    original weight.  Used by the reference path when the fused GEMM is unavailable.

    Parameters
    ----------
    wq : Tensor
        Bit-packed uint32 (I32-tagged) weight produced by :func:`quantize`.
    scales : Tensor
        Per-group scales returned by :func:`quantize`.
    biases : Tensor
        Per-group biases returned by :func:`quantize`.
    group_size : int, default 64
        Number of columns per group — must match the value used to quantize.
    bits : int, default 8
        Bit-width of the packed codes; either ``4`` or ``8``.

    Returns
    -------
    Tensor
        The reconstructed float weight on the same Metal device.
    """
    out = _C_engine.quantized.dequantize(
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
    """Multiply a float activation by a packed low-precision weight (Metal only).

    Runs MLX's fused low-precision GEMM: the packed weight is dequantized inside the
    kernel and multiplied with the float ``x`` in one pass, so no full float weight
    is ever materialized.  With ``transpose=True`` this computes ``x @ packed_wᵀ``,
    the layout in which a linear layer stores its weight.

    Parameters
    ----------
    x : Tensor
        Float activation on a Metal device.
    wq : Tensor
        Bit-packed uint32 (I32-tagged) weight produced by :func:`quantize`.
    scales : Tensor
        Per-group scales returned by :func:`quantize`.
    biases : Tensor
        Per-group biases returned by :func:`quantize`.
    transpose : bool, default True
        If ``True``, contract against the transposed weight (``x @ packed_wᵀ``).
    group_size : int, default 64
        Number of columns per group — must match the value used to quantize.
    bits : int, default 8
        Bit-width of the packed codes; either ``4`` or ``8``.

    Returns
    -------
    Tensor
        The float product on the same Metal device.
    """
    out = _C_engine.quantized.quantized_matmul(
        _unwrap(x),
        _unwrap(wq),
        _unwrap(scales),
        _unwrap(biases),
        transpose,
        group_size,
        bits,
    )
    return _wrap(out)
