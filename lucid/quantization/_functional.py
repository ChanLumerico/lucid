"""Core quantization math: ``quantize`` / ``dequantize`` / ``fake_quantize``.

These three primitives are the arithmetic core of the whole subsystem.
Everything else — observers, quantized modules, QAT — is expressed on
top of them.

``quantize`` maps reals to integer codes,

.. math::

    q = \\operatorname{clip}\\!\\big(\\operatorname{round}(x / s) + z,\\;
        q_\\min,\\; q_\\max\\big),

``dequantize`` inverts the affine map, :math:`\\hat{x} = (q - z)\\,s`,
and ``fake_quantize`` composes the two in the float domain
(:math:`\\hat{x} = \\text{dequantize}(\\text{quantize}(x))`) so a network
can *simulate* quantization while still training in float.

The rounding step has no useful derivative, so ``fake_quantize`` carries
a **straight-through estimator** (STE): gradient passes unchanged where
the code lands inside ``[quant_min, quant_max]`` and is zeroed where the
value saturates.  This is why it is implemented as a custom
:class:`~lucid.autograd.Function` rather than a plain composite.

All math runs through the public ``lucid.*`` op surface — no external
libraries (H4) — so the same code path is correct on both the Accelerate
(CPU) and MLX (GPU) streams.  Rounding and clipping are done in the
**float domain** and only the final cast lands in the integer storage
dtype, which sidesteps the missing CPU integer-clamp kernel.
"""

from typing import TYPE_CHECKING, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.autograd import Function, FunctionCtx

if TYPE_CHECKING:
    from lucid.quantization._qscheme import QDtype

    _ScaleLike = Tensor | float
    _ZeroPointLike = Tensor | float | int


def _reshape_for_channel(param: Tensor, ch_axis: int, ndim: int) -> Tensor:
    """Reshape a 1-D per-channel ``param`` to broadcast along ``ch_axis``.

    A length-``C`` vector of scales/zero-points becomes a shape with ``C``
    at ``ch_axis`` and ``1`` everywhere else, so it broadcasts against a
    rank-``ndim`` activation/weight tensor.
    """
    shape: list[int] = [1] * ndim
    shape[ch_axis] = -1
    return param.reshape(shape)


def quantize(
    x: Tensor,
    scale: _ScaleLike,
    zero_point: _ZeroPointLike,
    qdtype: QDtype,
    ch_axis: int | None = None,
) -> Tensor:
    r"""Map a real tensor to integer codes stored in ``qdtype.storage``.

    Computes :math:`q = \operatorname{clip}(\operatorname{round}(x/s)+z,
    q_\min, q_\max)` in the float domain, then casts to the integer
    storage dtype.

    Parameters
    ----------
    x : Tensor
        Real-valued input.
    scale : Tensor or float
        Quantization step ``s``.  A length-``C`` tensor for per-channel.
    zero_point : Tensor, float, or int
        Integer offset ``z`` mapping real 0 to a code.
    qdtype : QDtype
        Target grid — supplies ``quant_min`` / ``quant_max`` / ``storage``.
    ch_axis : int, optional
        Channel axis for per-channel quantization; ``None`` for per-tensor.

    Returns
    -------
    Tensor
        Integer codes with dtype ``getattr(lucid, qdtype.storage)``.
    """
    s: _ScaleLike = scale
    z: _ZeroPointLike = zero_point
    if ch_axis is not None and isinstance(s, Tensor):
        ndim = len(x.shape)
        s = _reshape_for_channel(s, ch_axis, ndim)
        if isinstance(z, Tensor):
            z = _reshape_for_channel(z, ch_axis, ndim)
    codes = lucid.clip(lucid.round(x / s) + z, qdtype.quant_min, qdtype.quant_max)
    storage_dtype: lucid.dtype = getattr(lucid, qdtype.storage)
    return codes.to(storage_dtype)


def dequantize(
    q: Tensor,
    scale: _ScaleLike,
    zero_point: _ZeroPointLike,
    ch_axis: int | None = None,
) -> Tensor:
    r"""Invert :func:`quantize`: :math:`\hat{x} = (q - z)\,s` in ``float32``.

    Parameters
    ----------
    q : Tensor
        Integer codes.
    scale : Tensor or float
        Quantization step used to produce ``q``.
    zero_point : Tensor, float, or int
        Zero-point used to produce ``q``.
    ch_axis : int, optional
        Channel axis for per-channel; ``None`` for per-tensor.

    Returns
    -------
    Tensor
        Reconstructed ``float32`` approximation of the original tensor.
    """
    s: _ScaleLike = scale
    z: _ZeroPointLike = zero_point
    if ch_axis is not None and isinstance(s, Tensor):
        ndim = len(q.shape)
        s = _reshape_for_channel(s, ch_axis, ndim)
        if isinstance(z, Tensor):
            z = _reshape_for_channel(z, ch_axis, ndim)
    return (q.to(lucid.float32) - z) * s


class _FakeQuantizeAffine(Function):
    """``dequantize(quantize(x))`` with a straight-through backward."""

    @override
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: FunctionCtx,
        x: Tensor,
        scale: _ScaleLike,
        zero_point: _ZeroPointLike,
        quant_min: int,
        quant_max: int,
        ch_axis: int | None,
    ) -> Tensor:
        """Fake-quantize ``x`` and cache the saturation mask for STE."""
        s: _ScaleLike = scale
        z: _ZeroPointLike = zero_point
        if ch_axis is not None and isinstance(s, Tensor):
            ndim = len(x.shape)
            s = _reshape_for_channel(s, ch_axis, ndim)
            if isinstance(z, Tensor):
                z = _reshape_for_channel(z, ch_axis, ndim)
        q_pre = lucid.round(x / s) + z
        q = lucid.clip(q_pre, quant_min, quant_max)
        dq = (q - z) * s
        # STE mask: 1 where the code is inside the grid, 0 where it saturates.
        in_lo = (q_pre >= quant_min).to(lucid.float32)
        in_hi = (q_pre <= quant_max).to(lucid.float32)
        ctx.save_for_backward(in_lo * in_hi)
        return dq

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:  # type: ignore[override]
        """Pass gradient through the non-saturated region only.

        ``x`` is the only positional (tensor) input to ``forward`` — the
        scale / zero-point / bounds are passed as keyword arguments so
        autograd does not track them — hence a single gradient is
        returned, matching the one registered input edge.
        """
        (mask,) = ctx.saved_tensors
        return grad_out * mask


def fake_quantize(
    x: Tensor,
    scale: _ScaleLike,
    zero_point: _ZeroPointLike,
    quant_min: int,
    quant_max: int,
    ch_axis: int | None = None,
) -> Tensor:
    r"""Simulate quantization in the float domain with a straight-through grad.

    Returns :math:`\hat{x} = \text{dequantize}(\text{quantize}(x))` as a
    ``float32`` tensor.  The backward pass is the straight-through
    estimator — identity where the code lands in ``[quant_min,
    quant_max]`` and zero where it saturates — so ``fake_quantize`` is
    differentiable and usable inside quantization-aware training.

    Parameters
    ----------
    x : Tensor
        Real-valued input (may require grad).
    scale : Tensor or float
        Quantization step.
    zero_point : Tensor, float, or int
        Zero-point offset.
    quant_min, quant_max : int
        Inclusive integer grid bounds.
    ch_axis : int, optional
        Channel axis for per-channel; ``None`` for per-tensor.

    Returns
    -------
    Tensor
        Fake-quantized ``float32`` tensor with STE gradient.
    """
    # Only ``x`` is positional; the rest are keyword arguments so autograd
    # registers a single input edge (STE flows to ``x`` alone).
    return _FakeQuantizeAffine.apply(  # type: ignore[return-value]
        x,
        scale=scale,
        zero_point=zero_point,
        quant_min=quant_min,
        quant_max=quant_max,
        ch_axis=ch_axis,
    )
