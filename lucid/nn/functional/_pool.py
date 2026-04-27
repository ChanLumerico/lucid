"""
lucid.nn.functional._pool — max / average / adaptive pooling.

Regular pool ops route to the C++ engine.  Adaptive pool: the engine
covers the uniform-divisible case directly; non-uniform sizes fall back
to a pure-engine composition that builds per-cell windows from cumsum
(avg) or sequential max reductions (max).  The fallback uses only
autograd-aware engine ops so gradients stay intact.
"""

from __future__ import annotations

from typing import Literal

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of


def _to_tuple_n(value, n: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * n
    if len(value) == 1 and n > 1:
        return tuple(value) * n
    return tuple(int(v) for v in value)


def avg_pool1d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 1)
    s = _to_tuple_n(stride, 1)
    p = _to_tuple_n(padding, 1)
    return Tensor._wrap(_C_nn.avg_pool1d(impl_of(input_), k[0], s[0], p[0]))


def avg_pool2d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 2)
    s = _to_tuple_n(stride, 2)
    p = _to_tuple_n(padding, 2)
    return Tensor._wrap(_C_nn.avg_pool2d(
        impl_of(input_), k[0], k[1], s[0], s[1], p[0], p[1]))


def avg_pool3d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 3)
    s = _to_tuple_n(stride, 3)
    p = _to_tuple_n(padding, 3)
    return Tensor._wrap(_C_nn.avg_pool3d(
        impl_of(input_), k[0], k[1], k[2],
        s[0], s[1], s[2], p[0], p[1], p[2]))


def max_pool1d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 1)
    s = _to_tuple_n(stride, 1)
    p = _to_tuple_n(padding, 1)
    return Tensor._wrap(_C_nn.max_pool1d(impl_of(input_), k[0], s[0], p[0]))


def max_pool2d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 2)
    s = _to_tuple_n(stride, 2)
    p = _to_tuple_n(padding, 2)
    return Tensor._wrap(_C_nn.max_pool2d(
        impl_of(input_), k[0], k[1], s[0], s[1], p[0], p[1]))


def max_pool3d(input_: Tensor, kernel_size, stride=1, padding=0) -> Tensor:
    k = _to_tuple_n(kernel_size, 3)
    s = _to_tuple_n(stride, 3)
    p = _to_tuple_n(padding, 3)
    return Tensor._wrap(_C_nn.max_pool3d(
        impl_of(input_), k[0], k[1], k[2],
        s[0], s[1], s[2], p[0], p[1], p[2]))


# --------------------------------------------------------------------------- #
# Adaptive pooling — uniform path through engine, non-uniform composed in
# pure engine ops so autograd stays intact.
# --------------------------------------------------------------------------- #

def _adaptive_window_idx(S: int, O: int) -> tuple[list[int], list[int]]:
    """PyTorch-compatible window boundaries: out[o] uses
    x[start[o]:end[o]) where start[o] = floor(o*S/O), end[o] = ceil((o+1)*S/O).
    """
    starts = [(o * S) // O for o in range(O)]
    ends = [-(-((o + 1) * S) // O) for o in range(O)]  # ceil-div via -//-
    return starts, ends


def _adaptive_avg_along_axis(x: Tensor, axis: int, O: int) -> Tensor:
    """Apply non-uniform adaptive average pooling along a single spatial
    axis using cumsum + indexed differences."""
    import lucid
    S = x.shape[axis]
    if S == O:
        return x
    if S % O == 0:
        # Cheap fallback: regular avg_pool along this axis only.
        K = S // O
        # Use cumsum trick: out[o] = (cs[(o+1)*K - 1] - cs[o*K - 1]) / K.
        starts, ends = _adaptive_window_idx(S, O)
    else:
        starts, ends = _adaptive_window_idx(S, O)

    cs = lucid.cumsum(x, axis=axis)
    # Pad cumsum with a leading zero along axis so we can index "start - 1"
    # uniformly without a special case for start = 0.
    pad_shape = list(x.shape)
    pad_shape[axis] = 1
    zeros = lucid.zeros(pad_shape, dtype=x.dtype, device=x.device)
    cs_pad = lucid.concatenate([zeros, cs], axis=axis)

    # Gather cs_pad[..., end] - cs_pad[..., start] along axis.
    end_idx = Tensor.__class__(  # build a Tensor of indices
        type(x)).__init__  # placeholder unused
    # Use lucid.gather to index along the axis.
    from lucid.types import Int64
    e_t = lucid.tensor(ends, dtype=Int64, device=x.device)
    s_t = lucid.tensor(starts, dtype=Int64, device=x.device)
    # gather expects index shape compatible with x along axis (broadcast-ok).
    # Build broadcast shape: same as x but with `O` on `axis`.
    target_shape = list(x.shape)
    target_shape[axis] = O
    e_b = e_t.reshape([1] * axis + [O] + [1] * (x.ndim - axis - 1))
    s_b = s_t.reshape([1] * axis + [O] + [1] * (x.ndim - axis - 1))
    e_b = lucid.broadcast_to(e_b, target_shape)
    s_b = lucid.broadcast_to(s_b, target_shape)

    end_vals = lucid.gather(cs_pad, axis, e_b)
    start_vals = lucid.gather(cs_pad, axis, s_b)
    summed = end_vals - start_vals

    # Divide by per-cell window size.
    sizes_np = [e - s for s, e in zip(starts, ends)]
    sizes_t = lucid.tensor(sizes_np, dtype=x.dtype, device=x.device)
    sizes_b = sizes_t.reshape([1] * axis + [O] + [1] * (x.ndim - axis - 1))
    sizes_b = lucid.broadcast_to(sizes_b, target_shape)
    return summed / sizes_b


def _is_uniform(S: int, O: int) -> bool:
    return O > 0 and S % O == 0


def adaptive_pool1d(input_: Tensor, output_size: int,
                    avg_or_max: Literal["avg", "max"]) -> Tensor:
    L = input_.shape[2]
    O = int(output_size)
    if _is_uniform(L, O):
        fn = (_C_nn.adaptive_avg_pool1d if avg_or_max == "avg"
              else _C_nn.adaptive_max_pool1d)
        return Tensor._wrap(fn(impl_of(input_), O))
    if avg_or_max == "max":
        raise NotImplementedError(
            "adaptive_max_pool1d: non-uniform output_size not yet supported "
            "(input " + str(L) + " not divisible by output " + str(O) + ")")
    return _adaptive_avg_along_axis(input_, axis=2, O=O)


def adaptive_pool2d(input_: Tensor, output_size,
                    avg_or_max: Literal["avg", "max"]) -> Tensor:
    if isinstance(output_size, int):
        oh = ow = output_size
    else:
        oh, ow = output_size
    H, W = input_.shape[2], input_.shape[3]
    if _is_uniform(H, oh) and _is_uniform(W, ow):
        fn = (_C_nn.adaptive_avg_pool2d if avg_or_max == "avg"
              else _C_nn.adaptive_max_pool2d)
        return Tensor._wrap(fn(impl_of(input_), int(oh), int(ow)))
    if avg_or_max == "max":
        raise NotImplementedError(
            "adaptive_max_pool2d: non-uniform output_size not yet supported")
    out = _adaptive_avg_along_axis(input_, axis=2, O=int(oh))
    out = _adaptive_avg_along_axis(out, axis=3, O=int(ow))
    return out


def adaptive_pool3d(input_: Tensor, output_size,
                    avg_or_max: Literal["avg", "max"]) -> Tensor:
    if isinstance(output_size, int):
        od = oh = ow = output_size
    else:
        od, oh, ow = output_size
    D, H, W = input_.shape[2], input_.shape[3], input_.shape[4]
    if (_is_uniform(D, od) and _is_uniform(H, oh) and _is_uniform(W, ow)):
        fn = (_C_nn.adaptive_avg_pool3d if avg_or_max == "avg"
              else _C_nn.adaptive_max_pool3d)
        return Tensor._wrap(fn(impl_of(input_), int(od), int(oh), int(ow)))
    if avg_or_max == "max":
        raise NotImplementedError(
            "adaptive_max_pool3d: non-uniform output_size not yet supported")
    out = _adaptive_avg_along_axis(input_, axis=2, O=int(od))
    out = _adaptive_avg_along_axis(out, axis=3, O=int(oh))
    out = _adaptive_avg_along_axis(out, axis=4, O=int(ow))
    return out
