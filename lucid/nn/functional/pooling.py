"""
nn.functional pooling operations.
"""

from typing import Callable, TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _int_or_tuple(v: int | tuple[int, ...], n: int) -> tuple[int, ...]:
    return (v,) * n if isinstance(v, int) else tuple(v)


def _check_return_indices(return_indices: bool, op_name: str) -> None:
    """Reject ``return_indices=True`` with a clear error.

    The engine pool ops do not yet emit per-window argmax indices, so
    silently dropping the request would desync ``MaxUnpool`` and similar
    consumers.  Surface the gap explicitly.
    """
    if return_indices:
        raise NotImplementedError(
            f"{op_name}: return_indices=True is not supported yet. "
            "Compute argmax-style indices manually if needed."
        )


def _adaptive_pool_python_avg(x: "Tensor", output_size: tuple[int, ...]) -> "Tensor":
    """Python fallback for adaptive average pooling with non-divisible sizes.

    Computes per-output-slot mean over ``input[..., start:end]`` where
    ``start = floor(i * Hin / Hout)`` and ``end = ceil((i+1) * Hin / Hout)``,
    matching the reference framework's contract.  Used when the engine
    declines because input dims aren't divisible by output dims.
    """
    import lucid as _lucid
    import numpy as _np

    n_spatial: int = len(output_size)
    in_spatial: tuple[int, ...] = tuple(int(s) for s in x.shape[-n_spatial:])

    # Pre-compute per-axis (start, end) ranges.
    ranges: list[list[tuple[int, int]]] = []
    for ax in range(n_spatial):
        in_d: int = in_spatial[ax]
        out_d: int = int(output_size[ax])
        per_axis: list[tuple[int, int]] = []
        for i in range(out_d):
            start: int = (i * in_d) // out_d
            # Match reference framework exactly: end uses ceil((i+1)·Hin/Hout).
            end: int = -(-(i + 1) * in_d // out_d)
            per_axis.append((start, end))
        ranges.append(per_axis)

    # Build the output via Python loop — operates only on output_size
    # slots so this stays small for typical adaptive pools.
    out_arr: _np.ndarray = _np.zeros(
        tuple(int(s) for s in x.shape[:-n_spatial]) + tuple(output_size),
        dtype=_np.float32,
    )
    x_np: _np.ndarray = x.numpy()
    if n_spatial == 1:
        for i in range(output_size[0]):
            s, e = ranges[0][i]
            out_arr[..., i] = x_np[..., s:e].mean(axis=-1)
    elif n_spatial == 2:
        for i in range(output_size[0]):
            si, ei = ranges[0][i]
            for j in range(output_size[1]):
                sj, ej = ranges[1][j]
                out_arr[..., i, j] = x_np[..., si:ei, sj:ej].mean(axis=(-2, -1))
    else:  # 3D
        for i in range(output_size[0]):
            si, ei = ranges[0][i]
            for j in range(output_size[1]):
                sj, ej = ranges[1][j]
                for k in range(output_size[2]):
                    sk, ek = ranges[2][k]
                    out_arr[..., i, j, k] = x_np[..., si:ei, sj:ej, sk:ek].mean(
                        axis=(-3, -2, -1)
                    )
    return _lucid.tensor(out_arr, dtype=x.dtype, device=x.device)


def _adaptive_avg_call(
    x: Tensor,
    output_size: tuple[int, ...],
    engine_fn: Callable[..., _C_engine.TensorImpl],
) -> Tensor:
    """Engine call with Python fallback when the input dims aren't divisible."""
    n_spatial: int = len(output_size)
    in_spatial: tuple[int, ...] = tuple(int(s) for s in x.shape[-n_spatial:])
    if all(in_spatial[i] % int(output_size[i]) == 0 for i in range(n_spatial)):
        return _wrap(engine_fn(_unwrap(x), *output_size))
    return _adaptive_pool_python_avg(x, output_size)


def max_pool1d(
    x: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """1D max pooling."""
    _check_return_indices(return_indices, "max_pool1d")
    k = _int_or_tuple(kernel_size, 1)[0]
    s = k if stride is None else _int_or_tuple(stride, 1)[0]
    p = _int_or_tuple(padding, 1)[0]
    d = _int_or_tuple(dilation, 1)[0]
    return _wrap(_C_engine.nn.max_pool1d(_unwrap(x), k, s, p))


def max_pool2d(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """2D max pooling."""
    _check_return_indices(return_indices, "max_pool2d")
    kh, kw = _int_or_tuple(kernel_size, 2)
    sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 2)
    ph, pw = _int_or_tuple(padding, 2)
    dh, dw = _int_or_tuple(dilation, 2)
    return _wrap(_C_engine.nn.max_pool2d(_unwrap(x), kh, kw, sh, sw, ph, pw))


def avg_pool1d(
    x: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """1D average pooling."""
    k = _int_or_tuple(kernel_size, 1)[0]
    s = k if stride is None else _int_or_tuple(stride, 1)[0]
    p = _int_or_tuple(padding, 1)[0]
    return _wrap(_C_engine.nn.avg_pool1d(_unwrap(x), k, s, p))


def avg_pool2d(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """2D average pooling."""
    kh, kw = _int_or_tuple(kernel_size, 2)
    sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 2)
    ph, pw = _int_or_tuple(padding, 2)
    return _wrap(_C_engine.nn.avg_pool2d(_unwrap(x), kh, kw, sh, sw, ph, pw))


def adaptive_avg_pool1d(x: Tensor, output_size: int | tuple[int, ...]) -> Tensor:
    """1D adaptive average pooling."""
    sz: tuple[int, ...] = (_int_or_tuple(output_size, 1)[0],)
    return _adaptive_avg_call(x, sz, _C_engine.nn.adaptive_avg_pool1d)


def adaptive_avg_pool2d(x: Tensor, output_size: int | tuple[int, int]) -> Tensor:
    """2D adaptive average pooling."""
    oh, ow = _int_or_tuple(output_size, 2)
    return _adaptive_avg_call(x, (oh, ow), _C_engine.nn.adaptive_avg_pool2d)


def adaptive_max_pool2d(
    x: Tensor,
    output_size: int | tuple[int, int],
    return_indices: bool = False,
) -> Tensor:
    """2D adaptive max pooling."""
    _check_return_indices(return_indices, "adaptive_max_pool2d")
    oh, ow = _int_or_tuple(output_size, 2)
    return _wrap(_C_engine.nn.adaptive_max_pool2d(_unwrap(x), oh, ow))


def adaptive_max_pool1d(
    x: Tensor,
    output_size: int | tuple[int, ...],
    return_indices: bool = False,
) -> Tensor:
    """1D adaptive max pooling."""
    _check_return_indices(return_indices, "adaptive_max_pool1d")
    sz = _int_or_tuple(output_size, 1)[0]
    return _wrap(_C_engine.nn.adaptive_max_pool1d(_unwrap(x), sz))


def adaptive_max_pool3d(
    x: Tensor,
    output_size: int | tuple[int, int, int],
    return_indices: bool = False,
) -> Tensor:
    """3D adaptive max pooling."""
    _check_return_indices(return_indices, "adaptive_max_pool3d")
    od, oh, ow = _int_or_tuple(output_size, 3)
    return _wrap(_C_engine.nn.adaptive_max_pool3d(_unwrap(x), od, oh, ow))


def adaptive_avg_pool3d(
    x: Tensor,
    output_size: int | tuple[int, int, int],
) -> Tensor:
    """3D adaptive average pooling."""
    od, oh, ow = _int_or_tuple(output_size, 3)
    return _adaptive_avg_call(x, (od, oh, ow), _C_engine.nn.adaptive_avg_pool3d)


def max_pool3d(
    x: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """3D max pooling."""
    _check_return_indices(return_indices, "max_pool3d")
    kd, kh, kw = _int_or_tuple(kernel_size, 3)
    sd, sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 3)
    pd, ph, pw = _int_or_tuple(padding, 3)
    return _wrap(
        _C_engine.nn.max_pool3d(_unwrap(x), kd, kh, kw, sd, sh, sw, pd, ph, pw)
    )


def avg_pool3d(
    x: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """3D average pooling."""
    kd, kh, kw = _int_or_tuple(kernel_size, 3)
    sd, sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 3)
    pd, ph, pw = _int_or_tuple(padding, 3)
    return _wrap(
        _C_engine.nn.avg_pool3d(_unwrap(x), kd, kh, kw, sd, sh, sw, pd, ph, pw)
    )
