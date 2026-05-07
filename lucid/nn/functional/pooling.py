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


# ── P3 fills: lp_pool / max_unpool ──────────────────────────────────────────


def _lp_pool(
    x: Tensor,
    norm_type: float,
    avg_pool_fn: Callable[..., Tensor],
    *,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None,
    ceil_mode: bool,
    n: int,
) -> Tensor:
    """Shared body of ``lp_pool1d`` and ``lp_pool2d``.

    ``Lp_pool(x) = (avg_pool(|x|^p) · K) ^ (1/p)`` where ``K`` is the
    pool window size.  We compute ``avg_pool`` on ``|x|^p``, multiply
    back by ``K`` to undo the averaging (so we get a sum), then take
    the ``p``-th root.  ``ceil_mode`` is forwarded to the underlying
    ``avg_pool*`` call where supported.
    """
    import lucid as _l

    p = float(norm_type)
    if p <= 0.0:
        raise ValueError(f"lp_pool: norm_type must be > 0, got {p}")
    abs_pow = _l.abs(x) ** p
    K = 1
    for k in _int_or_tuple(kernel_size, n):
        K *= int(k)
    pooled = avg_pool_fn(abs_pow, kernel_size=kernel_size, stride=stride)
    summed = pooled * float(K)
    return summed ** (1.0 / p)


def lp_pool1d(
    x: Tensor,
    norm_type: float,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    """1-D power-average pooling: ``(avg_pool1d(|x|^p) · K)^(1/p)``."""
    return _lp_pool(
        x, norm_type, avg_pool1d,
        kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode, n=1,
    )


def lp_pool2d(
    x: Tensor,
    norm_type: float,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    """2-D power-average pooling: ``(avg_pool2d(|x|^p) · K)^(1/p)``."""
    return _lp_pool(
        x, norm_type, avg_pool2d,
        kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode, n=2,
    )


def _scatter_unpool(
    x: Tensor,
    indices: Tensor,
    output_spatial: tuple[int, ...],
    n_spatial: int,
) -> Tensor:
    """Shared body of ``max_unpool{1,2,3}d``.

    Scatters the values in ``x`` at the flat positions given by
    ``indices`` (the per-window argmax indices saved by the matching
    ``max_pool*d`` call) into a zero tensor whose spatial shape is
    ``output_spatial``.  Leading batch + channel dims are preserved.
    Implemented via ``scatter_add`` over a flattened spatial axis;
    ``scatter_add`` is differentiable (gradient flows back to ``x``).
    """
    import lucid as _l

    if x.shape != indices.shape:
        raise ValueError(
            f"max_unpool: expected input and indices shapes to match, got "
            f"{tuple(x.shape)} vs {tuple(indices.shape)}"
        )
    leading = list(x.shape[:-n_spatial])
    spatial_numel = 1
    for s in output_spatial:
        spatial_numel *= int(s)
    out_flat_shape = leading + [spatial_numel]
    zeros = _l.zeros(*out_flat_shape, dtype=x.dtype, device=x.device)

    # Flatten the trailing spatial dims of ``x`` and ``indices`` so that
    # ``scatter_add`` works on a single 1-D axis.
    x_flat_shape = leading + [int(x.shape[-n_spatial:].numel()) if hasattr(x.shape[-n_spatial:], "numel") else 1]
    flat_count = 1
    for s in x.shape[-n_spatial:]:
        flat_count *= int(s)
    x_flat = x.reshape(*leading, flat_count)
    idx_flat = indices.reshape(*leading, flat_count)

    out = zeros.scatter_add(-1, idx_flat, x_flat)
    return out.reshape(*leading, *output_spatial)


def max_unpool1d(
    x: Tensor,
    indices: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    output_size: tuple[int, ...] | None = None,
) -> Tensor:
    """Inverse of ``max_pool1d`` — scatters ``x`` into a zero tensor at
    the positions saved by the corresponding ``max_pool1d(...,
    return_indices=True)`` call.

    ``output_size`` is required (Lucid does not infer it from
    ``kernel_size`` / ``stride`` / ``padding`` since the engine pool ops
    don't yet expose a return-indices path that would let us round-trip
    the original spatial shape).  Pass the original input's last
    dimension(s).
    """
    if output_size is None:
        raise ValueError(
            "max_unpool1d: output_size is required (engine return-indices "
            "is not yet wired so we can't infer it)."
        )
    spatial = output_size[-1:] if len(output_size) > 1 else (output_size[0],)
    return _scatter_unpool(x, indices, tuple(int(s) for s in spatial), n_spatial=1)


def max_unpool2d(
    x: Tensor,
    indices: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    output_size: tuple[int, ...] | None = None,
) -> Tensor:
    """Inverse of ``max_pool2d`` — see :func:`max_unpool1d`."""
    if output_size is None:
        raise ValueError("max_unpool2d: output_size is required.")
    spatial = tuple(int(s) for s in output_size[-2:])
    return _scatter_unpool(x, indices, spatial, n_spatial=2)


def max_unpool3d(
    x: Tensor,
    indices: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    output_size: tuple[int, ...] | None = None,
) -> Tensor:
    """Inverse of ``max_pool3d`` — see :func:`max_unpool1d`."""
    if output_size is None:
        raise ValueError("max_unpool3d: output_size is required.")
    spatial = tuple(int(s) for s in output_size[-3:])
    return _scatter_unpool(x, indices, spatial, n_spatial=3)


def fractional_max_pool2d(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    output_size: tuple[int, int] | None = None,
    output_ratio: tuple[float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor:
    """Fractional max-pool 2-D (Graham 2014).

    Not yet implemented — fractional pooling needs a per-batch random
    window-position generator.  Filed as a follow-up to keep this layer
    consistent with the rest of the pooling surface (which currently
    reaches the engine via fixed-stride kernels).
    """
    raise NotImplementedError(
        "fractional_max_pool2d is not yet wired.  Track parity gap §2 — the "
        "implementation needs the engine's pool kernels to emit per-window "
        "indices, which they don't currently do."
    )


def fractional_max_pool3d(
    x: Tensor,
    kernel_size: int | tuple[int, int, int],
    output_size: tuple[int, int, int] | None = None,
    output_ratio: tuple[float, float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor:
    """Fractional max-pool 3-D — see :func:`fractional_max_pool2d`."""
    raise NotImplementedError(
        "fractional_max_pool3d is not yet wired.  Same blocker as the 2-D "
        "form — engine pool kernels need a return-indices path."
    )
