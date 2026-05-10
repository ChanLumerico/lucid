"""
nn.functional pooling operations.
"""

from typing import Callable, TYPE_CHECKING

import lucid as _lucid
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


def _adaptive_pool_python_avg(x: Tensor, output_size: tuple[int, ...]) -> Tensor:
    """Engine fallback for adaptive average pooling with non-divisible sizes.

    Computes per-output-slot mean over ``input[..., start:end]`` where
    ``start = floor(i * Hin / Hout)`` and ``end = ceil((i+1) * Hin / Hout)``,
    matching the reference framework's contract.  Iterates per output slot
    via ``narrow`` + ``mean`` — each step is an engine op so the result
    stays on the original device with no host round-trip.
    """
    n_spatial: int = len(output_size)
    in_spatial: tuple[int, ...] = tuple(int(s) for s in x.shape[-n_spatial:])
    ndim: int = x.ndim

    def _ranges(ax: int) -> list[tuple[int, int]]:
        in_d: int = in_spatial[ax]
        out_d: int = int(output_size[ax])
        out: list[tuple[int, int]] = []
        for i in range(out_d):
            start: int = (i * in_d) // out_d
            # Reference contract: end uses ceil((i+1)·Hin/Hout).
            end: int = -(-(i + 1) * in_d // out_d)
            out.append((start, end))
        return out

    # Convert spatial axis index (0..n_spatial-1) to absolute dim in ``x``.
    def _abs(ax: int) -> int:
        return ndim - n_spatial + ax

    if n_spatial == 1:
        cols: list[Tensor] = []
        for s, e in _ranges(0):
            cols.append(x.narrow(_abs(0), s, e - s).mean(dim=_abs(0)))
        # Each ``cols[i]`` has shape == ``x.shape[:-1]``; stack along last dim.
        return _lucid.stack(cols, dim=-1)

    if n_spatial == 2:
        rows: list[Tensor] = []
        for si, ei in _ranges(0):
            slab: Tensor = x.narrow(_abs(0), si, ei - si)
            cols2: list[Tensor] = []
            for sj, ej in _ranges(1):
                pane: Tensor = slab.narrow(_abs(1), sj, ej - sj)
                cols2.append(pane.mean(dim=(_abs(0), _abs(1))))
            rows.append(_lucid.stack(cols2, dim=-1))
        return _lucid.stack(rows, dim=-2)

    # n_spatial == 3.
    planes: list[Tensor] = []
    for si, ei in _ranges(0):
        slab_i: Tensor = x.narrow(_abs(0), si, ei - si)
        rows3: list[Tensor] = []
        for sj, ej in _ranges(1):
            slab_ij: Tensor = slab_i.narrow(_abs(1), sj, ej - sj)
            cols3: list[Tensor] = []
            for sk, ek in _ranges(2):
                cube: Tensor = slab_ij.narrow(_abs(2), sk, ek - sk)
                cols3.append(cube.mean(dim=(_abs(0), _abs(1), _abs(2))))
            rows3.append(_lucid.stack(cols3, dim=-1))
        planes.append(_lucid.stack(rows3, dim=-2))
    return _lucid.stack(planes, dim=-3)


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
    p = float(norm_type)
    if p <= 0.0:
        raise ValueError(f"lp_pool: norm_type must be > 0, got {p}")
    abs_pow = _lucid.abs(x) ** p
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
        x,
        norm_type,
        avg_pool1d,
        kernel_size=kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
        n=1,
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
        x,
        norm_type,
        avg_pool2d,
        kernel_size=kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
        n=2,
    )


def lp_pool3d(
    x: Tensor,
    norm_type: float,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    """3-D power-average pooling: ``(avg_pool3d(|x|^p) · K)^(1/p)``."""
    return _lp_pool(
        x,
        norm_type,
        avg_pool3d,
        kernel_size=kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
        n=3,
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
    zeros = _lucid.zeros(*out_flat_shape, dtype=x.dtype, device=x.device)

    # Flatten the trailing spatial dims of ``x`` and ``indices`` so that
    # ``scatter_add`` works on a single 1-D axis.
    x_flat_shape = leading + [
        (
            int(x.shape[-n_spatial:].numel())  # type: ignore[attr-defined]
            if hasattr(x.shape[-n_spatial:], "numel")
            else 1
        )
    ]
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


def _frac_pool_starts(
    size: int, kernel: int, out_size: int, sample: float
) -> list[int]:
    """Fractional window start positions for one spatial dimension (Graham 2014 §3).

    ``alpha = (size - kernel) / (out_size - 1)``
    ``start[i] = floor((i + sample) * alpha) - floor(sample * alpha)``
    Last index always clamps to ``size - kernel`` to avoid out-of-bounds.
    """
    if out_size == 1:
        return [size - kernel]
    alpha = (size - kernel) / (out_size - 1)
    base = int(sample * alpha)
    starts: list[int] = []
    for i in range(out_size):
        if i == out_size - 1:
            starts.append(size - kernel)
        else:
            starts.append(int((i + sample) * alpha) - base)
    return starts


def fractional_max_pool2d(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    output_size: int | tuple[int, int] | None = None,
    output_ratio: float | tuple[float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor:
    """Fractional max-pooling over a 2-D spatial input (Graham 2014).

    Implemented as a pure-Python composite: no engine change needed.
    For each (batch, channel) pair, random pool boundaries are drawn
    from ``_random_samples`` (shape ``(N, C, 2)``), then each output
    cell takes the max over the corresponding input window.  Gradients
    flow through ``Tensor.max()`` which already has a registered backward.

    When ``return_indices=True``, returns ``(output, indices)`` where
    ``indices`` holds the flat ``H × W`` position of each pooled maximum.
    """
    if x.ndim != 4:
        raise ValueError(
            f"fractional_max_pool2d expects 4-D input (N, C, H, W), "
            f"got shape {tuple(x.shape)}"
        )
    kH, kW = _int_or_tuple(kernel_size, 2)
    N, C, H, W = (
        int(x.shape[0]),
        int(x.shape[1]),
        int(x.shape[2]),
        int(x.shape[3]),
    )

    if output_size is not None and output_ratio is not None:
        raise ValueError(
            "fractional_max_pool2d: specify output_size or output_ratio, not both"
        )
    if output_size is not None:
        oH, oW = _int_or_tuple(output_size, 2)
    elif output_ratio is not None:
        rH, rW = (
            (output_ratio, output_ratio)
            if isinstance(output_ratio, float)
            else (float(output_ratio[0]), float(output_ratio[1]))
        )
        oH, oW = max(1, int(H * rH)), max(1, int(W * rW))
    else:
        raise ValueError(
            "fractional_max_pool2d: one of output_size or output_ratio must be given"
        )

    if _random_samples is None:
        _random_samples = _lucid.rand(N, C, 2, dtype=_lucid.float32, device=x.device)

    batch_out: list[Tensor] = []
    batch_idx: list[Tensor] = []

    for n in range(N):
        chan_out: list[Tensor] = []
        chan_idx: list[Tensor] = []

        for c in range(C):
            sh = float(_random_samples[n, c, 0].item())
            sw = float(_random_samples[n, c, 1].item())
            h_starts = _frac_pool_starts(H, kH, oH, sh)
            w_starts = _frac_pool_starts(W, kW, oW, sw)

            plane_out: list[Tensor] = []
            plane_idx: list[Tensor] = []

            for hs in h_starts:
                row_out: list[Tensor] = []
                row_idx: list[Tensor] = []
                for ws in w_starts:
                    patch = x[n, c, hs : hs + kH, ws : ws + kW].reshape(-1)
                    row_out.append(patch.max().unsqueeze(0))
                    if return_indices:
                        li = int(patch.argmax().item())
                        lr, lc = divmod(li, kW)
                        flat = (hs + lr) * W + (ws + lc)
                        row_idx.append(
                            _lucid.tensor([flat], dtype=_lucid.int64, device=x.device)
                        )
                plane_out.append(_lucid.cat(row_out))  # (oW,)
                if return_indices:
                    plane_idx.append(_lucid.cat(row_idx))

            chan_out.append(_lucid.stack(plane_out))  # (oH, oW)
            if return_indices:
                chan_idx.append(_lucid.stack(plane_idx))

        batch_out.append(_lucid.stack(chan_out))  # (C, oH, oW)
        if return_indices:
            batch_idx.append(_lucid.stack(chan_idx))

    out = _lucid.stack(batch_out)  # (N, C, oH, oW)
    if return_indices:
        return out, _lucid.stack(batch_idx)  # type: ignore[return-value]
    return out


def fractional_max_pool3d(
    x: Tensor,
    kernel_size: int | tuple[int, int, int],
    output_size: int | tuple[int, int, int] | None = None,
    output_ratio: float | tuple[float, float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor:
    """Fractional max-pooling over a 3-D spatial input (Graham 2014).

    Extends :func:`fractional_max_pool2d` by one depth dimension.
    ``_random_samples`` has shape ``(N, C, 3)`` — one sample per spatial axis.
    When ``return_indices=True``, returns ``(output, indices)`` where
    ``indices`` holds the flat ``D × H × W`` position of each max.
    """
    if x.ndim != 5:
        raise ValueError(
            f"fractional_max_pool3d expects 5-D input (N, C, D, H, W), "
            f"got shape {tuple(x.shape)}"
        )
    kD, kH, kW = _int_or_tuple(kernel_size, 3)
    N, C, D, H, W = (
        int(x.shape[0]),
        int(x.shape[1]),
        int(x.shape[2]),
        int(x.shape[3]),
        int(x.shape[4]),
    )

    if output_size is not None and output_ratio is not None:
        raise ValueError(
            "fractional_max_pool3d: specify output_size or output_ratio, not both"
        )
    if output_size is not None:
        oD, oH, oW = _int_or_tuple(output_size, 3)
    elif output_ratio is not None:
        if isinstance(output_ratio, float):
            rD = rH = rW = output_ratio
        else:
            rD = float(output_ratio[0])
            rH = float(output_ratio[1])
            rW = float(output_ratio[2])
        oD = max(1, int(D * rD))
        oH = max(1, int(H * rH))
        oW = max(1, int(W * rW))
    else:
        raise ValueError(
            "fractional_max_pool3d: one of output_size or output_ratio must be given"
        )

    if _random_samples is None:
        _random_samples = _lucid.rand(N, C, 3, dtype=_lucid.float32, device=x.device)

    batch_out: list[Tensor] = []
    batch_idx: list[Tensor] = []

    for n in range(N):
        chan_out: list[Tensor] = []
        chan_idx: list[Tensor] = []

        for c in range(C):
            sd = float(_random_samples[n, c, 0].item())
            sh = float(_random_samples[n, c, 1].item())
            sw = float(_random_samples[n, c, 2].item())
            d_starts = _frac_pool_starts(D, kD, oD, sd)
            h_starts = _frac_pool_starts(H, kH, oH, sh)
            w_starts = _frac_pool_starts(W, kW, oW, sw)

            vol_out: list[Tensor] = []
            vol_idx: list[Tensor] = []

            for ds in d_starts:
                plane_out: list[Tensor] = []
                plane_idx: list[Tensor] = []
                for hs in h_starts:
                    row_out: list[Tensor] = []
                    row_idx: list[Tensor] = []
                    for ws in w_starts:
                        patch = x[
                            n, c, ds : ds + kD, hs : hs + kH, ws : ws + kW
                        ].reshape(-1)
                        row_out.append(patch.max().unsqueeze(0))
                        if return_indices:
                            li = int(patch.argmax().item())
                            ld = li // (kH * kW)
                            lr = (li % (kH * kW)) // kW
                            lc = li % kW
                            flat = ((ds + ld) * H + (hs + lr)) * W + (ws + lc)
                            row_idx.append(
                                _lucid.tensor(
                                    [flat], dtype=_lucid.int64, device=x.device
                                )
                            )
                    plane_out.append(_lucid.cat(row_out))  # (oW,)
                    if return_indices:
                        plane_idx.append(_lucid.cat(row_idx))
                vol_out.append(_lucid.stack(plane_out))  # (oH, oW)
                if return_indices:
                    vol_idx.append(_lucid.stack(plane_idx))

            chan_out.append(_lucid.stack(vol_out))  # (oD, oH, oW)
            if return_indices:
                chan_idx.append(_lucid.stack(vol_idx))

        batch_out.append(_lucid.stack(chan_out))  # (C, oD, oH, oW)
        if return_indices:
            batch_idx.append(_lucid.stack(chan_idx))

    out = _lucid.stack(batch_out)  # (N, C, oD, oH, oW)
    if return_indices:
        return out, _lucid.stack(batch_idx)  # type: ignore[return-value]
    return out
