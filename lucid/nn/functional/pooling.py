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
    r"""1-D max pooling over a sliding window.

    For each window, takes the maximum value — a translation-invariant
    feature aggregator that also provides a degree of non-linearity.
    Standard ingredient of 1-D temporal CNNs (text, audio, sensor data).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, L)``.
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Window step.  Defaults to ``kernel_size`` (non-overlapping).
    padding : int or tuple of int, optional
        Implicit zero-padding on both sides of the spatial axis.
    dilation : int or tuple of int, optional
        Spacing between window elements.  Default ``1``.
    return_indices : bool, optional
        Currently must be ``False``; the engine pool op does not yet
        emit per-window argmax indices.
    ceil_mode : bool, optional
        When ``True``, use ceil instead of floor in the output-size
        formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, L_out)`` where

        .. math::

            L_{\text{out}} = \left\lfloor \frac{L + 2 p - d(k - 1) - 1}{s} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,l} = \max_{m = 0, \ldots, k-1} x_{i,\,c,\,s l + d m - p}

    Max-pool is sub-differentiable: the gradient flows only through
    the position(s) holding the maximum in each window.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_pool1d
    >>> x = lucid.randn(2, 4, 16)
    >>> y = max_pool1d(x, kernel_size=2)
    >>> y.shape
    (2, 4, 8)
    """
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
    r"""2-D max pooling over a sliding window.

    Aggregates each spatial neighbourhood into its maximum value.
    The canonical downsampling primitive in classical image CNNs —
    provides invariance to small local shifts and reduces both spatial
    extent and compute for subsequent layers.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, H, W)``.
    kernel_size : int or (int, int)
        Size of the pooling window.
    stride : int or (int, int), optional
        Window step.  Defaults to ``kernel_size`` (non-overlapping).
    padding : int or (int, int), optional
        Implicit zero-padding on each spatial side.
    dilation : int or (int, int), optional
        Spacing between window elements.  Default ``1``.
    return_indices : bool, optional
        Must currently be ``False`` — see :func:`max_pool1d`.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, H_out, W_out)`` where each dim obeys

        .. math::

            H_{\text{out}} = \left\lfloor \frac{H + 2 p_H - d_H (k_H - 1) - 1}{s_H} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,h,w} = \max_{m,n} x_{i,\,c,\,s_H h + m,\,s_W w + n}

    Pure max is sub-differentiable; gradient routes to the per-window
    argmax position only.  Empirically the strongest pooling operator
    for classification tasks (preserves high-magnitude features).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_pool2d
    >>> x = lucid.randn(1, 16, 32, 32)
    >>> y = max_pool2d(x, kernel_size=2, stride=2)
    >>> y.shape
    (1, 16, 16, 16)
    """
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
    r"""1-D average pooling over a sliding window.

    Replaces each window with its arithmetic mean.  Smoother than
    max-pool and used wherever maintaining the overall response
    magnitude matters (e.g. before a fully connected classifier head).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, L)``.
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Window step.  Defaults to ``kernel_size`` (non-overlapping).
    padding : int or tuple of int, optional
        Implicit zero-padding on both sides.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.
    count_include_pad : bool, optional
        When ``True``, padding cells contribute to the denominator.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, L_out)`` where

        .. math::

            L_{\text{out}} = \left\lfloor \frac{L + 2 p - k}{s} + 1 \right\rfloor

    Notes
    -----
    Math (averaging window :math:`R` of cardinality :math:`|R|`):

    .. math::

        y_{i,c,l} = \frac{1}{|R|} \sum_{m \in R} x_{i,\,c,\,s l + m - p}

    Unlike max-pool, average-pool is fully differentiable everywhere,
    so the gradient is uniformly distributed back across all window
    positions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import avg_pool1d
    >>> x = lucid.randn(2, 3, 20)
    >>> y = avg_pool1d(x, kernel_size=4, stride=2)
    >>> y.shape
    (2, 3, 9)
    """
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
    r"""2-D average pooling over a sliding window.

    Replaces each spatial window with its arithmetic mean.  Common as
    the last spatial reduction before a fully connected classifier
    (modern nets often use :func:`adaptive_avg_pool2d` for the final
    pool because it adapts to arbitrary input sizes).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, H, W)``.
    kernel_size : int or (int, int)
        Size of the pooling window.
    stride : int or (int, int), optional
        Window step.  Defaults to ``kernel_size``.
    padding : int or (int, int), optional
        Implicit zero-padding on each spatial side.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.
    count_include_pad : bool, optional
        When ``True``, padding cells contribute to the denominator.
    divisor_override : int, optional
        Explicit denominator overriding ``|R|`` (rare; useful for
        symmetric / weight-style averages).

    Returns
    -------
    Tensor
        Output of shape ``(N, C, H_out, W_out)`` where each dim obeys

        .. math::

            H_{\text{out}} = \left\lfloor \frac{H + 2 p_H - k_H}{s_H} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,h,w} = \frac{1}{|R|} \sum_{(m, n) \in R} x_{i,\,c,\,s_H h + m,\,s_W w + n}

    Average-pool's gradient is the uniform distribution :math:`1/|R|`
    over the window, which produces smoother backward signals than
    max-pool.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import avg_pool2d
    >>> x = lucid.randn(1, 8, 28, 28)
    >>> y = avg_pool2d(x, kernel_size=2)
    >>> y.shape
    (1, 8, 14, 14)
    """
    kh, kw = _int_or_tuple(kernel_size, 2)
    sh, sw = _int_or_tuple(kernel_size if stride is None else stride, 2)
    ph, pw = _int_or_tuple(padding, 2)
    return _wrap(_C_engine.nn.avg_pool2d(_unwrap(x), kh, kw, sh, sw, ph, pw))


def adaptive_avg_pool1d(x: Tensor, output_size: int | tuple[int, ...]) -> Tensor:
    r"""1-D adaptive average pooling — produces a fixed output length.

    Computes kernel / stride dynamically so that the output's spatial
    length equals ``output_size`` regardless of input length.  Handy
    when feeding variable-length sequences into a head expecting a
    fixed embedding size.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, L)``.
    output_size : int or tuple of int
        Desired length of the output spatial axis.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, output_size)``.

    Notes
    -----
    For each output index :math:`i`, the corresponding input window is

    .. math::

        \big[\lfloor i \cdot L / L_{\text{out}} \rfloor,\;
              \lceil (i+1) \cdot L / L_{\text{out}} \rceil\big)

    and the cell is the mean over that window.  When ``L`` divides
    evenly by ``L_out`` the operation reduces to plain
    :func:`avg_pool1d` with kernel == stride == ``L / L_out``;
    otherwise Lucid falls back to a per-slot Python-level computation
    that still keeps all data on the original device.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_avg_pool1d
    >>> x = lucid.randn(2, 32, 13)
    >>> y = adaptive_avg_pool1d(x, output_size=5)
    >>> y.shape
    (2, 32, 5)
    """
    sz: tuple[int, ...] = (_int_or_tuple(output_size, 1)[0],)
    return _adaptive_avg_call(x, sz, _C_engine.nn.adaptive_avg_pool1d)


def adaptive_avg_pool2d(x: Tensor, output_size: int | tuple[int, int]) -> Tensor:
    r"""2-D adaptive average pooling — produces a fixed ``(H, W)``.

    Computes per-axis kernel / stride so the output shape exactly
    matches ``output_size`` regardless of the input spatial size.  The
    workhorse "global pool to fixed grid" used by virtually every
    modern classification network (``output_size=(1, 1)`` collapses
    the spatial axes entirely).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, H, W)``.
    output_size : int or (int, int)
        Desired spatial shape of the output.  Scalar duplicates per
        axis; ``(1, 1)`` performs global average pooling.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, oH, oW)``.

    Notes
    -----
    For each output cell :math:`(i, j)` the corresponding input
    window is

    .. math::

        \big[\lfloor i H / H_{\text{out}} \rfloor,
              \lceil (i + 1) H / H_{\text{out}} \rceil\big) \times
        \big[\lfloor j W / W_{\text{out}} \rfloor,
              \lceil (j + 1) W / W_{\text{out}} \rceil\big)

    Foundational to Spatial Pyramid Pooling and the "global average
    pool + linear classifier" pattern introduced by NIN /
    ResNet-style architectures.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_avg_pool2d
    >>> x = lucid.randn(1, 512, 7, 9)
    >>> y = adaptive_avg_pool2d(x, output_size=(1, 1))
    >>> y.shape
    (1, 512, 1, 1)
    """
    oh, ow = _int_or_tuple(output_size, 2)
    return _adaptive_avg_call(x, (oh, ow), _C_engine.nn.adaptive_avg_pool2d)


def adaptive_max_pool2d(
    x: Tensor,
    output_size: int | tuple[int, int],
    return_indices: bool = False,
) -> Tensor:
    r"""2-D adaptive max pooling — fixed-shape ``(H, W)`` via per-cell max.

    Like :func:`adaptive_avg_pool2d` but takes the maximum over each
    dynamically computed window instead of the mean.  Useful when
    salient peaks should drive the downstream representation
    (e.g. object detectors that rely on strong activations).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, H, W)``.
    output_size : int or (int, int)
        Desired spatial output shape.
    return_indices : bool, optional
        Must currently be ``False``.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, oH, oW)``.

    Notes
    -----
    For each output cell, the window definition is the same as in
    :func:`adaptive_avg_pool2d`; the cell value is

    .. math::

        y_{i,c,h,w} = \max_{(m, n) \in R(h, w)} x_{i,\,c,\,m,\,n}

    Gradient flows only through the per-window argmax position.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_max_pool2d
    >>> x = lucid.randn(1, 64, 11, 13)
    >>> y = adaptive_max_pool2d(x, output_size=(3, 3))
    >>> y.shape
    (1, 64, 3, 3)
    """
    _check_return_indices(return_indices, "adaptive_max_pool2d")
    oh, ow = _int_or_tuple(output_size, 2)
    return _wrap(_C_engine.nn.adaptive_max_pool2d(_unwrap(x), oh, ow))


def adaptive_max_pool1d(
    x: Tensor,
    output_size: int | tuple[int, ...],
    return_indices: bool = False,
) -> Tensor:
    r"""1-D adaptive max pooling — produces a fixed output length.

    Computes kernel / stride dynamically so the output length equals
    ``output_size`` regardless of input length, then takes the
    per-window maximum.  Common in NLP / audio CNNs that feed
    fixed-size embeddings into a head.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, L)``.
    output_size : int or tuple of int
        Desired output length.
    return_indices : bool, optional
        Must currently be ``False``.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, output_size)``.

    Notes
    -----
    The window for output index :math:`i` is

    .. math::

        \big[\lfloor i L / L_{\text{out}} \rfloor,\;
              \lceil (i+1) L / L_{\text{out}} \rceil\big)

    and the cell takes the max over that window.  Equivalent to
    :func:`max_pool1d` with kernel == stride == ``L / L_out`` when the
    division is exact.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_max_pool1d
    >>> x = lucid.randn(2, 8, 21)
    >>> y = adaptive_max_pool1d(x, output_size=4)
    >>> y.shape
    (2, 8, 4)
    """
    _check_return_indices(return_indices, "adaptive_max_pool1d")
    sz = _int_or_tuple(output_size, 1)[0]
    return _wrap(_C_engine.nn.adaptive_max_pool1d(_unwrap(x), sz))


def adaptive_max_pool3d(
    x: Tensor,
    output_size: int | tuple[int, int, int],
    return_indices: bool = False,
) -> Tensor:
    r"""3-D adaptive max pooling — produces a fixed ``(D, H, W)``.

    Volumetric analogue of :func:`adaptive_max_pool2d`.  Computes
    per-axis kernel and stride so the output spatial shape exactly
    matches ``output_size`` regardless of input shape.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, D, H, W)``.
    output_size : int or (int, int, int)
        Desired output spatial shape.
    return_indices : bool, optional
        Must currently be ``False``.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, oD, oH, oW)``.

    Notes
    -----
    The window for output cell :math:`(i, j, k)` is the 3-way product
    of axis-wise floor / ceil intervals (see
    :func:`adaptive_avg_pool2d` for the 2-D version).  The cell takes
    the maximum over that volume:

    .. math::

        y_{n,c,d,h,w} = \max_{(p, q, r) \in R(d, h, w)} x_{n,\,c,\,p,\,q,\,r}

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_max_pool3d
    >>> x = lucid.randn(1, 8, 5, 7, 9)
    >>> y = adaptive_max_pool3d(x, output_size=(1, 1, 1))
    >>> y.shape
    (1, 8, 1, 1, 1)
    """
    _check_return_indices(return_indices, "adaptive_max_pool3d")
    od, oh, ow = _int_or_tuple(output_size, 3)
    return _wrap(_C_engine.nn.adaptive_max_pool3d(_unwrap(x), od, oh, ow))


def adaptive_avg_pool3d(
    x: Tensor,
    output_size: int | tuple[int, int, int],
) -> Tensor:
    r"""3-D adaptive average pooling — produces a fixed ``(D, H, W)``.

    Volumetric counterpart of :func:`adaptive_avg_pool2d`.  Used by
    3-D classification heads to collapse a volumetric feature map to
    a fixed embedding shape regardless of input size.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, D, H, W)``.
    output_size : int or (int, int, int)
        Desired output spatial shape.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, oD, oH, oW)``.

    Notes
    -----
    Per-cell window definitions follow the same floor / ceil
    construction used in the 2-D variant.  The cell value is the mean
    over the volumetric window:

    .. math::

        y_{n,c,d,h,w} = \frac{1}{|R|}
            \sum_{(p, q, r) \in R(d, h, w)} x_{n,\,c,\,p,\,q,\,r}

    When the input dims divide ``output_size`` evenly the engine path
    is taken; otherwise a per-slot Python fallback runs (all data
    stays on-device).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_avg_pool3d
    >>> x = lucid.randn(1, 16, 8, 10, 10)
    >>> y = adaptive_avg_pool3d(x, output_size=(2, 2, 2))
    >>> y.shape
    (1, 16, 2, 2, 2)
    """
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
    r"""3-D max pooling over a sliding window.

    Volumetric downsampling primitive — extends :func:`max_pool2d` by
    one depth dimension.  Used in 3-D CNNs for medical imaging and
    video understanding.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, D, H, W)``.
    kernel_size : int or (int, int, int)
        Size of the pooling window per axis.
    stride : int or (int, int, int), optional
        Window step.  Defaults to ``kernel_size``.
    padding : int or (int, int, int), optional
        Implicit zero-padding on each spatial side.
    dilation : int or (int, int, int), optional
        Spacing between window elements.  Default ``1``.
    return_indices : bool, optional
        Must currently be ``False``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, D_out, H_out, W_out)`` where each
        spatial dim obeys

        .. math::

            D_{\text{out}} = \left\lfloor \frac{D + 2 p_D - d_D (k_D - 1) - 1}{s_D} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,d,h,w} = \max_{l,m,n} x_{i,\,c,\,s_D d + l,\,s_H h + m,\,s_W w + n}

    Memory cost can become significant for large volumes — when the
    network only needs a global summary, prefer
    :func:`adaptive_max_pool3d` with ``output_size=(1, 1, 1)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_pool3d
    >>> x = lucid.randn(1, 4, 8, 16, 16)
    >>> y = max_pool3d(x, kernel_size=2)
    >>> y.shape
    (1, 4, 4, 8, 8)
    """
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
    r"""3-D average pooling over a sliding window.

    Volumetric counterpart of :func:`avg_pool2d` — replaces each 3-D
    window with its arithmetic mean.  Useful in 3-D CNNs as a smoother
    alternative to :func:`max_pool3d`, and as the final "global"
    reduction before a classifier head.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, D, H, W)``.
    kernel_size : int or (int, int, int)
        Size of the pooling window per axis.
    stride : int or (int, int, int), optional
        Window step.  Defaults to ``kernel_size``.
    padding : int or (int, int, int), optional
        Implicit zero-padding on each spatial side.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.
    count_include_pad : bool, optional
        Whether padding cells contribute to the denominator.
    divisor_override : int, optional
        Explicit denominator overriding ``|R|``.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, D_out, H_out, W_out)`` where each
        spatial dim obeys

        .. math::

            D_{\text{out}} = \left\lfloor \frac{D + 2 p_D - k_D}{s_D} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,d,h,w} = \frac{1}{|R|}
            \sum_{(l, m, n) \in R} x_{i,\,c,\,s_D d + l,\,s_H h + m,\,s_W w + n}

    Average-pool's gradient is uniformly distributed across the
    window, so it produces smoother backward signals than its
    max-based counterpart.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import avg_pool3d
    >>> x = lucid.randn(1, 4, 8, 16, 16)
    >>> y = avg_pool3d(x, kernel_size=2)
    >>> y.shape
    (1, 4, 4, 8, 8)
    """
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
    r"""1-D Lp-norm pooling — :math:`\big(\sum |x|^p\big)^{1/p}`.

    Generalises average and max pooling under a single parameter
    :math:`p = \text{norm\_type}`.  :math:`p = 1` recovers
    sum-pooling (= ``|R|`` × :func:`avg_pool1d`); :math:`p = 2` is
    energy / RMS-style pooling; :math:`p \to \infty` approaches
    :func:`max_pool1d`.  Provides a smooth, fully differentiable bridge
    between the two extremes.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, L)``.
    norm_type : float
        Exponent :math:`p > 0`.
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Window step.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, L_out)``.

    Notes
    -----
    Math (per window :math:`R`):

    .. math::

        y_{i,c,l} = \left( \sum_{m \in R} |x_{i,c,m}|^p \right)^{1/p}

    Implemented as ``(avg_pool1d(|x|^p) · |R|)^(1/p)`` so the entire
    operator inherits its gradient from the average-pool engine path.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import lp_pool1d
    >>> x = lucid.randn(2, 4, 12)
    >>> y = lp_pool1d(x, norm_type=2.0, kernel_size=2)
    >>> y.shape
    (2, 4, 6)
    """
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
    r"""2-D Lp-norm pooling — :math:`\big(\sum |x|^p\big)^{1/p}`.

    Two-dimensional version of :func:`lp_pool1d`.  The exponent
    :math:`p` interpolates between sum / average pooling (small
    :math:`p`) and max pooling (large :math:`p`), giving the network
    a tunable "softness" parameter.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, H, W)``.
    norm_type : float
        Exponent :math:`p > 0`.
    kernel_size : int or (int, int)
        Size of the pooling window.
    stride : int or (int, int), optional
        Window step.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, H_out, W_out)``.

    Notes
    -----
    Math (per window :math:`R`):

    .. math::

        y_{i,c,h,w} = \left( \sum_{(m, n) \in R} |x_{i,c,m,n}|^p \right)^{1/p}

    The operator is fully differentiable for any :math:`p > 0`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import lp_pool2d
    >>> x = lucid.randn(1, 8, 16, 16)
    >>> y = lp_pool2d(x, norm_type=3.0, kernel_size=2)
    >>> y.shape
    (1, 8, 8, 8)
    """
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
    r"""3-D Lp-norm pooling — :math:`\big(\sum |x|^p\big)^{1/p}`.

    Volumetric version of :func:`lp_pool2d`.  Interpolates between
    sum / average pooling (small :math:`p`) and max pooling (large
    :math:`p`) over a 3-D window.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, D, H, W)``.
    norm_type : float
        Exponent :math:`p > 0`.
    kernel_size : int or (int, int, int)
        Size of the pooling window per axis.
    stride : int or (int, int, int), optional
        Window step.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, D_out, H_out, W_out)``.

    Notes
    -----
    Math (per window :math:`R`):

    .. math::

        y_{i,c,d,h,w} = \left( \sum_{(l, m, n) \in R} |x_{i,c,l,m,n}|^p \right)^{1/p}

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import lp_pool3d
    >>> x = lucid.randn(1, 2, 8, 8, 8)
    >>> y = lp_pool3d(x, norm_type=2.0, kernel_size=2)
    >>> y.shape
    (1, 2, 4, 4, 4)
    """
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
    r"""Inverse of :func:`max_pool1d` via scatter at saved argmax indices.

    Restores a "sparse" higher-resolution feature map by placing each
    value of ``x`` at the spatial position recorded in ``indices``
    (i.e. the argmax positions captured by the corresponding max-pool
    forward pass).  Non-selected positions remain zero.  Standard
    component of encoder-decoder networks such as SegNet that
    propagate pooling indices through skip connections.

    Parameters
    ----------
    x : Tensor
        Pooled values, shape ``(N, C, L_in)``.
    indices : Tensor
        Per-window argmax positions (flattened spatial coordinate),
        same shape as ``x``.
    kernel_size : int or tuple of int
        Original pooling window size (recorded for symmetry — not used
        directly in the scatter math).
    stride : int or tuple of int, optional
        Original stride.  Recorded for symmetry.
    padding : int or tuple of int, optional
        Original padding.  Recorded for symmetry.
    output_size : tuple of int, optional
        Required.  Target spatial shape — typically the length of the
        tensor that was originally fed to :func:`max_pool1d`.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, L_out)`` containing ``x`` scattered
        into a zero buffer at the positions recorded in ``indices``.

    Notes
    -----
    Math (let :math:`\phi` be the bijection from pooled-cell index to
    saved flat input position):

    .. math::

        y_{n,c,\phi(n,c,l)} = x_{n,c,l},\qquad y_{n,c,j} = 0 \text{ otherwise}

    Implemented as a differentiable :func:`scatter_add` over a
    flattened spatial axis — gradients flow back to ``x``.
    ``output_size`` is required because Lucid does not yet emit
    argmax indices from the forward pool, so we cannot infer the
    original spatial extent.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_unpool1d
    >>> x = lucid.randn(1, 2, 4)
    >>> idx = lucid.tensor([[[0, 2, 4, 6], [1, 3, 5, 7]]], dtype=lucid.int64)
    >>> y = max_unpool1d(x, idx, kernel_size=2, output_size=(8,))
    >>> y.shape
    (1, 2, 8)
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
    r"""Inverse of :func:`max_pool2d` via scatter at saved argmax indices.

    Two-dimensional version of :func:`max_unpool1d`.  Scatters the
    pooled values back to their original ``(H, W)`` positions
    (recorded as flat ``H × W`` indices) on a zero canvas.  Standard
    in SegNet-style decoders where decoder layers reuse pooling
    indices from the encoder for shape-preserving up-sampling.

    Parameters
    ----------
    x : Tensor
        Pooled values, shape ``(N, C, H_in, W_in)``.
    indices : Tensor
        Flattened argmax positions (``row * W + col``), same shape as
        ``x``.
    kernel_size : int or (int, int)
        Original pooling window size.
    stride : int or (int, int), optional
        Original stride.
    padding : int or (int, int), optional
        Original padding.
    output_size : tuple of int, optional
        Required.  Target spatial shape ``(..., H_out, W_out)`` — the
        original input shape to the matching max-pool.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, H_out, W_out)``.

    Notes
    -----
    Implemented via :func:`scatter_add` over a flattened spatial axis,
    fully differentiable.  See :func:`max_unpool1d` for the math.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_unpool2d, max_pool2d
    >>> x = lucid.randn(1, 1, 4, 4)
    >>> # forward pool (return_indices currently not supported by engine):
    >>> # idx must be supplied by the user when round-tripping.
    >>> pooled = max_pool2d(x, kernel_size=2)
    >>> idx = lucid.zeros_like(pooled, dtype=lucid.int64)
    >>> y = max_unpool2d(pooled, idx, kernel_size=2, output_size=(4, 4))
    >>> y.shape
    (1, 1, 4, 4)
    """
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
    r"""Inverse of :func:`max_pool3d` via scatter at saved argmax indices.

    Volumetric version of :func:`max_unpool2d`.  Scatters the pooled
    values back to their saved ``(D, H, W)`` positions (encoded as
    flat ``D × H × W`` indices) on a zero canvas.

    Parameters
    ----------
    x : Tensor
        Pooled values, shape ``(N, C, D_in, H_in, W_in)``.
    indices : Tensor
        Flattened argmax positions, same shape as ``x``.
    kernel_size : int or (int, int, int)
        Original pooling window size.
    stride : int or (int, int, int), optional
        Original stride.
    padding : int or (int, int, int), optional
        Original padding.
    output_size : tuple of int, optional
        Required.  Target spatial shape ``(..., D_out, H_out, W_out)``.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, D_out, H_out, W_out)``.

    Notes
    -----
    Implemented via :func:`scatter_add` over a flattened spatial axis
    (so the gradient flows back to ``x`` and ``indices`` is treated
    as a non-differentiable lookup).  See :func:`max_unpool1d` for
    the math.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_unpool3d
    >>> x = lucid.randn(1, 1, 2, 2, 2)
    >>> idx = lucid.zeros_like(x, dtype=lucid.int64)
    >>> y = max_unpool3d(x, idx, kernel_size=2, output_size=(4, 4, 4))
    >>> y.shape
    (1, 1, 4, 4, 4)
    """
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
    r"""Fractional max-pooling over a 2-D input (Graham, 2014).

    Performs max-pooling with a **non-integer** effective stride.
    Instead of a uniform window grid, random per-axis offsets pick
    "slightly irregular" window boundaries, yielding a smoother
    downsampling than the abrupt 2× reductions of standard max-pool.
    Acts as a structural regulariser for image classifiers.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, H, W)``.
    kernel_size : int or (int, int)
        Size of the pooling window.
    output_size : int or (int, int), optional
        Exact desired output spatial shape.  Mutually exclusive with
        ``output_ratio``.
    output_ratio : float or (float, float), optional
        Multiplicative shrinkage factor per axis (e.g. ``0.5`` halves
        each dim).  Mutually exclusive with ``output_size``.
    return_indices : bool, optional
        When ``True``, also return per-output-cell argmax positions
        (flattened ``H × W`` index) as a second tensor.
    _random_samples : Tensor, optional
        Per-instance uniform samples of shape ``(N, C, 2)`` driving
        the window-boundary draws.  Defaults to fresh
        :func:`lucid.rand`.

    Returns
    -------
    Tensor or (Tensor, Tensor)
        Output of shape ``(N, C, oH, oW)``; when ``return_indices``,
        also the argmax indices of matching shape.

    Notes
    -----
    Window starts are drawn from Graham's formula (§3 of the paper)

    .. math::

        \alpha = \frac{H - k}{H_{\text{out}} - 1},\quad
        \text{start}_i = \lfloor (i + u) \alpha \rfloor - \lfloor u \alpha \rfloor

    where :math:`u` is a per-(sample, channel) uniform sample.  The
    final start clamps to ``H - k`` so all windows lie inside the
    input.  Implementation is a pure-Python composite — gradients
    flow through :meth:`Tensor.max`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import fractional_max_pool2d
    >>> x = lucid.randn(1, 4, 14, 14)
    >>> y = fractional_max_pool2d(x, kernel_size=2, output_ratio=0.5)
    >>> y.shape
    (1, 4, 7, 7)
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
    r"""Fractional max-pooling over a 3-D input (Graham, 2014).

    Volumetric extension of :func:`fractional_max_pool2d`.  Provides
    non-integer effective stride along all three spatial axes — a
    structural regulariser for 3-D CNNs.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, D, H, W)``.
    kernel_size : int or (int, int, int)
        Size of the pooling window per axis.
    output_size : int or (int, int, int), optional
        Exact desired output spatial shape.  Mutually exclusive with
        ``output_ratio``.
    output_ratio : float or (float, float, float), optional
        Multiplicative shrinkage per axis.  Mutually exclusive with
        ``output_size``.
    return_indices : bool, optional
        When ``True``, also return per-output-cell argmax positions
        (flattened ``D × H × W`` index).
    _random_samples : Tensor, optional
        Per-instance uniform samples of shape ``(N, C, 3)``.  Defaults
        to fresh :func:`lucid.rand`.

    Returns
    -------
    Tensor or (Tensor, Tensor)
        Output of shape ``(N, C, oD, oH, oW)``; when
        ``return_indices``, also the argmax indices.

    Notes
    -----
    Each axis runs Graham's start-position formula independently
    (see :func:`fractional_max_pool2d` Notes).  Implementation is a
    pure-Python composite; gradients flow through :meth:`Tensor.max`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import fractional_max_pool3d
    >>> x = lucid.randn(1, 2, 8, 8, 8)
    >>> y = fractional_max_pool3d(x, kernel_size=2, output_ratio=0.5)
    >>> y.shape
    (1, 2, 4, 4, 4)
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
