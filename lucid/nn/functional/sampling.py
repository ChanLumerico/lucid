"""
nn.functional sampling / interpolation / padding operations.
"""

from typing import TYPE_CHECKING

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def interpolate(
    x: Tensor,
    size: int | tuple[int, ...] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
) -> Tensor:
    r"""Resample an N-D tensor to a target spatial size or scale factor.

    Supports the standard family of image / volume resampling kernels.
    The leading two axes (``N`` batch and ``C`` channels) are preserved;
    only the trailing spatial axes are resampled.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, *spatial)`` where ``len(spatial)`` is 1
        (linear), 2 (bilinear / bicubic / area / nearest-2d) or 3
        (trilinear / nearest-3d).
    size : int or tuple of int, optional
        Target spatial size.  Mutually exclusive with ``scale_factor``.
    scale_factor : float or tuple of float, optional
        Multiplicative factor applied to each spatial axis.  Output size
        is ``floor(in_size * scale_factor)``.
    mode : str, optional
        One of ``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``,
        ``"trilinear"``, ``"area"``, ``"nearest-exact"``.  Default
        ``"nearest"``.
    align_corners : bool, optional
        Only meaningful for ``"linear"`` / ``"bilinear"`` / ``"bicubic"``
        / ``"trilinear"``.  If ``True``, the corner pixels of the input
        and output are exactly aligned, and the values at those corners
        are preserved.  If ``False`` (default), pixels are treated as
        ``1×1`` squares and the sampling grid is centred — this matches
        the convention used by most modern vision frameworks.
    recompute_scale_factor : bool, optional
        If ``True``, recompute ``scale_factor`` from the resolved
        ``size`` to avoid floating-point drift when chaining resamples.

    Returns
    -------
    Tensor
        Resampled tensor of shape ``(N, C, *out_spatial)``.

    Notes
    -----
    The two ``align_corners`` conventions produce different sampling
    grids — :math:`x_{\mathrm{src}} = (x_{\mathrm{dst}} + 0.5) \cdot
    (S_{\mathrm{in}} / S_{\mathrm{out}}) - 0.5` for the default, vs
    :math:`x_{\mathrm{src}} = x_{\mathrm{dst}} \cdot
    ((S_{\mathrm{in}} - 1) / (S_{\mathrm{out}} - 1))` for
    ``align_corners=True``.  The choice matters when fine alignment with
    other ops (warping, geometric losses) is required.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import interpolate
    >>> x = lucid.randn(1, 3, 32, 32)
    >>> y = interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    >>> y.shape
    (1, 3, 64, 64)
    """
    if mode in ("nearest", "nearest-exact"):
        ndim = x.ndim - 2
        if ndim == 2:
            if size is not None:
                oh, ow = (size, size) if isinstance(size, int) else size
            else:
                assert scale_factor is not None
                sf = (
                    (scale_factor, scale_factor)
                    if isinstance(scale_factor, (int, float))
                    else scale_factor
                )
                oh = int(x.shape[2] * sf[0])
                ow = int(x.shape[3] * sf[1])
            return _wrap(_C_engine.nn.interpolate_nearest_2d(_unwrap(x), oh, ow))
        if ndim == 3:
            if size is not None:
                od, oh, ow = (size, size, size) if isinstance(size, int) else size
            else:
                assert scale_factor is not None
                sf = (
                    (scale_factor,) * 3
                    if isinstance(scale_factor, (int, float))
                    else scale_factor
                )
                od = int(x.shape[2] * sf[0])
                oh = int(x.shape[3] * sf[1])
                ow = int(x.shape[4] * sf[2])
            return _wrap(_C_engine.nn.interpolate_nearest_3d(_unwrap(x), od, oh, ow))
    if mode == "bilinear":
        if size is not None:
            oh, ow = (size, size) if isinstance(size, int) else size
        else:
            assert scale_factor is not None
            sf = (
                (scale_factor, scale_factor)
                if isinstance(scale_factor, (int, float))
                else scale_factor
            )
            oh = int(x.shape[2] * sf[0])
            ow = int(x.shape[3] * sf[1])
        ac = align_corners if align_corners is not None else False
        return _wrap(_C_engine.nn.interpolate_bilinear(_unwrap(x), oh, ow, ac))
    if mode == "trilinear":
        if size is not None:
            od, oh, ow = (size, size, size) if isinstance(size, int) else size
        else:
            assert scale_factor is not None
            sf = (
                (scale_factor,) * 3
                if isinstance(scale_factor, (int, float))
                else scale_factor
            )
            od = int(x.shape[2] * sf[0])
            oh = int(x.shape[3] * sf[1])
            ow = int(x.shape[4] * sf[2])
        ac = align_corners if align_corners is not None else False
        return _wrap(_C_engine.nn.interpolate_trilinear(_unwrap(x), od, oh, ow, ac))
    raise ValueError(f"Unsupported interpolation mode: {mode!r}")


def grid_sample(
    x: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool | None = None,
) -> Tensor:
    r"""Sample an input feature map at flow-field coordinates.

    For each output location, looks up a value in ``x`` at the
    (sub-pixel) location specified by ``grid``, using bilinear (or
    nearest) interpolation.  Forms the second half of a spatial
    transformer network (STN) when paired with :func:`affine_grid`.

    The grid uses normalised coordinates in :math:`[-1, 1]`:
    :math:`(-1, -1)` denotes the top-left corner of ``x`` and
    :math:`(+1, +1)` the bottom-right corner (with the
    ``align_corners=False`` convention placing corners just outside the
    edge pixels).

    Parameters
    ----------
    x : Tensor
        Source feature map of shape ``(N, C, H_in, W_in)`` (4-D) or
        ``(N, C, D_in, H_in, W_in)`` (5-D for volumetric sampling).
    grid : Tensor
        Sampling grid of shape ``(N, H_out, W_out, 2)`` for 2-D input
        or ``(N, D_out, H_out, W_out, 3)`` for 3-D input.  Last axis
        stores ``(x, y)`` (or ``(x, y, z)``) in :math:`[-1, 1]`.
    mode : str, optional
        Interpolation mode: ``"bilinear"`` (default) or ``"nearest"``.
    padding_mode : str, optional
        Behaviour for grid locations outside :math:`[-1, 1]`:
        ``"zeros"`` (default), ``"border"`` (clamp to edge), or
        ``"reflection"``.
    align_corners : bool, optional
        Controls whether ``±1`` denotes the centre of the corner pixels
        (``True``) or the outer edge of the image (``False``, default).

    Returns
    -------
    Tensor
        Sampled output of shape ``(N, C, H_out, W_out)`` (or 5-D
        analogue), with the same dtype as ``x``.

    Notes
    -----
    ``grid_sample`` is differentiable w.r.t. both ``x`` and ``grid``.
    The gradient w.r.t. ``grid`` is what makes STN-style learned warps
    trainable end-to-end (Jaderberg et al., 2015).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import affine_grid, grid_sample
    >>> x = lucid.randn(1, 3, 32, 32)
    >>> theta = lucid.eye(2, 3).unsqueeze(0)        # identity
    >>> g = affine_grid(theta, (1, 3, 32, 32), align_corners=False)
    >>> y = grid_sample(x, g, align_corners=False)
    >>> y.shape
    (1, 3, 32, 32)
    """
    ac = align_corners if align_corners is not None else False
    return _wrap(_C_engine.nn.grid_sample(_unwrap(x), _unwrap(grid), ac))


def affine_grid(
    theta: Tensor,
    size: list[int] | tuple[int, ...],
    align_corners: bool | None = None,
) -> Tensor:
    r"""Generate a sampling grid from a batch of affine transform matrices.

    Builds the flow field needed to apply an affine transform via
    :func:`grid_sample`.  Together they form a Spatial Transformer
    Network (STN, Jaderberg et al. 2015), where ``theta`` is typically
    the output of a localisation network.

    For each output pixel :math:`(x_{\mathrm{out}}, y_{\mathrm{out}})`
    the corresponding source coordinate is computed as:

    .. math::

        \begin{pmatrix} x_{\mathrm{src}} \\ y_{\mathrm{src}} \end{pmatrix}
        = \theta \,
          \begin{pmatrix} x_{\mathrm{out}} \\ y_{\mathrm{out}} \\ 1 \end{pmatrix}

    Parameters
    ----------
    theta : Tensor
        Batch of affine matrices of shape ``(N, 2, 3)`` (2-D) or
        ``(N, 3, 4)`` (3-D).  Each row encodes one element of the affine
        transform applied to homogeneous output coordinates.
    size : sequence of int
        Target output shape passed to ``grid_sample`` afterwards —
        ``(N, C, H, W)`` for 2-D, ``(N, C, D, H, W)`` for 3-D.
    align_corners : bool, optional
        Must match the ``align_corners`` argument later passed to
        :func:`grid_sample`.  Controls whether ``-1``/``+1`` refer to
        the centre of the corner pixels (``True``) or to the outer edge
        (``False``, default).

    Returns
    -------
    Tensor
        Sampling grid of shape ``(N, H, W, 2)`` (or
        ``(N, D, H, W, 3)`` for 3-D) in normalised :math:`[-1, 1]`
        coordinates, ready to be fed to :func:`grid_sample`.

    Notes
    -----
    ``affine_grid`` does not look at the source image — only at the
    requested output size — so the resulting grid is reusable across any
    input of matching spatial dims and any channel count.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import affine_grid
    >>> theta = lucid.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # identity
    >>> g = affine_grid(theta, (1, 3, 4, 4), align_corners=False)
    >>> g.shape
    (1, 4, 4, 2)
    """
    ac = align_corners if align_corners is not None else False
    # Engine signature: (theta, N, H, W, align_corners) — no C dimension.
    sz = list(size)
    N, H, W = sz[0], sz[2], sz[3]
    return _wrap(_C_engine.nn.affine_grid(_unwrap(theta), N, H, W, ac))


def unfold(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
) -> Tensor:
    r"""Extract sliding local blocks (im2col) from a batched 4-D tensor.

    Slides a ``(kH, kW)`` window across the spatial extent of ``x`` (with
    given stride, padding, and dilation) and flattens each window into a
    column of the output.  This is the classical *im2col* operation —
    convolution can be written as ``unfold`` + matmul + ``fold``.

    The number of output locations is

    .. math::

        L = \prod_d
            \left\lfloor
                \frac{S_{\mathrm{in},d} + 2 p_d - d_d (k_d - 1) - 1}{s_d}
            \right\rfloor + 1

    where :math:`d` indexes spatial dims and :math:`s, p, k, d` are
    stride / padding / kernel / dilation respectively.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, H, W)``.
    kernel_size : int or (int, int)
        Spatial size of the sliding window.
    dilation : int or (int, int), optional
        Spacing between elements within a window.  Default ``1``.
    padding : int or (int, int), optional
        Implicit zero padding applied symmetrically on both sides.
    stride : int or (int, int), optional
        Step between successive windows.  Default ``1``.

    Returns
    -------
    Tensor
        Tensor of shape ``(N, C·kH·kW, L)``.

    Notes
    -----
    The inverse, :func:`fold`, sums overlapping blocks back into an
    image.  ``fold(unfold(x)) == x`` only when stride equals kernel
    size and padding is zero — otherwise overlapping windows produce
    higher counts at interior pixels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import unfold
    >>> x = lucid.randn(1, 3, 8, 8)
    >>> u = unfold(x, kernel_size=3, stride=1, padding=1)
    >>> u.shape                        # (N, C·kH·kW, L) = (1, 27, 64)
    (1, 27, 64)
    """

    def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
        return (v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]

    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)
    return _wrap(
        _C_engine.nn.unfold(_unwrap(x), [kh, kw], [sh, sw], [ph, pw], [dh, dw])
    )


def fold(
    x: Tensor,
    output_size: tuple[int, int],
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
) -> Tensor:
    r"""Combine an array of sliding local blocks back into an image (col2im).

    Inverse of :func:`unfold`.  Given an ``(N, C·kH·kW, L)`` tensor of
    column-vectorised blocks, scatter-adds each block into its place in
    a fresh ``(N, C, outH, outW)`` canvas.  Overlapping positions are
    summed — which is precisely what is needed for the gradient of
    convolution and for transposed-convolution-style upsampling.

    Parameters
    ----------
    x : Tensor
        Block tensor of shape ``(N, C·kH·kW, L)``.
    output_size : (int, int)
        Spatial size ``(outH, outW)`` of the destination canvas.
    kernel_size : int or (int, int)
        Spatial size of each block.  Must match the ``kernel_size`` used
        to produce ``x``.
    dilation : int or (int, int), optional
        Spacing between elements within a block.
    padding : int or (int, int), optional
        Implicit zero padding to subtract from the destination canvas
        (mirrors the ``padding`` argument of :func:`unfold`).
    stride : int or (int, int), optional
        Step between block positions on the destination canvas.

    Returns
    -------
    Tensor
        Reconstructed tensor of shape ``(N, C, outH, outW)``.

    Notes
    -----
    CPU path uses a scatter-add loop; the GPU path emits a single
    ``scatter_add_axis`` over precomputed flat destination indices, with
    no host round-trip.  In conjunction with :func:`unfold`,
    ``fold`` lets you implement arbitrary local linear operators as
    plain matrix multiplications.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import unfold, fold
    >>> x = lucid.randn(1, 3, 8, 8)
    >>> u = unfold(x, kernel_size=3, stride=3)
    >>> y = fold(u, output_size=(8, 8), kernel_size=3, stride=3)
    >>> y.shape
    (1, 3, 8, 8)
    """

    def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
        return (v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]

    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)
    out_h, out_w = output_size

    impl = _C_engine.nn.fold(
        _unwrap(x),
        [out_h, out_w],
        [kh, kw],
        [sh, sw],
        [ph, pw],
        [dh, dw],
    )
    from lucid._dispatch import _wrap as _w

    return _w(impl)


def embedding_bag(
    x: Tensor,
    weight: Tensor,
    offsets: Tensor | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Tensor | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> Tensor:
    r"""Aggregate embeddings into per-bag pooled vectors.

    Conceptually equivalent to looking up each index with
    :func:`embedding` and then reducing across the bag axis, but fused
    into a single op that avoids materialising the per-token embedding
    matrix — essential for very large vocabularies in recommendation
    and NLP models.

    Given a flat index sequence ``x`` partitioned into bags by
    ``offsets``, the ``i``-th output row is

    .. math::

        \mathrm{out}[i] = \mathrm{reduce}_{j \in \mathrm{bag}_i} W[\, x[j]\,]

    where ``reduce`` is one of ``sum``, ``mean``, or ``max``.

    Parameters
    ----------
    x : Tensor
        Either a 1-D index tensor (use with ``offsets``) or a 2-D
        ``(num_bags, seq_len)`` tensor of indices where each row is a
        bag of equal length.
    weight : Tensor
        Embedding table of shape ``(num_embeddings, embedding_dim)``.
    offsets : Tensor, optional
        Required when ``x`` is 1-D.  Integer tensor whose ``i``-th
        element is the starting index of bag ``i`` within ``x``.
    max_norm : float, optional
        Renormalise embedding rows with :math:`L_p` norm exceeding
        ``max_norm`` before lookup.
    norm_type : float, optional
        :math:`p` exponent for ``max_norm``.  Default ``2.0``.
    scale_grad_by_freq : bool, optional
        Scale gradients of each embedding row by inverse mini-batch
        frequency.
    mode : str, optional
        Bag reduction: ``"sum"``, ``"mean"`` (default), or ``"max"``.
    sparse : bool, optional
        Request a sparse gradient (accepted for compatibility).
    per_sample_weights : Tensor, optional
        Optional per-element weights applied before reduction.  Same
        shape as ``x`` (only valid for ``mode="sum"`` in most
        reference implementations).
    include_last_offset : bool, optional
        If ``True``, ``offsets`` has length ``num_bags + 1`` and its
        last entry is the total number of indices in ``x``.
    padding_idx : int, optional
        Embedding row to mask out (its lookup result contributes zero).

    Returns
    -------
    Tensor
        Pooled output of shape ``(num_bags, embedding_dim)``.

    Notes
    -----
    Compared with ``embedding`` + manual reduction, ``embedding_bag``
    saves a full materialisation of the per-token table and fuses the
    reduction into a single scatter-add (or scatter-max) pass.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import embedding_bag
    >>> w = lucid.randn(10, 4)
    >>> ids = lucid.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=lucid.int64)
    >>> off = lucid.tensor([0, 4], dtype=lucid.int64)
    >>> out = embedding_bag(ids, w, offsets=off, mode="mean")
    >>> out.shape
    (2, 4)
    """
    _mode_map = {"sum": 0, "mean": 1, "max": 2}
    mode_int = _mode_map.get(mode, 1)
    pad_idx = int(padding_idx) if padding_idx is not None else -1

    x_impl = _unwrap(x)
    w_impl = _unwrap(weight)
    x_ndim = len(x_impl.shape)

    if x_ndim == 2:
        # 2-D: each row is a bag — build synthetic 1-D x and offsets
        n_rows, seq_len = int(x_impl.shape[0]), int(x_impl.shape[1])
        # Flatten indices to 1-D
        flat_x = _C_engine.reshape(x_impl, [n_rows * seq_len])
        # Offsets: [0, seq_len, 2*seq_len, ...]
        off_vals = list(range(0, n_rows * seq_len, seq_len))
        off_impl = _C_engine.full([n_rows], 0, _C_engine.I32, x_impl.device)
        # Build offsets as arange * seq_len via engine
        arange_impl = _C_engine.arange(
            0, n_rows * seq_len, seq_len, _C_engine.I32, x_impl.device
        )
        off_impl = arange_impl
        impl = _C_engine.nn.embedding_bag(
            w_impl, flat_x, off_impl, mode_int, pad_idx, False
        )
    else:
        # 1-D x with explicit offsets
        if offsets is None:
            raise ValueError("embedding_bag: offsets required for 1-D input")
        # Cast offsets to I32 if needed
        off_impl = _unwrap(offsets)
        if off_impl.dtype != _C_engine.I32:
            off_impl = _C_engine.cast(off_impl, _C_engine.I32)  # type: ignore[attr-defined]
        impl = _C_engine.nn.embedding_bag(
            w_impl, x_impl, off_impl, mode_int, pad_idx, include_last_offset
        )

    return _wrap(impl)


_PAD_MODES: frozenset[str] = frozenset({"constant", "reflect", "replicate", "circular"})


def _flat_to_per_dim_pairs(
    padding: tuple[int, ...], ndim: int
) -> list[tuple[int, int]]:
    """Convert flat (last→first) padding tuple to per-dim (first→last) pairs."""
    n_pad_dims: int = len(padding) // 2
    pad_pairs: list[tuple[int, int]] = [(0, 0)] * ndim
    for i in range(n_pad_dims):
        dim_idx: int = ndim - 1 - i
        pad_pairs[dim_idx] = (padding[2 * i], padding[2 * i + 1])
    return pad_pairs


def _gather_along(x: Tensor, dim: int, indices_1d: list[int]) -> Tensor:
    """Gather x along `dim` with a 1-D index list, broadcasting to x.shape.

    ``indices_1d`` is a Python list of ints of length ``k``.  Returns a
    tensor of the same shape as ``x`` except ``dim`` has size ``k``.  Uses
    ``lucid.gather`` so backward accumulates gradients into the correct
    source positions — this is what makes reflect/circular padding
    correctly differentiable.
    """
    k: int = len(indices_1d)
    target_shape: list[int] = [1] * x.ndim
    target_shape[dim] = k
    bcast_shape: list[int] = [k if i == dim else int(x.shape[i]) for i in range(x.ndim)]
    idx_flat: Tensor = _lucid.tensor(
        indices_1d, dtype=_lucid.int32, device=x.device
    ).reshape(target_shape)
    idx: Tensor = idx_flat.broadcast_to(bcast_shape).contiguous()
    return _lucid.gather(x, idx, dim)


def _pad_one_dim(x: Tensor, dim: int, lo: int, hi: int, mode: str) -> Tensor:
    """Apply non-constant padding along a single dimension.

    `mode` must be one of {"reflect", "replicate", "circular"}.  Constant
    padding is handled separately by the engine pad op.

    Implementation: build the lo-side and hi-side slabs via `gather` with
    explicit index lists, then `cat` everything along the target dim.
    Gather is differentiable via scatter-add, which gives the correct
    gradient accumulation for reflect (where the same source element
    contributes to both the centre and the reflected copy).
    """
    if lo == 0 and hi == 0:
        return x
    size: int = x.shape[dim]
    parts: list[Tensor] = []
    idx_list: list[int]
    if lo > 0:
        if mode == "replicate":
            idx_list = [0] * lo
        elif mode == "reflect":
            if lo > size - 1:
                raise ValueError(
                    f"reflect padding {lo} exceeds input size-1 ({size - 1}) "
                    f"on dim {dim}"
                )
            # Reflect mode: indices [lo, lo-1, ..., 1] (boundary excluded).
            idx_list = list(range(lo, 0, -1))
        elif mode == "circular":
            if lo > size:
                raise ValueError(
                    f"circular padding {lo} exceeds input size ({size}) on dim {dim}"
                )
            idx_list = list(range(size - lo, size))
        else:
            raise ValueError(f"unsupported pad mode: {mode!r}")
        parts.append(_gather_along(x, dim, idx_list))
    parts.append(x)
    if hi > 0:
        if mode == "replicate":
            idx_list = [size - 1] * hi
        elif mode == "reflect":
            if hi > size - 1:
                raise ValueError(
                    f"reflect padding {hi} exceeds input size-1 ({size - 1}) "
                    f"on dim {dim}"
                )
            # Reflect mode: indices [size-2, size-3, ..., size-1-hi].
            idx_list = list(range(size - 2, size - 2 - hi, -1))
        elif mode == "circular":
            if hi > size:
                raise ValueError(
                    f"circular padding {hi} exceeds input size ({size}) on dim {dim}"
                )
            idx_list = list(range(0, hi))
        parts.append(_gather_along(x, dim, idx_list))
    if len(parts) == 1:
        return parts[0]
    return _lucid.cat(parts, dim)


def pad(
    x: Tensor,
    padding: tuple[int, ...],
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    r"""Pad an N-D tensor along an arbitrary set of trailing dimensions.

    Generalises ``np.pad`` with several boundary modes useful in deep
    learning: zero/constant fill, mirror reflection, edge replication,
    and circular wrap-around.

    Parameters
    ----------
    x : Tensor
        Input tensor of any rank ``N``.
    padding : tuple of int
        Flat padding spec **starting from the last dimension**, in
        ``(left, right)`` pairs.  For example:

        - ``(l, r)`` pads only the last dim,
        - ``(l, r, t, b)`` pads the last two dims (last with ``(l, r)``,
          second-to-last with ``(t, b)``),
        - ``(l, r, t, b, f, k)`` extends the pattern to three dims.

        Length must be even and at most ``2·N``.
    mode : str, optional
        One of:

        - ``"constant"`` (default): fill with ``value``.
        - ``"reflect"``: mirror around the boundary, excluding the
          boundary itself (so size-1 along a padded dim is illegal).
        - ``"replicate"``: repeat the edge element.
        - ``"circular"``: wrap around (toroidal boundary conditions).
    value : float, optional
        Fill value used only when ``mode == "constant"``.

    Returns
    -------
    Tensor
        Padded tensor with each padded dim ``d`` enlarged by
        ``padding[2·k] + padding[2·k+1]`` where ``k`` is its offset from
        the last dim.

    Notes
    -----
    Non-constant modes are decomposed into gather + concat in Python,
    relying on existing autograd machinery (gather's scatter-add
    backward correctly accumulates gradients for reflected /
    replicated entries).  ``"reflect"`` requires
    ``pad_amount <= size - 1`` on each side; ``"circular"`` requires
    ``pad_amount <= size``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pad
    >>> x = lucid.arange(6).reshape(1, 1, 2, 3).astype(lucid.float32)
    >>> y = pad(x, (1, 1, 0, 0), mode="reflect")
    >>> y.shape                    # last dim grows by 2
    (1, 1, 2, 5)
    """
    if mode not in _PAD_MODES:
        raise ValueError(
            f"unknown pad mode {mode!r}; expected one of {sorted(_PAD_MODES)}"
        )
    impl: object = _unwrap(x)
    ndim: int = len(impl.shape)  # type: ignore[attr-defined]
    pad_pairs: list[tuple[int, int]] = _flat_to_per_dim_pairs(padding, ndim)
    if mode == "constant":
        return _wrap(_C_engine.pad(impl, pad_pairs, value))  # type: ignore[arg-type]
    # Non-constant: pad one dim at a time using existing ops.
    result: Tensor = x
    for d, (lo, hi) in enumerate(pad_pairs):
        if lo == 0 and hi == 0:
            continue
        result = _pad_one_dim(result, d, lo, hi, mode)
    return result


def pixel_shuffle(x: Tensor, upscale_factor: int) -> Tensor:
    r"""Sub-pixel upsampling: rearrange channels into spatial resolution.

    Reshapes a low-resolution multi-channel feature map into a
    higher-resolution map with fewer channels, *without* introducing any
    interpolation.  Originally proposed by Shi et al., 2016 ("Real-Time
    Single Image and Video Super-Resolution"), this is the canonical
    decoder layer in single-image super-resolution and many GAN /
    diffusion upsamplers.

    Concretely, with upscale factor :math:`r`:

    .. math::

        (N, C \cdot r^2, H, W) \;\longrightarrow\; (N, C, H \cdot r, W \cdot r)

    Each :math:`r \times r` patch in the output is taken from a
    block of :math:`r^2` channels in the input.

    Parameters
    ----------
    x : Tensor
        4-D input of shape ``(N, C·r², H, W)``.  The channel dimension
        must be divisible by ``upscale_factor²``.
    upscale_factor : int
        Spatial upscaling factor :math:`r`.

    Returns
    -------
    Tensor
        Upsampled tensor of shape ``(N, C, H·r, W·r)``.

    Notes
    -----
    The transformation is a pure reshape + permute + reshape — autograd
    flows through ``ReshapeBackward`` and ``PermuteBackward`` with no
    custom gradient.  Because no convolution is involved, the operator
    is essentially free; it is most often preceded by a convolution
    that produces the required ``C·r²`` channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pixel_shuffle
    >>> x = lucid.randn(1, 12, 8, 8)      # C=3, r=2 → C·r²=12
    >>> y = pixel_shuffle(x, upscale_factor=2)
    >>> y.shape
    (1, 3, 16, 16)
    """
    r: int = int(upscale_factor)
    if len(x.shape) != 4:
        raise ValueError(
            f"pixel_shuffle: expected 4-D input, got shape {tuple(x.shape)}"
        )
    n, c_r2, h, w = x.shape
    if c_r2 % (r * r) != 0:
        raise ValueError(
            f"pixel_shuffle: channels {c_r2} not divisible by upscale_factor² ({r * r})"
        )
    c: int = c_r2 // (r * r)
    impl = _unwrap(x)
    t = _C_engine.reshape(impl, [n, c, r, r, h, w])
    t = _C_engine.permute(t, [0, 1, 4, 2, 5, 3])
    return _wrap(_C_engine.reshape(t, [n, c, h * r, w * r]))


def pixel_unshuffle(x: Tensor, downscale_factor: int) -> Tensor:
    r"""Inverse of :func:`pixel_shuffle`: pack spatial blocks into channels.

    Folds each :math:`r \times r` spatial block into ``r²`` extra
    channels, shrinking the spatial extent by :math:`r` along each axis:

    .. math::

        (N, C, H \cdot r, W \cdot r) \;\longrightarrow\; (N, C \cdot r^2, H, W)

    Useful as a strided "downsampling without information loss" stage
    in invertible architectures (e.g. RealNVP, normalising flows) and
    as the analyzer step in sub-pixel encoders.

    Parameters
    ----------
    x : Tensor
        4-D input of shape ``(N, C, H·r, W·r)``.  Both spatial dims
        must be divisible by ``downscale_factor``.
    downscale_factor : int
        Spatial downscaling factor :math:`r`.

    Returns
    -------
    Tensor
        Downsampled tensor of shape ``(N, C·r², H, W)``.

    Notes
    -----
    Like :func:`pixel_shuffle`, this is a reshape + permute + reshape
    chain — completely lossless and free of multiply-add cost.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pixel_unshuffle
    >>> x = lucid.randn(1, 3, 16, 16)
    >>> y = pixel_unshuffle(x, downscale_factor=2)
    >>> y.shape
    (1, 12, 8, 8)
    """
    r: int = int(downscale_factor)
    if len(x.shape) != 4:
        raise ValueError(
            f"pixel_unshuffle: expected 4-D input, got shape {tuple(x.shape)}"
        )
    n, c, h_r, w_r = x.shape
    if h_r % r != 0 or w_r % r != 0:
        raise ValueError(
            f"pixel_unshuffle: spatial dims ({h_r}, {w_r}) not divisible by {r}"
        )
    h, w = h_r // r, w_r // r
    impl = _unwrap(x)
    t = _C_engine.reshape(impl, [n, c, h, r, w, r])
    t = _C_engine.permute(t, [0, 1, 3, 5, 2, 4])
    return _wrap(_C_engine.reshape(t, [n, c * r * r, h, w]))


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor | None = None,
    in_proj_bias: Tensor | None = None,
    bias_k: Tensor | None = None,
    bias_v: Tensor | None = None,
    add_zero_attn: bool = False,
    dropout_p: float = 0.0,
    out_proj_weight: Tensor | None = None,
    out_proj_bias: Tensor | None = None,
    training: bool = True,
    key_padding_mask: Tensor | None = None,
    need_weights: bool = True,
    attn_mask: Tensor | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Tensor | None = None,
    k_proj_weight: Tensor | None = None,
    v_proj_weight: Tensor | None = None,
    static_k: Tensor | None = None,
    static_v: Tensor | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor | None]:
    r"""Stateless functional multi-head attention forward pass.

    Performs the full Vaswani et al. (2017) multi-head attention
    computation as a single pure function — useful for porting code
    that builds attention layers without instantiating a stateful
    module.

    The computation has four stages:

    1. **Input projection.**  Linear projections produce
       :math:`Q = X_q W_Q`, :math:`K = X_k W_K`, :math:`V = X_v W_V`
       (when ``use_separate_proj_weight=False``, these come from a
       single fused ``in_proj_weight``).
    2. **Head split.**  Reshape each of ``Q``, ``K``, ``V`` from
       ``(L, B, d_model)`` to ``(L, B·H, d_model/H)`` so each of the
       :math:`H` heads attends independently.
    3. **Scaled dot-product attention.**

       .. math::

           \mathrm{Attn} = \mathrm{softmax}\!\left(
               \frac{Q K^\top}{\sqrt{d_k}} + M\right) V

       where ``M`` is the union of the optional ``attn_mask``,
       ``key_padding_mask``, and the implicit causal mask when
       ``is_causal=True``.

    4. **Output projection.**  Concatenate heads and apply
       :math:`\mathrm{out} = \mathrm{Attn}_\text{concat} W_O`.

    Parameters
    ----------
    query, key, value : Tensor
        Inputs of shape ``(L, N, E)`` (target / source / source length
        × batch × ``embed_dim``).  Self-attention uses the same tensor
        for all three; cross-attention takes a separate ``query``.
    embed_dim_to_check : int
        Must equal ``query.shape[-1]``; sanity check.
    num_heads : int
        Number of attention heads :math:`H`.  Must divide ``embed_dim``.
    in_proj_weight, in_proj_bias : Tensor, optional
        Fused QKV projection.  ``in_proj_weight`` has shape
        ``(3·embed_dim, embed_dim)``.
    bias_k, bias_v : Tensor, optional
        Optional learned bias vectors appended to ``K`` / ``V`` along
        the sequence axis.
    add_zero_attn : bool, optional
        If ``True``, append a row of zeros to ``K`` / ``V`` — not yet
        supported in Lucid (will raise ``NotImplementedError``).
    dropout_p : float, optional
        Dropout probability applied to attention weights during
        training.
    out_proj_weight, out_proj_bias : Tensor, optional
        Output linear projection :math:`W_O`.
    training : bool, optional
        If ``True``, dropout is active.
    key_padding_mask : Tensor, optional
        Boolean mask of shape ``(N, S)`` with ``True`` at padded
        positions to be ignored.
    need_weights : bool, optional
        If ``True``, also return the attention weight matrix.
    attn_mask : Tensor, optional
        Additional additive mask, e.g. a causal triangular mask.
    use_separate_proj_weight : bool, optional
        Use ``q_proj_weight`` / ``k_proj_weight`` / ``v_proj_weight``
        instead of a fused ``in_proj_weight``.  Not yet supported.
    q_proj_weight, k_proj_weight, v_proj_weight : Tensor, optional
        Separate Q / K / V projection weights (unsupported).
    static_k, static_v : Tensor, optional
        Precomputed K / V to attend over (unsupported).
    average_attn_weights : bool, optional
        If ``True``, the returned weight tensor is averaged across
        heads; otherwise per-head weights are returned.
    is_causal : bool, optional
        Apply an upper-triangular causal mask.  Mutually exclusive
        with a user-supplied ``attn_mask``.

    Returns
    -------
    (Tensor, Tensor or None)
        - The attention output of shape ``(L, N, E)``.
        - The attention weights of shape ``(N, L, S)`` (or
          ``(N, H, L, S)`` if ``average_attn_weights=False``),
          or ``None`` if ``need_weights=False``.

    Notes
    -----
    This functional is implemented by binding the supplied weight
    tensors onto a transient ``MultiheadAttention`` module so the
    forward behaviour is bit-identical to the module path.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multi_head_attention_forward
    >>> L, N, E, H = 16, 2, 64, 8
    >>> q = k = v = lucid.randn(L, N, E)
    >>> Wi = lucid.randn(3 * E, E)
    >>> Wo = lucid.randn(E, E)
    >>> out, attn = multi_head_attention_forward(
    ...     q, k, v, embed_dim_to_check=E, num_heads=H,
    ...     in_proj_weight=Wi, out_proj_weight=Wo,
    ...     need_weights=False,
    ... )
    >>> out.shape
    (16, 2, 64)
    """
    from lucid.nn.modules.attention import MultiheadAttention

    # The unused-but-validated arguments below are kept on the signature for
    # ``F.multi_head_attention_forward`` source-level compatibility.  Static
    # K/V and zero-attn slots are advanced features the module path doesn't
    # cover yet — they raise rather than silently misbehaving.
    if static_k is not None or static_v is not None:
        raise NotImplementedError(
            "multi_head_attention_forward: static_k/static_v unsupported"
        )
    if add_zero_attn:
        raise NotImplementedError(
            "multi_head_attention_forward: add_zero_attn unsupported"
        )
    if use_separate_proj_weight:
        raise NotImplementedError(
            "multi_head_attention_forward: use_separate_proj_weight unsupported"
        )

    # Build a temporary module; bind external weights so the call mirrors the
    # functional contract exactly.
    mha: MultiheadAttention = MultiheadAttention(
        embed_dim=int(embed_dim_to_check),
        num_heads=int(num_heads),
        dropout=float(dropout_p),
        bias=in_proj_bias is not None or out_proj_bias is not None,
        add_bias_kv=bias_k is not None,
        batch_first=False,
    )
    if in_proj_weight is not None and mha.in_proj_weight is not None:
        mha.in_proj_weight._impl = _unwrap(in_proj_weight)
    if in_proj_bias is not None and mha.in_proj_bias is not None:
        mha.in_proj_bias._impl = _unwrap(in_proj_bias)
    if out_proj_weight is not None:
        mha.out_proj_weight._impl = _unwrap(out_proj_weight)
    if out_proj_bias is not None and mha.out_proj_bias is not None:
        mha.out_proj_bias._impl = _unwrap(out_proj_bias)
    if bias_k is not None and mha.bias_k is not None:
        mha.bias_k._impl = _unwrap(bias_k)
    if bias_v is not None and mha.bias_v is not None:
        mha.bias_v._impl = _unwrap(bias_v)

    if training:
        mha.train()
    else:
        mha.eval()
    return mha(  # type: ignore[return-value]
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )


# ── P3 fills: channel_shuffle / pdist ──────────────────────────────────────


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    r"""Group-then-transpose channel rearrangement (ShuffleNet).

    Splits the channel axis into ``groups`` equal-sized groups,
    transposes the group and intra-group axes, and flattens back.
    This re-orders channels so that, after a subsequent grouped
    convolution, information from each input group reaches every
    output group — enabling cheap cross-group information flow that
    grouped convolutions alone cannot provide.

    Conceptually:

    .. math::

        (N, g \cdot c, H, W)
            \xrightarrow{\mathrm{reshape}} (N, g, c, H, W)
            \xrightarrow{\mathrm{transpose}} (N, c, g, H, W)
            \xrightarrow{\mathrm{reshape}} (N, g \cdot c, H, W)

    where ``g = groups`` and ``c = C / groups``.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(N, C, *spatial)`` (at least 2-D).
        ``C`` must be divisible by ``groups``.
    groups : int
        Number of channel groups :math:`g`.

    Returns
    -------
    Tensor
        Channel-shuffled tensor of the same shape as ``x``.

    Notes
    -----
    Introduced in Zhang et al., 2018 — "ShuffleNet: An Extremely
    Efficient Convolutional Neural Network for Mobile Devices".  The
    operation is a pure reshape + permute, so it adds no FLOPs and
    very little memory traffic.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import channel_shuffle
    >>> x = lucid.arange(24).reshape(1, 6, 2, 2).astype(lucid.float32)
    >>> y = channel_shuffle(x, groups=3)
    >>> y.shape
    (1, 6, 2, 2)
    """
    if x.ndim < 2:
        raise ValueError(
            f"channel_shuffle: expected at least 2-D input (N, C, *), got "
            f"shape {tuple(x.shape)}"
        )
    n = int(x.shape[0])
    c = int(x.shape[1])
    if c % int(groups) != 0:
        raise ValueError(
            f"channel_shuffle: channel count {c} not divisible by groups " f"{groups}"
        )
    ch_per_group = c // int(groups)
    spatial = list(x.shape[2:])

    # (N, groups, C/groups, *spatial) → (N, C/groups, groups, *spatial) →
    # flatten the two leading channel-related axes back into a single ``C``.
    reshaped = x.reshape(n, int(groups), ch_per_group, *spatial)
    perm = [0, 2, 1] + list(range(3, 3 + len(spatial)))
    transposed = _lucid.permute(reshaped, *perm)  # type: ignore[arg-type]
    return transposed.reshape(n, c, *spatial)


def pdist(x: Tensor, p: float = 2.0) -> Tensor:
    r"""Pairwise :math:`L_p` distances between rows of a 2-D tensor.

    For an input of shape ``(N, M)``, returns the
    :math:`N \cdot (N - 1) / 2` strictly-upper-triangular pairwise
    distances in row-major order:

    .. math::

        \mathrm{out}[k(i, j)] =
            \left( \sum_{m=1}^{M} | x_{i, m} - x_{j, m} |^p \right)^{1/p}
        \quad \text{for } 0 \le i < j < N

    where ``k(i, j)`` enumerates pairs in row-major order.

    Parameters
    ----------
    x : Tensor
        2-D input of shape ``(N, M)``.
    p : float, optional
        Exponent of the :math:`L_p` norm.  Common values: ``2`` for
        Euclidean (default), ``1`` for Manhattan, ``float("inf")`` for
        Chebyshev.

    Returns
    -------
    Tensor
        1-D tensor of length :math:`N (N - 1) / 2` (empty when
        :math:`N < 2`).

    Notes
    -----
    Internally computed by extracting the strict upper triangle of the
    full :func:`lucid.cdist` matrix, which keeps the implementation
    simple and fully differentiable but is :math:`O(N^2)` in both
    compute and intermediate memory — for very large ``N``, prefer a
    blocked / streaming pairwise routine.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pdist
    >>> x = lucid.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> pdist(x, p=2)
    Tensor([1.0000, 1.0000, 1.4142])
    """
    if x.ndim != 2:
        raise ValueError(f"pdist: expected a 2-D input, got shape {tuple(x.shape)}")
    n = int(x.shape[0])
    if n < 2:
        return _lucid.zeros(0, dtype=x.dtype, device=x.device)
    full = _lucid.cdist(x, x, p=p)  # type: ignore[arg-type]  # (N, N)
    # Pull out the strict upper triangle (i < j) in row-major order via the
    # index pair generator from ``triu_indices``.
    pairs = _lucid.triu_indices(n, n, offset=1)  # type: ignore[arg-type]  # shape (2, N·(N-1)/2)
    flat = full.reshape(n * n)
    flat_idx = pairs[0] * n + pairs[1]
    return _lucid.gather(flat, flat_idx, dim=0)
