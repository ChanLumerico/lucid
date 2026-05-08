"""
nn.functional sampling / interpolation / padding operations.
"""

from typing import TYPE_CHECKING
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
    """Interpolate a tensor to a given size or scale factor."""
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
    """Sample x using grid coordinates."""
    ac = align_corners if align_corners is not None else False
    return _wrap(_C_engine.nn.grid_sample(_unwrap(x), _unwrap(grid), ac))


def affine_grid(
    theta: Tensor,
    size: list[int] | tuple[int, ...],
    align_corners: bool | None = None,
) -> Tensor:
    """Generate a sampling grid for affine_grid / grid_sample."""
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
    """Extract sliding local blocks from a batched 4-D input tensor.

    Args:
        x:           Input of shape (N, C, H, W).
        kernel_size: Size of the sliding blocks.
        dilation:    Stride between elements within a sliding block.
        padding:     Implicit zero padding on both sides.
        stride:      Stride of the sliding blocks.

    Returns:
        Tensor of shape (N, C*kH*kW, L) where L = output locations.
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
    """Combine an array of sliding local blocks into a large containing tensor.

    This is the inverse operation of ``unfold``.  Given an input of shape
    ``(N, C*kH*kW, L)``, returns a tensor of shape ``(N, C, outH, outW)``.
    Overlapping blocks are summed.
    CPU: scatter-add loop.  GPU: CPU fallback.
    Parameters match ``the reference fold API``.
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
    """Computes sums, means, or maxima of embeddings in bags.

    When ``x`` is 2-D the bags are the rows; when 1-D and ``offsets`` is given,
    ``offsets`` marks the start of each bag (reference convention).
    CPU: gather + reduce loop.  GPU: MLX gather + scatter_add/scatter_max.
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
            off_impl = _C_engine.cast(off_impl, _C_engine.I32)
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
    import lucid as _lucid

    k: int = len(indices_1d)
    target_shape: list[int] = [1] * x.ndim
    target_shape[dim] = k
    bcast_shape: list[int] = [
        k if i == dim else int(x.shape[i]) for i in range(x.ndim)
    ]
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
    import lucid as _lucid

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
    """Pad a tensor.

    padding follows reference convention: flat tuple starting from the LAST dimension.
    For example, (l, r) pads the last dim; (l, r, t, b) pads last two dims.

    `mode` ∈ {"constant", "reflect", "replicate", "circular"}.
    Non-constant modes are implemented in Python via slice + flip + cat;
    autograd flows through the existing op backwards.
    """
    if mode not in _PAD_MODES:
        raise ValueError(
            f"unknown pad mode {mode!r}; expected one of {sorted(_PAD_MODES)}"
        )
    impl: object = _unwrap(x)
    ndim: int = len(impl.shape)
    pad_pairs: list[tuple[int, int]] = _flat_to_per_dim_pairs(padding, ndim)
    if mode == "constant":
        return _wrap(_C_engine.pad(impl, pad_pairs, value))
    # Non-constant: pad one dim at a time using existing ops.
    result: Tensor = x
    for d, (lo, hi) in enumerate(pad_pairs):
        if lo == 0 and hi == 0:
            continue
        result = _pad_one_dim(result, d, lo, hi, mode)
    return result


def pixel_shuffle(x: Tensor, upscale_factor: int) -> Tensor:
    """Rearrange ``(N, C·r², H, W)`` → ``(N, C, H·r, W·r)`` for a 4-D input.

    The transformation is a reshape + permute + reshape — autograd flows
    through ``ReshapeBackward`` and ``PermuteBackward`` automatically.
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
    """Inverse of ``pixel_shuffle``: ``(N, C, H·r, W·r)`` → ``(N, C·r², H, W)``."""
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
    """Stateless functional form of ``MultiheadAttention.forward``.

    Mirrors the reference framework's ``F.multi_head_attention_forward`` —
    the same signature ``MultiheadAttention`` calls internally.  Most users
    should reach for the ``MultiheadAttention`` module instead; this lives
    here for code that builds attention layers without instantiating a module
    (e.g. when porting reference code that calls the functional directly).

    The implementation delegates the heavy lifting to a temporary
    ``MultiheadAttention`` whose parameters are bound to the supplied
    weight tensors — keeps the behaviour bit-identical to the module.
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
    return mha(
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
    """ShuffleNet channel-shuffle: split the channel axis into ``groups``,
    transpose, and flatten back.  Re-orders channels so that information
    from each group reaches every other group in the next layer.

    Expects an ``(N, C, *spatial)`` tensor with ``C`` divisible by
    ``groups``.
    """
    import lucid as _l

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
    transposed = _l.permute(reshaped, *perm)
    return transposed.reshape(n, c, *spatial)


def pdist(x: Tensor, p: float = 2.0) -> Tensor:
    """Pairwise ``Lp`` distances between rows of a 2-D tensor.

    ``x`` has shape ``(N, M)`` and the result is a 1-D vector of length
    ``N · (N - 1) / 2`` holding the upper-triangular (excluding the
    diagonal) distances in row-major order — same convention as the
    reference framework's ``pdist``.
    """
    import lucid as _l

    if x.ndim != 2:
        raise ValueError(f"pdist: expected a 2-D input, got shape {tuple(x.shape)}")
    n = int(x.shape[0])
    if n < 2:
        return _l.zeros(0, dtype=x.dtype, device=x.device)
    full = _l.cdist(x, x, p=p)  # (N, N)
    # Pull out the strict upper triangle (i < j) in row-major order via the
    # index pair generator from ``triu_indices``.
    pairs = _l.triu_indices(n, n, offset=1)  # shape (2, N·(N-1)/2)
    flat = full.reshape(n * n)
    flat_idx = pairs[0] * n + pairs[1]
    return _l.gather(flat, flat_idx, dim=0)
