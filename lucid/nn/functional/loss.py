"""
nn.functional loss functions.
"""

from typing import TYPE_CHECKING

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_REDUCTION_MAP: dict[str, int] = {"none": 0, "mean": 1, "sum": 2}


def _validate_reduction(reduction: str, allow_batchmean: bool = False) -> None:
    valid: tuple[str, ...] = (
        ("none", "mean", "sum", "batchmean")
        if allow_batchmean
        else ("none", "mean", "sum")
    )
    if reduction not in valid:
        raise ValueError(f"reduction must be one of {valid}, got {reduction!r}")


def mse_loss(x: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error loss."""
    _validate_reduction(reduction)
    red: int = _REDUCTION_MAP[reduction]
    return _wrap(_C_engine.nn.mse_loss(_unwrap(x), _unwrap(target), red))


def l1_loss(x: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean absolute error loss."""
    _validate_reduction(reduction)
    diff: object = _C_engine.abs(_C_engine.sub(_unwrap(x), _unwrap(target)))
    if reduction == "mean":
        return _wrap(_C_engine.mean(diff, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(diff, [], False))
    return _wrap(diff)


def smooth_l1_loss(
    x: Tensor, target: Tensor, beta: float = 1.0, reduction: str = "mean"
) -> Tensor:
    """Smooth L1 (Huber) loss."""
    return huber_loss(x, target, delta=beta, reduction=reduction)


def huber_loss(
    x: Tensor, target: Tensor, delta: float = 1.0, reduction: str = "mean"
) -> Tensor:
    """Huber loss."""
    _validate_reduction(reduction)
    red: int = _REDUCTION_MAP[reduction]
    return _wrap(_C_engine.nn.huber_loss(_unwrap(x), _unwrap(target), delta, red))


def cross_entropy(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy loss for multi-class classification.

    Implements the full contract: per-class ``weight`` broadcasting,
    ``ignore_index`` masking, and ``label_smoothing``.

    Inputs
    ------
    x : (N, C) or (N, C, ...) logits
    target : (N,) or (N, ...) integer class indices
    weight : (C,) optional per-class weight
    ignore_index : int — samples with this class are excluded
    label_smoothing : α ∈ [0, 1) — interpolates between hard targets
        and the uniform distribution
    """
    _validate_reduction(reduction)
    from lucid.nn.functional.activations import log_softmax as _log_softmax

    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing!r}")

    log_p: Tensor = _log_softmax(x, dim=1)
    # Class dim is 1 for both (N, C) and (N, C, *) inputs.
    num_classes: int = log_p.shape[1]

    # Build a per-sample NLL by gathering along the class axis.
    target_long: Tensor = target.to(dtype=_lucid.int32)
    target_unsq: Tensor = target_long.unsqueeze(1)
    gathered: Tensor = _lucid.gather(log_p, target_unsq, 1).squeeze(1)
    nll: Tensor = -gathered  # (N, ...)

    # ── weight (per-class) ──────────────────────────────────────────────
    sample_weight: Tensor | None = None
    if weight is not None:
        sample_weight = _lucid.gather(weight, target_long, 0)
        nll = nll * sample_weight

    # ── ignore_index mask ───────────────────────────────────────────────
    keep_mask_f: Tensor | None = None
    if ignore_index is not None:
        from lucid._factories.creation import full as _full

        ig_t: Tensor = _full(
            target_long.shape, int(ignore_index), dtype=_lucid.int32, device=x.device
        )
        keep_mask: Tensor = target_long != ig_t
        keep_mask_f = keep_mask.to(dtype=x.dtype)
        nll = nll * keep_mask_f

    # ── label_smoothing ─────────────────────────────────────────────────
    if label_smoothing > 0.0:
        # Smoothing term — uniform distribution NLL = -mean over classes
        # of log_softmax, which is -sum/C.  When weight is set, the uniform
        # term is weighted by (mean class weight) following the reference
        # framework's behaviour.
        smooth_per_sample: Tensor = -log_p.mean(dim=1)  # (N, ...)
        if weight is not None:
            # Weighted-uniform: −Σ_c w_c · log_softmax / C
            log_p_weighted: Tensor = log_p * weight.reshape(
                [1, num_classes] + [1] * (log_p.ndim - 2)
            )
            smooth_per_sample = -log_p_weighted.sum(dim=1) / num_classes
        if keep_mask_f is not None:
            smooth_per_sample = smooth_per_sample * keep_mask_f
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth_per_sample

    # ── reduction ───────────────────────────────────────────────────────
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    # mean — the divisor depends on weight + ignore_index.
    if weight is None and keep_mask_f is None:
        return nll.mean()
    if weight is not None and keep_mask_f is not None:
        denom: Tensor = (sample_weight * keep_mask_f).sum()
    elif weight is not None:
        denom = sample_weight.sum()
    else:
        denom = keep_mask_f.sum()
    return nll.sum() / denom


def nll_loss(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """Negative log-likelihood loss.  Input ``x`` is already log-probabilities."""
    _validate_reduction(reduction)
    target_long: Tensor = target.to(dtype=_lucid.int32)
    target_unsq: Tensor = target_long.unsqueeze(1)
    gathered: Tensor = _lucid.gather(x, target_unsq, 1).squeeze(1)
    nll: Tensor = -gathered

    sample_weight: Tensor | None = None
    if weight is not None:
        sample_weight = _lucid.gather(weight, target_long, 0)
        nll = nll * sample_weight

    keep_mask_f: Tensor | None = None
    if ignore_index is not None:
        from lucid._factories.creation import full as _full

        ig_t: Tensor = _full(
            target_long.shape, int(ignore_index), dtype=_lucid.int32, device=x.device
        )
        keep_mask: Tensor = target_long != ig_t
        keep_mask_f = keep_mask.to(dtype=x.dtype)
        nll = nll * keep_mask_f

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if weight is None and keep_mask_f is None:
        return nll.mean()
    if weight is not None and keep_mask_f is not None:
        denom: Tensor = (sample_weight * keep_mask_f).sum()
    elif weight is not None:
        denom = sample_weight.sum()
    else:
        denom = keep_mask_f.sum()
    return nll.sum() / denom


def binary_cross_entropy(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Binary cross-entropy loss.  ``weight`` is broadcast element-wise."""
    _validate_reduction(reduction)
    eps: float = 1e-12
    one: Tensor = _lucid.ones((), dtype=x.dtype, device=x.device)
    eps_t: Tensor = _lucid.tensor(eps, dtype=x.dtype, device=x.device)
    x_clamped: Tensor = x.clamp(eps, 1.0 - eps)
    bce: Tensor = -(target * x_clamped.log() + (one - target) * (one - x_clamped).log())
    if weight is not None:
        bce = bce * weight
    if reduction == "none":
        return bce
    if reduction == "sum":
        return bce.sum()
    return bce.mean()


def binary_cross_entropy_with_logits(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """BCE with logits — combines sigmoid + BCE for numerical stability.

    The numerically stable form, equivalent to BCE(sigmoid(x), y) but
    without intermediate underflow for large |x|::

        max(x, 0) − x · y + log(1 + exp(−|x|))

    With ``pos_weight`` (per-class weight on the positive term) the
    formula becomes::

        (1 + (pos_weight − 1) · y) · base + (pos_weight − 1) · y · clamp_neg(x)
    """
    _validate_reduction(reduction)
    one: Tensor = _lucid.ones((), dtype=x.dtype, device=x.device)
    abs_x: Tensor = x.abs()
    # log(1 + exp(-|x|)) — softplus(-|x|).
    log1pexp: Tensor = (one + (-abs_x).exp()).log()
    inf: float = float("inf")
    max_x: Tensor = x.clamp(0.0, inf)
    if pos_weight is None:
        loss: Tensor = max_x - x * target + log1pexp
    else:
        # Reference implementation:
        #   loss = (1 - y) * x + (1 + (pw - 1) * y) * (log(1 + exp(-|x|)) + max(-x, 0))
        max_neg: Tensor = (-x).clamp(0.0, inf)
        loss = (one - target) * x + (one + (pos_weight - one) * target) * (
            log1pexp + max_neg
        )
    if weight is not None:
        loss = loss * weight
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def kl_div(
    x: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    """Kullback-Leibler divergence.

    Reduction modes: ``none``, ``mean``, ``sum``, ``batchmean``.
    ``batchmean`` divides the summed loss by the batch (leading) dimension —
    this matches the mathematically correct KL divergence value when
    averaging over a batch.
    """
    _validate_reduction(reduction, allow_batchmean=True)
    # `x` is log_q (log of predicted probability) per the standard contract.
    # When log_target=False, target is the raw probability p; when True it
    # is log(p).  Loss elementwise = target * (log(target) - log_q).
    xi: object = _unwrap(x)
    ti: object = _unwrap(target)
    if log_target:
        # log_target=True → target itself is log(p); use exp(t) as the weight.
        diff: object = _C_engine.sub(ti, xi)
        kl: object = _C_engine.mul(_C_engine.exp(ti), diff)
    else:
        # Standard: target * (log(target) − x).
        diff = _C_engine.sub(_C_engine.log(ti), xi)
        kl = _C_engine.mul(ti, diff)
    if reduction == "mean":
        return _wrap(_C_engine.mean(kl, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(kl, [], False))
    if reduction == "batchmean":
        total: object = _C_engine.sum(kl, [], False)
        batch_size: int = int(x.shape[0])
        return _wrap(total) / batch_size
    return _wrap(kl)


def _apply_reduction(t: _C_engine.TensorImpl, reduction: str) -> Tensor:
    """Apply reduction to a batch of per-sample losses."""
    if reduction == "mean":
        return _wrap(_C_engine.mean(t, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(t, [], False))
    return _wrap(t)


def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Triplet margin loss: max(d(a,p) - d(a,n) + margin, 0)."""
    from lucid.nn.functional.activations import pairwise_distance

    d_ap = _unwrap(pairwise_distance(anchor, positive, p=p, eps=eps))
    d_an = _unwrap(pairwise_distance(anchor, negative, p=p, eps=eps))
    if swap:
        d_pn = _unwrap(pairwise_distance(positive, negative, p=p, eps=eps))
        d_an = _C_engine.minimum(d_an, d_pn)
    margin_t = _C_engine.full(d_ap.shape, margin, d_ap.dtype, d_ap.device)
    loss = _C_engine.relu(_C_engine.add(_C_engine.sub(d_ap, d_an), margin_t))
    return _apply_reduction(loss, reduction)


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    distance_function: object | None = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Triplet margin loss with a user-supplied distance callable.

    ``distance_function(x, y) -> Tensor`` defaults to L₂ pairwise
    distance (matches the reference framework).  When ``swap=True`` the
    "anchor-swap" trick from Balntas et al. 2016 is applied: replace
    ``d(a, n)`` with ``min(d(a, n), d(p, n))`` so the harder negative
    drives the gradient.

    The Lucid module wrapper :class:`lucid.nn.TripletMarginWithDistanceLoss`
    forwards into this function — kept here so callers can use the
    functional surface directly.
    """
    from lucid.nn.functional.activations import pairwise_distance

    df: object = distance_function
    if df is None:

        def df(a: Tensor, b: Tensor) -> Tensor:  # type: ignore[no-redef]
            return pairwise_distance(a, b, p=2.0)

    d_ap: Tensor = df(anchor, positive)  # type: ignore[operator]
    d_an: Tensor = df(anchor, negative)  # type: ignore[operator]
    if swap:
        d_pn: Tensor = df(positive, negative)  # type: ignore[operator]
        d_an = d_an.minimum(d_pn)

    zero: Tensor = _lucid.zeros_like(d_ap)
    loss_t: Tensor = (d_ap - d_an + margin).maximum(zero)

    _validate_reduction(reduction)
    if reduction == "mean":
        return loss_t.mean()
    if reduction == "sum":
        return loss_t.sum()
    return loss_t


def cosine_embedding_loss(
    x1: Tensor,
    x2: Tensor,
    y: Tensor,
    margin: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """Cosine embedding loss.

    y=1 → loss = 1 - cos(x1, x2)
    y=-1 → loss = max(0, cos(x1, x2) - margin)
    """
    from lucid.nn.functional.activations import cosine_similarity

    cos = _unwrap(cosine_similarity(x1, x2, dim=1))
    ones = _C_engine.full(cos.shape, 1.0, cos.dtype, cos.device)
    zeros = _C_engine.zeros(cos.shape, cos.dtype, cos.device)
    margin_t = _C_engine.full(cos.shape, margin, cos.dtype, cos.device)
    yi = _unwrap(y)
    loss_pos = _C_engine.sub(ones, cos)  # y=1
    loss_neg = _C_engine.relu(_C_engine.sub(cos, margin_t))  # y=-1
    # select by sign of y: y==1 → loss_pos, else → loss_neg
    mask = _C_engine.greater(yi, zeros)
    loss = _C_engine.where(mask, loss_pos, loss_neg)
    return _apply_reduction(loss, reduction)


def margin_ranking_loss(
    x1: Tensor,
    x2: Tensor,
    y: Tensor,
    margin: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """Margin ranking loss: max(0, -y*(x1-x2) + margin)."""
    diff = _C_engine.sub(_unwrap(x1), _unwrap(x2))
    margin_t = _C_engine.full(diff.shape, margin, diff.dtype, diff.device)
    neg_y_diff = _C_engine.mul(_C_engine.neg(_unwrap(y)), diff)
    loss = _C_engine.relu(_C_engine.add(neg_y_diff, margin_t))
    return _apply_reduction(loss, reduction)


def hinge_embedding_loss(
    x: Tensor,
    y: Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Hinge embedding loss.

    y=1 → x; y=-1 → max(0, margin - x)
    """
    xi = _unwrap(x)
    yi = _unwrap(y)
    ones = _C_engine.full(xi.shape, 1.0, xi.dtype, xi.device)
    zeros = _C_engine.zeros(xi.shape, xi.dtype, xi.device)
    margin_t = _C_engine.full(xi.shape, margin, xi.dtype, xi.device)
    loss_pos = xi  # y=1
    loss_neg = _C_engine.relu(_C_engine.sub(margin_t, xi))  # y=-1
    mask = _C_engine.greater(yi, zeros)
    loss = _C_engine.where(mask, loss_pos, loss_neg)
    return _apply_reduction(loss, reduction)


def poisson_nll_loss(
    x: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    """Poisson negative log-likelihood loss."""
    xi = _unwrap(x)
    ti = _unwrap(target)
    if log_input:
        # loss = exp(x) - target * x
        loss = _C_engine.sub(_C_engine.exp(xi), _C_engine.mul(ti, xi))
    else:
        # loss = x - target * log(x + eps)
        log_xeps = _C_engine.log(
            _C_engine.add(xi, _C_engine.full(xi.shape, eps, xi.dtype, xi.device))
        )
        loss = _C_engine.sub(xi, _C_engine.mul(ti, log_xeps))
    return _apply_reduction(loss, reduction)


def gaussian_nll_loss(
    x: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    """Gaussian negative log-likelihood loss.

    0.5 * (log(var) + (x - target)^2 / var)
    """
    xi = _unwrap(x)
    ti = _unwrap(target)
    vi = _C_engine.maximum(
        _unwrap(var),
        _C_engine.full(
            _unwrap(var).shape, eps, _unwrap(var).dtype, _unwrap(var).device
        ),
    )
    diff2 = _C_engine.square(_C_engine.sub(xi, ti))
    half = _C_engine.full(diff2.shape, 0.5, diff2.dtype, diff2.device)
    loss = _C_engine.mul(
        half,
        _C_engine.add(_C_engine.log(vi), _C_engine.div(diff2, vi)),
    )
    return _apply_reduction(loss, reduction)


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    """Connectionist Temporal Classification loss.

    Parameters
    ----------
    log_probs : Tensor
        Shape ``(T, N, C)`` — log-probabilities from the model.
    targets : Tensor
        Shape ``(N, S)`` or ``(sum(target_lengths),)`` int32 target indices.
    input_lengths : Tensor
        Shape ``(N,)`` int32 — valid sequence lengths.
    target_lengths : Tensor
        Shape ``(N,)`` int32 — target sequence lengths.
    CPU: forward DP in log-domain (Accelerate arithmetic).  GPU: CPU fallback.
    """
    from lucid._dispatch import _unwrap

    # Flatten targets to 1-D int32 if needed.
    tgt_impl = _unwrap(targets)
    if len(list(tgt_impl.shape)) > 1:
        tgt_impl = _C_engine.reshape(tgt_impl, [-1])

    # Ensure integer dtype for lengths and targets.
    def _to_i32(impl: object) -> object:
        if getattr(impl, "dtype", None) != _C_engine.I32:
            return _C_engine.cast(impl, _C_engine.I32)
        return impl

    tgt_impl = _to_i32(tgt_impl)
    il_impl = _to_i32(_unwrap(input_lengths))
    tl_impl = _to_i32(_unwrap(target_lengths))

    loss_t = _C_engine.nn.ctc_loss(
        _unwrap(log_probs), tgt_impl, il_impl, tl_impl, blank, zero_infinity
    )
    return _apply_reduction(loss_t, reduction)


def multi_margin_loss(
    x: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Multi-class margin loss (SVM-style) via engine tensor ops.

    loss(i) = sum_{j != y[i]} max(0, margin - x[i,y[i]] + x[i,j])^p / C
    """
    xi = _unwrap(x)
    ti = _unwrap(target)
    N = xi.shape[0]
    C = xi.shape[1]

    # Gather the score at the true class for each sample: (N, 1)
    ti_2d = _C_engine.reshape(ti, [N, 1])
    # gather along dim=1 → (N, 1) scores at true class
    correct = _C_engine.gather(xi, ti_2d, 1)  # (N, 1)
    correct_bc = _C_engine.broadcast_to(correct, [N, C])  # (N, C)

    # margin + x[i,j] - x[i,y[i]]  for every j
    margin_t = _C_engine.full([N, C], margin, xi.dtype, xi.device)
    diff = _C_engine.add(_C_engine.sub(margin_t, correct_bc), xi)  # (N, C)

    # Zero out the correct-class position: gather mask
    loss_nc = _C_engine.relu(diff)  # max(0, ...)

    if p > 1:
        loss_nc = _C_engine.pow_scalar(loss_nc, float(p))

    if weight is not None:
        # weight[y[i]] per sample: gather from weight vector
        wi = _unwrap(weight)
        w_per_sample = _C_engine.gather(
            _C_engine.reshape(wi, [1, C]), _C_engine.reshape(ti, [N, 1]), 1
        )  # (N,1)
        w_bc = _C_engine.broadcast_to(w_per_sample, [N, C])
        loss_nc = _C_engine.mul(loss_nc, w_bc)

    # Zero out the correct-class position using a scatter mask
    # Build (N, 1) zero update, scatter into a ones mask along dim=1
    ones_mask = _C_engine.ones([N, C], xi.dtype, xi.device)
    zeros_nc = _C_engine.zeros([N, 1], xi.dtype, xi.device)
    mask = _C_engine.scatter_add(ones_mask, ti_2d, zeros_nc, 1)
    # After scatter_add the target column has 1+0=1, others are 1 — invert:
    # We want to zero the target. Use: where(correct_class, 0, loss).
    # Simpler: multiply by (1 - one_hot(target))
    onehot_neg = _C_engine.scatter_add(
        _C_engine.zeros([N, C], xi.dtype, xi.device),
        ti_2d,
        _C_engine.ones([N, 1], xi.dtype, xi.device),
        1,
    )  # one-hot for target class
    keep_mask = _C_engine.sub(_C_engine.ones([N, C], xi.dtype, xi.device), onehot_neg)
    loss_nc = _C_engine.mul(loss_nc, keep_mask)

    # Sum over classes and divide by C
    c_t = _C_engine.full([N], float(C), xi.dtype, xi.device)
    loss_n = _C_engine.div(_C_engine.sum(loss_nc, [1], False), c_t)  # (N,)
    return _apply_reduction(loss_n, reduction)


def multilabel_margin_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Multi-label margin loss using engine ops.

    x: (N, C) or (C,) scores; target: same shape, -1 = padding.
    loss = sum_{positive t} sum_{j not in target} max(0, 1 - x[t] + x[j]) / C
    """
    xi = _unwrap(x)
    ti = _unwrap(target)

    # Handle 1D inputs
    if len(xi.shape) == 1:
        xi = _C_engine.reshape(xi, [1, xi.shape[0]])
        ti = _C_engine.reshape(ti, [1, ti.shape[0]])

    N, C = int(xi.shape[0]), int(xi.shape[1])

    # Build positive mask from target: pos_mask[i,j]=1 if target[i,k]==j for some k
    # Use: for each k, scatter 1 at position target[i,k] if target[i,k]>=0
    pos_mask = _C_engine.zeros([N, C], xi.dtype, xi.device)
    zeros_nc = _C_engine.zeros([N, 1], xi.dtype, xi.device)
    ones_nc = _C_engine.ones([N, 1], xi.dtype, xi.device)

    # Iterate over the K columns of target (K = C at most)
    for k in range(C):
        # target column k: (N,) → (N, 1) indices; skip -1 entries
        col_idx = _C_engine.gather(
            ti, _C_engine.full([N, 1], k, _C_engine.I32, ti.device), 1
        )  # (N,1)
        # Clamp negatives to 0 so scatter doesn't fail, weight by (idx >= 0)
        zero_i32 = _C_engine.zeros([N, 1], _C_engine.I32, ti.device)
        valid = _C_engine.greater_equal(col_idx, zero_i32)  # bool (N,1)
        safe_idx = _C_engine.where(valid, col_idx, zero_i32)  # clamp to 0
        # Convert valid to float for weighting
        val_f = _C_engine.where(
            valid, _C_engine.ones([N, 1], xi.dtype, xi.device), zeros_nc
        )
        pos_mask = _C_engine.scatter_add(pos_mask, safe_idx, val_f, 1)

    # Clamp to [0,1] to handle duplicates
    pos_mask = _C_engine.clip(pos_mask, 0.0, 1.0)

    # Negative mask = 1 - pos_mask
    neg_mask = _C_engine.sub(_C_engine.ones([N, C], xi.dtype, xi.device), pos_mask)

    # For each positive label t and each negative j: max(0, 1 - x[t] + x[j])
    # Broadcast: x_pos[i, j, k] = x[i, pos_k]; x_neg[i, j, k] = x[i, j]
    # Approximate via: sum_t pos_mask[i,t] * sum_j neg_mask[i,j] * max(0,1-x[i,t]+x[i,j])
    #
    # Use outer product via broadcasting:
    # x: (N, C) → x_t: (N, C, 1), x_j: (N, 1, C)
    x_t = _C_engine.reshape(xi, [N, C, 1])
    x_j = _C_engine.reshape(xi, [N, 1, C])
    pm_t = _C_engine.reshape(pos_mask, [N, C, 1])
    nm_j = _C_engine.reshape(neg_mask, [N, 1, C])

    margin_val = _C_engine.full([N, C, C], 1.0, xi.dtype, xi.device)
    diff = _C_engine.add(_C_engine.sub(margin_val, x_t), x_j)  # (N, C, C)
    hinge = _C_engine.relu(diff)  # max(0, ...)
    pm_bc = _C_engine.broadcast_to(pm_t, [N, C, C])
    nm_bc = _C_engine.broadcast_to(nm_j, [N, C, C])
    loss_tck = _C_engine.mul(_C_engine.mul(hinge, pm_bc), nm_bc)  # (N, C, C)
    # Sum over t and j, divide by C
    loss_n = _C_engine.div(
        _C_engine.sum(loss_tck, [1, 2], False),
        _C_engine.full([N], float(C), xi.dtype, xi.device),
    )  # (N,)
    return _apply_reduction(loss_n, reduction)


# ── P3 fills: soft_margin_loss / multilabel_soft_margin_loss ───────────────


def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Logistic-style binary loss: ``log(1 + exp(-target · input))``
    summed (or averaged) over all elements.  ``target`` is expected to
    hold ±1 sentinels, but the formula works for any real value.

    Implemented over ``softplus(-target · input)`` for numerical
    stability instead of ``log(1 + exp(...))`` directly.
    """
    raw = _lucid.nn.functional.softplus(-target * input)
    if reduction == "mean":
        return _lucid.mean(raw)
    if reduction == "sum":
        return _lucid.sum(raw)
    if reduction == "none":
        return raw
    raise ValueError(f"soft_margin_loss: unknown reduction={reduction!r}")


def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Multi-label sigmoid loss:
    ``-Σ_c [t_c · log σ(x_c) + (1 - t_c) · log(1 - σ(x_c))] / C``.

    Equivalent to ``binary_cross_entropy_with_logits`` averaged over the
    last (class) axis, then reduced over the batch.  ``weight`` rescales
    each class contribution element-wise before the per-sample mean.
    """
    # logσ(x)   = -softplus(-x);  log(1-σ(x)) = -softplus(x).  Both forms
    # are numerically stable for large |x|.
    log_sig = -_lucid.nn.functional.softplus(-input)
    log_one_minus_sig = -_lucid.nn.functional.softplus(input)
    per_class = -(target * log_sig + (1.0 - target) * log_one_minus_sig)
    if weight is not None:
        per_class = per_class * weight
    per_sample = _lucid.mean(per_class, dim=-1, keepdim=False)

    if reduction == "mean":
        return _lucid.mean(per_sample)
    if reduction == "sum":
        return _lucid.sum(per_sample)
    if reduction == "none":
        return per_sample
    raise ValueError(f"multilabel_soft_margin_loss: unknown reduction={reduction!r}")
