"""
nn.functional loss functions.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_REDUCTION_MAP = {"none": 0, "mean": 1, "sum": 2}


def mse_loss(x: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error loss."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.mse_loss(_unwrap(x), _unwrap(target), red))


def l1_loss(x: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean absolute error loss."""
    diff = _C_engine.abs(_C_engine.sub(_unwrap(x), _unwrap(target)))
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
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.huber_loss(_unwrap(x), _unwrap(target), delta, red))


def cross_entropy(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy loss for multi-class classification."""
    red = _REDUCTION_MAP.get(reduction, 1)
    w = _unwrap(weight) if weight is not None else None
    return _wrap(_C_engine.nn.cross_entropy_loss(_unwrap(x), _unwrap(target), red))


def nll_loss(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """Negative log-likelihood loss."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.nll_loss(_unwrap(x), _unwrap(target), red))


def binary_cross_entropy(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Binary cross-entropy loss."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.bce_loss(_unwrap(x), _unwrap(target), red))


def binary_cross_entropy_with_logits(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """BCE with logits loss (combines sigmoid + BCE for numerical stability)."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.bce_with_logits(_unwrap(x), _unwrap(target), red))


def kl_div(
    x: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    """Kullback-Leibler divergence."""
    if log_target:
        diff = _C_engine.sub(_unwrap(target), _unwrap(x))
        kl = _C_engine.mul(_C_engine.exp(_unwrap(target)), diff)
    else:
        log_x = _C_engine.log(_unwrap(x))
        diff = _C_engine.sub(_C_engine.log(_unwrap(target)), log_x)
        kl = _C_engine.mul(_unwrap(target), diff)
    if reduction == "mean":
        return _wrap(_C_engine.mean(kl, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(kl, [], False))
    return _wrap(kl)


def _apply_reduction(t, reduction: str):
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
    loss = _C_engine.relu(
        _C_engine.add(_C_engine.sub(d_ap, d_an), margin_t)
    )
    return _apply_reduction(loss, reduction)


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
    loss_pos = _C_engine.sub(ones, cos)                     # y=1
    loss_neg = _C_engine.relu(_C_engine.sub(cos, margin_t)) # y=-1
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
    loss_pos = xi                                             # y=1
    loss_neg = _C_engine.relu(_C_engine.sub(margin_t, xi))   # y=-1
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
        _C_engine.full(_unwrap(var).shape, eps, _unwrap(var).dtype, _unwrap(var).device),
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
    """Connectionist Temporal Classification loss (pure-Python reference).

    This is a reference implementation using the standard CTC forward
    algorithm.  It is not differentiable through the CTC computation itself.

    Parameters
    ----------
    log_probs : Tensor
        Shape ``(T, N, C)`` — log-probabilities from the model.
    targets : Tensor
        Shape ``(N, S)`` or ``(sum(target_lengths),)`` — target class indices.
    input_lengths : Tensor
        Shape ``(N,)`` — valid lengths of each sequence in *log_probs*.
    target_lengths : Tensor
        Shape ``(N,)`` — lengths of each target sequence.
    """
    import numpy as np

    lp_np = np.array(log_probs._impl.data_as_python()).reshape(log_probs.shape)  # (T, N, C)
    T, N, C = lp_np.shape

    tgt_flat = np.array(targets._impl.data_as_python(), dtype=np.int64).ravel()
    il_np = np.array(input_lengths._impl.data_as_python(), dtype=np.int64).ravel()
    tl_np = np.array(target_lengths._impl.data_as_python(), dtype=np.int64).ravel()

    losses = np.zeros(N, dtype=np.float32)
    offset = 0
    NEG_INF = -1e30

    for b in range(N):
        T_b = int(il_np[b])
        S = int(tl_np[b])
        t_seq = tgt_flat[offset: offset + S]
        offset += S

        # Extended label sequence: blank, t[0], blank, t[1], ..., blank
        L = 2 * S + 1
        ext = np.full(L, blank, dtype=np.int64)
        ext[1::2] = t_seq

        alpha = np.full((T_b, L), NEG_INF, dtype=np.float64)
        alpha[0, 0] = lp_np[0, b, ext[0]]
        if L > 1:
            alpha[0, 1] = lp_np[0, b, ext[1]]

        for t in range(1, T_b):
            for s in range(L):
                a = alpha[t - 1, s]
                if s > 0:
                    a = np.logaddexp(a, alpha[t - 1, s - 1])
                if s > 1 and ext[s] != ext[s - 2]:
                    a = np.logaddexp(a, alpha[t - 1, s - 2])
                alpha[t, s] = a + lp_np[t, b, ext[s]]

        end = alpha[T_b - 1, L - 1]
        if L >= 2:
            end = np.logaddexp(end, alpha[T_b - 1, L - 2])
        v = -float(end)
        if zero_infinity and (np.isinf(v) or np.isnan(v)):
            v = 0.0
        losses[b] = v

    loss_t = _C_engine.TensorImpl(losses, _C_engine.CPU, False)
    return _apply_reduction(loss_t, reduction)
