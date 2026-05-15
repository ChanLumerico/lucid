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
    r"""Mean-squared-error (L2) loss between input and target.

    The workhorse loss for regression problems.  Penalises large
    errors *quadratically*, which makes it sensitive to outliers but
    yields a well-conditioned optimisation surface — the gradient is
    linear in the residual, so SGD updates scale gracefully near the
    optimum.  Compare with :func:`l1_loss` (constant gradient,
    robust to outliers) and :func:`huber_loss` (a smooth blend of
    the two).

    Parameters
    ----------
    x : Tensor
        Predicted values, any shape.
    target : Tensor
        Target values; must be broadcast-compatible with ``x``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar loss for ``"mean"`` / ``"sum"``, or a per-element
        tensor with ``x``'s shape for ``"none"``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = (x_i - y_i)^2

    With reduction:

    .. math::

        L = \begin{cases}
            \tfrac{1}{N}\sum_i L_i & \text{``mean''} \\
            \sum_i L_i & \text{``sum''} \\
            L_i & \text{``none''}
        \end{cases}

    The gradient w.r.t. ``x`` is :math:`2(x_i - y_i) / N` under
    ``"mean"`` reduction — proportional to the error, so updates
    naturally shrink near the optimum.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import mse_loss
    >>> pred = lucid.tensor([1.0, 2.0, 3.0])
    >>> target = lucid.tensor([1.5, 2.5, 2.5])
    >>> mse_loss(pred, target)
    Tensor(0.25)
    """
    _validate_reduction(reduction)
    red: int = _REDUCTION_MAP[reduction]
    return _wrap(_C_engine.nn.mse_loss(_unwrap(x), _unwrap(target), red))


def l1_loss(x: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    r"""Mean-absolute-error (L1) loss between input and target.

    Robust regression loss whose gradient is constant in magnitude
    (:math:`\pm 1`) and therefore *insensitive to outliers* —
    extreme residuals do not dominate the update direction as they
    do under :func:`mse_loss`.  The trade-off is a non-smooth
    optimisation surface at :math:`x = y` (sub-gradient at zero),
    which can produce slightly slower convergence for small errors.
    Often paired with :func:`smooth_l1_loss` to recover smoothness
    while keeping outlier robustness.

    Parameters
    ----------
    x : Tensor
        Predicted values, any shape.
    target : Tensor
        Target values; broadcast-compatible with ``x``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar loss for ``"mean"`` / ``"sum"``, or a per-element
        tensor with ``x``'s shape for ``"none"``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = |x_i - y_i|

    Gradient w.r.t. ``x`` is :math:`\operatorname{sign}(x_i - y_i)`
    (zero at the origin).  Used heavily in image-to-image regression
    (super-resolution, denoising) where outlier-robustness matters.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import l1_loss
    >>> pred = lucid.tensor([1.0, 2.0, 3.0])
    >>> target = lucid.tensor([1.5, 2.5, 2.5])
    >>> l1_loss(pred, target)
    Tensor(0.5)
    """
    _validate_reduction(reduction)
    diff: _C_engine.TensorImpl = _C_engine.abs(
        _C_engine.sub(_unwrap(x), _unwrap(target))
    )
    if reduction == "mean":
        return _wrap(_C_engine.mean(diff, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(diff, [], False))
    return _wrap(diff)


def smooth_l1_loss(
    x: Tensor, target: Tensor, beta: float = 1.0, reduction: str = "mean"
) -> Tensor:
    r"""Smooth L1 loss — a quadratic-near-zero, linear-far-from-zero hybrid.

    Combines the best of :func:`mse_loss` and :func:`l1_loss`: the
    quadratic region near the origin gives a smooth gradient and
    fast convergence, while the linear tails make the loss robust
    to outliers.  This is the standard regression head used in
    Fast R-CNN-style object detection (bounding-box regression),
    where outlier bounding boxes would otherwise dominate the
    training signal.

    The function is a thin wrapper around :func:`huber_loss` with
    ``delta = beta`` and an additional ``1/beta`` scaling inside the
    quadratic region (so the loss has unit slope at the transition).

    Parameters
    ----------
    x : Tensor
        Predicted values, any shape.
    target : Tensor
        Target values; broadcast-compatible with ``x``.
    beta : float, optional
        Transition point between quadratic and linear regions
        (default ``1.0``).  Smaller ``beta`` makes the loss behave
        more like :func:`l1_loss`; larger ``beta`` makes it behave
        more like :func:`mse_loss`.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar (``"mean"``/``"sum"``) or full-shape (``"none"``).

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = \begin{cases}
            \tfrac{1}{2}(x_i - y_i)^2 / \beta & |x_i - y_i| < \beta \\
            |x_i - y_i| - \tfrac{1}{2}\beta & \text{otherwise}
        \end{cases}

    Continuously differentiable everywhere; gradient is
    :math:`(x_i - y_i)/\beta` inside the quadratic region and
    :math:`\operatorname{sign}(x_i - y_i)` outside.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import smooth_l1_loss
    >>> pred = lucid.tensor([0.0, 2.0])
    >>> target = lucid.tensor([0.5, 5.0])
    >>> smooth_l1_loss(pred, target, beta=1.0)
    Tensor(1.3125)
    """
    return huber_loss(x, target, delta=beta, reduction=reduction)


def huber_loss(
    x: Tensor, target: Tensor, delta: float = 1.0, reduction: str = "mean"
) -> Tensor:
    r"""Huber loss — robust regression with a tunable transition point.

    Identical in shape to :func:`smooth_l1_loss` but parameterised
    by the *slope-clip* point :math:`\delta` rather than the
    quadratic-region scale :math:`\beta`.  Inside the
    :math:`|x-y| < \delta` region the loss is quadratic; outside,
    the gradient saturates at :math:`\pm\delta` so a single huge
    residual cannot dominate the update.

    Originally proposed by Peter Huber (1964) as the maximum-
    likelihood estimator for a contaminated Gaussian model — i.e.,
    "mostly Gaussian noise but with occasional outliers".  Use it
    when you suspect a small fraction of your residuals come from
    a heavy-tailed distribution.

    Parameters
    ----------
    x : Tensor
        Predicted values.
    target : Tensor
        Target values; broadcast-compatible with ``x``.
    delta : float, optional
        Threshold at which the loss transitions from quadratic to
        linear (default ``1.0``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape, per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = \begin{cases}
            \tfrac{1}{2}(x_i - y_i)^2 & |x_i - y_i| \le \delta \\
            \delta\,\big(|x_i - y_i| - \tfrac{1}{2}\delta\big) & \text{otherwise}
        \end{cases}

    Unlike :func:`smooth_l1_loss`, the quadratic region is *not*
    rescaled by :math:`1/\delta`, so the loss magnitude itself
    grows with :math:`\delta`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import huber_loss
    >>> pred = lucid.tensor([0.0, 5.0])
    >>> target = lucid.tensor([0.5, 0.0])
    >>> huber_loss(pred, target, delta=1.0)
    Tensor(2.3125)
    """
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
    r"""Cross-entropy loss for multi-class classification.

    The canonical training objective for categorical classifiers.
    Combines :func:`~lucid.nn.functional.log_softmax` and
    :func:`nll_loss` into a single, numerically stable expression —
    operating directly on raw logits avoids the catastrophic
    cancellation that arises when softmax probabilities are taken
    to ``log()``.  Implements the full contract: per-class
    ``weight`` rescaling, ``ignore_index`` masking, and
    label-smoothing regularisation.

    Parameters
    ----------
    x : Tensor
        Raw logits of shape :math:`(N, C)` or :math:`(N, C, d_1, \dots, d_k)`.
    target : Tensor
        Either integer class indices of shape :math:`(N,)` /
        :math:`(N, d_1, \dots, d_k)` or per-class probabilities of
        shape matching ``x``.
    weight : Tensor or None, optional
        Per-class weight vector of shape :math:`(C,)` — useful for
        class-imbalanced training.
    ignore_index : int, optional
        Class index whose samples are skipped entirely (default ``-100``).
        Common for masked / padded targets in sequence models.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.  Under
        ``"mean"``, the divisor is the sum of effective sample weights
        (after ``weight`` and ``ignore_index``), not the raw element
        count.
    label_smoothing : float, optional
        Interpolation factor :math:`\alpha \in [0, 1)` between hard
        one-hot targets and a uniform distribution
        (Szegedy et al. 2016).  Acts as a regulariser by discouraging
        over-confident predictions.

    Returns
    -------
    Tensor
        Scalar (``"mean"``/``"sum"``) or per-sample tensor (``"none"``).

    Notes
    -----
    Per-sample loss:

    .. math::

        L_i = -\sum_c w_c \, y_{i,c} \, \log \mathrm{softmax}(x_i)_c

    where :math:`y_{i,c}` is the (smoothed) target distribution.
    With ``label_smoothing = \alpha``,

    .. math::

        y_{i,c} = (1-\alpha)\,\mathbb{1}[c = t_i] + \alpha / C.

    Gradient w.r.t. the logits has the well-known clean form
    :math:`\mathrm{softmax}(x_i) - y_i` (up to weighting), which is
    the reason cross-entropy is preferred over MSE for classification:
    no sigmoid saturation, no vanishing gradient.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import cross_entropy
    >>> logits = lucid.tensor([[2.0, 0.5, 0.1], [0.0, 1.5, 0.2]])
    >>> target = lucid.tensor([0, 1])
    >>> cross_entropy(logits, target)
    Tensor(0.3490...)
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
    r"""Negative log-likelihood loss for multi-class classification.

    The "back-half" of :func:`cross_entropy`: assumes the input is
    already a tensor of log-probabilities (typically produced by
    :func:`~lucid.nn.functional.log_softmax`).  Provided as a
    separate entry-point so models that need log-probabilities for
    downstream use (e.g., beam search) can avoid recomputing them.

    Parameters
    ----------
    x : Tensor
        Log-probabilities of shape :math:`(N, C)` or
        :math:`(N, C, d_1, \dots, d_k)`.
    target : Tensor
        Integer class indices of shape :math:`(N,)` /
        :math:`(N, d_1, \dots, d_k)`.
    weight : Tensor or None, optional
        Per-class weight vector :math:`(C,)`.
    ignore_index : int, optional
        Class index whose samples are excluded (default ``-100``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-sample tensor depending on ``reduction``.

    Notes
    -----
    Per-sample loss:

    .. math::

        L_i = -w_{t_i}\,x_{i, t_i}

    Under ``"mean"`` reduction, the divisor is the sum of effective
    sample weights — i.e., :math:`\sum_i w_{t_i}\,\mathbb{1}[t_i \ne \text{ignore}]`
    — not the raw count.  This matches the standard convention so
    that the loss is invariant to a global rescaling of ``weight``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import nll_loss, log_softmax
    >>> logits = lucid.tensor([[2.0, 0.5, 0.1], [0.0, 1.5, 0.2]])
    >>> target = lucid.tensor([0, 1])
    >>> nll_loss(log_softmax(logits, dim=1), target)
    Tensor(0.3490...)
    """
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
    r"""Binary cross-entropy between predicted probabilities and targets.

    The standard objective for binary classification and for
    multi-label classification with independent class predictions.
    Operates on *probabilities* in :math:`(0, 1)` — typically the
    output of a :func:`~lucid.nn.functional.sigmoid`.  When working
    with raw logits, prefer :func:`binary_cross_entropy_with_logits`
    for numerical stability.

    Inputs are clamped to :math:`[\varepsilon, 1 - \varepsilon]`
    with :math:`\varepsilon = 10^{-12}` before the logarithm to
    prevent ``-inf`` gradients at the boundaries.

    Parameters
    ----------
    x : Tensor
        Predicted probabilities in :math:`(0, 1)`, any shape.
    target : Tensor
        Target probabilities (typically binary) of the same shape.
    weight : Tensor or None, optional
        Element-wise rescaling factor (broadcast-compatible with
        ``x``).  Use to up-weight rare classes or hard examples.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = -\big(y_i\,\log p_i + (1 - y_i)\,\log(1 - p_i)\big)

    Gradient w.r.t. ``x`` is :math:`(p_i - y_i) / (p_i(1 - p_i))`,
    which diverges as :math:`p_i \to 0` or :math:`1` — the reason
    the logits-form (which yields a clean :math:`\sigma(x) - y`
    gradient) is preferred when training stability is critical.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import binary_cross_entropy
    >>> p = lucid.tensor([0.9, 0.2, 0.7])
    >>> y = lucid.tensor([1.0, 0.0, 1.0])
    >>> binary_cross_entropy(p, y)
    Tensor(0.2284...)
    """
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
    r"""Binary cross-entropy from raw logits (numerically stable).

    Mathematically equivalent to
    ``binary_cross_entropy(sigmoid(x), target)`` but evaluated in a
    log-sum-exp-style form that avoids overflow/underflow when
    ``|x|`` is large.  This is the preferred binary classification
    loss for training — composing a separate sigmoid with BCE risks
    catastrophic cancellation in :math:`\log(1 - \sigma(x))` for
    large positive logits.

    Parameters
    ----------
    x : Tensor
        Raw logits (un-bounded reals), any shape.
    target : Tensor
        Target probabilities (typically binary), same shape as ``x``.
    weight : Tensor or None, optional
        Element-wise rescaling factor.
    pos_weight : Tensor or None, optional
        Per-class weight applied to the *positive* term only —
        useful for highly-imbalanced binary tasks, where setting
        ``pos_weight = n_neg / n_pos`` recovers the prevalence-
        balanced gradient.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    The numerically stable base form is

    .. math::

        L_i = \max(x_i, 0) - x_i\,y_i + \log\!\big(1 + e^{-|x_i|}\big),

    equivalent to :math:`-(y\log\sigma(x) + (1-y)\log(1-\sigma(x)))`
    but free of overflow.  With ``pos_weight``:

    .. math::

        L_i = (1 - y_i)\,x_i + \big(1 + (w^{+} - 1) y_i\big)
              \big(\log(1 + e^{-|x_i|}) + \max(-x_i, 0)\big).

    Gradient w.r.t. ``x`` is the clean :math:`\sigma(x_i) - y_i`
    (modulo weighting) — the canonical reason this form is used in
    practice instead of the explicit sigmoid + BCE composition.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import binary_cross_entropy_with_logits
    >>> logits = lucid.tensor([2.0, -1.0, 0.5])
    >>> target = lucid.tensor([1.0, 0.0, 1.0])
    >>> binary_cross_entropy_with_logits(logits, target)
    Tensor(0.3567...)
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
    r"""Kullback-Leibler divergence between two distributions.

    Measures the "information gain" from approximating distribution
    :math:`p` (the target) with distribution :math:`q` (the input).
    Used heavily in knowledge distillation (matching student logits
    to a teacher's), variational inference (the ELBO's KL term),
    and policy-gradient regularisation in RL.

    The convention here matches the reference framework: ``x`` is
    :math:`\log q` (log-predicted), and ``target`` is :math:`p`
    (target probability) — or :math:`\log p` if ``log_target=True``.
    Note this is asymmetric in its arguments — KL is *not* a metric.

    Parameters
    ----------
    x : Tensor
        Log-probabilities of the *predicted* distribution
        :math:`\log q`, any shape.
    target : Tensor
        Probabilities of the *target* distribution :math:`p`,
        or its log when ``log_target=True``.  Same shape as ``x``.
    size_average : bool or None, optional
        Deprecated.  Retained for signature compatibility; ignored
        — use ``reduction`` instead.
    reduction : str, optional
        ``"none"``, ``"mean"``, ``"sum"``, or ``"batchmean"``
        (default ``"mean"``).  ``"batchmean"`` divides the summed
        loss by the leading (batch) dimension and is the *only*
        reduction that yields the mathematically correct KL value
        in expectation.
    log_target : bool, optional
        When ``True``, treat ``target`` as already-logged
        (:math:`\log p`).  This often avoids a redundant
        :math:`\log` / :math:`\exp` round-trip.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = p_i \cdot (\log p_i - \log q_i)

    Globally:

    .. math::

        D_{\mathrm{KL}}(p \,\|\, q) = \sum_i p_i \log \tfrac{p_i}{q_i} \ge 0,

    with equality iff :math:`p = q` almost everywhere.  The standard
    ``"mean"`` reduction divides by the element count, not the
    batch size, so it under-reports the divergence value — prefer
    ``"batchmean"`` whenever the absolute scale matters.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import kl_div, log_softmax
    >>> log_q = log_softmax(lucid.tensor([[2.0, 0.5, 0.1]]), dim=1)
    >>> p = lucid.tensor([[0.8, 0.15, 0.05]])
    >>> kl_div(log_q, p, reduction="batchmean")
    Tensor(0.0641...)
    """
    _validate_reduction(reduction, allow_batchmean=True)
    # `x` is log_q (log of predicted probability) per the standard contract.
    # When log_target=False, target is the raw probability p; when True it
    # is log(p).  Loss elementwise = target * (log(target) - log_q).
    xi: _C_engine.TensorImpl = _unwrap(x)
    ti: _C_engine.TensorImpl = _unwrap(target)
    if log_target:
        # log_target=True → target itself is log(p); use exp(t) as the weight.
        diff: _C_engine.TensorImpl = _C_engine.sub(ti, xi)
        kl: _C_engine.TensorImpl = _C_engine.mul(_C_engine.exp(ti), diff)
    else:
        # Standard: target * (log(target) − x).
        diff = _C_engine.sub(_C_engine.log(ti), xi)
        kl = _C_engine.mul(ti, diff)
    if reduction == "mean":
        return _wrap(_C_engine.mean(kl, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(kl, [], False))
    if reduction == "batchmean":
        total: _C_engine.TensorImpl = _C_engine.sum(kl, [], False)
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
    r"""Triplet margin loss for metric learning.

    Pulls an "anchor" embedding toward a "positive" example (same
    class / matching pair) and pushes it away from a "negative"
    example by at least a fixed ``margin``.  This is the workhorse
    objective for learning embeddings used in face verification
    (FaceNet), image retrieval, and contrastive representation
    learning.

    Optionally applies the "anchor-swap" trick of Balntas et al.
    2016 — when ``d(p, n) < d(a, n)`` the positive is closer to the
    negative than the anchor is, so we swap roles and use the
    *harder* of the two distances, focusing gradient on the more
    informative triplet.

    Parameters
    ----------
    anchor : Tensor
        Embedding of shape :math:`(N, D)`.
    positive : Tensor
        Positive sample embedding of the same shape.
    negative : Tensor
        Negative sample embedding of the same shape.
    margin : float, optional
        Minimum desired gap between positive and negative distances
        (default ``1.0``).  Triplets satisfying the margin already
        receive zero loss / zero gradient.
    p : float, optional
        Norm degree of the pairwise distance (default ``2.0`` —
        Euclidean).
    eps : float, optional
        Numerical floor inside the distance to avoid zero-derivative
        at coincident points (default ``1e-6``).
    swap : bool, optional
        Enable the Balntas-anchor-swap trick (default ``False``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-triplet tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-triplet loss:

    .. math::

        L_i = \max\!\big(0,\; d(a_i, p_i) - d(a_i, n_i) + \text{margin}\big),

    where :math:`d(\cdot, \cdot)` is the :math:`L_p` pairwise
    distance.  Easy triplets (already separated by ``margin``)
    contribute exactly zero — only "semi-hard" / "hard" triplets
    drive the update, which is why batch mining strategies matter
    so much in practice.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import triplet_margin_loss
    >>> a = lucid.tensor([[1.0, 0.0]])
    >>> p = lucid.tensor([[1.0, 0.1]])
    >>> n = lucid.tensor([[0.0, 1.0]])
    >>> triplet_margin_loss(a, p, n, margin=1.0)
    Tensor(0.6862...)
    """
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
    r"""Triplet margin loss with a user-supplied distance function.

    Identical in form to :func:`triplet_margin_loss` but lets the
    caller plug in any binary distance callable — useful when the
    embedding space is non-Euclidean (e.g., learned Mahalanobis
    distances, cosine distance, or hyperbolic embeddings).  When
    no ``distance_function`` is supplied it defaults to the
    :math:`L_2` pairwise distance, matching the reference framework
    semantics.

    Parameters
    ----------
    anchor : Tensor
        Anchor embedding of shape :math:`(N, D)`.
    positive : Tensor
        Positive sample embedding of the same shape.
    negative : Tensor
        Negative sample embedding of the same shape.
    distance_function : callable or None, optional
        Function ``(x, y) -> Tensor`` returning a non-negative
        distance of shape :math:`(N,)`.  Defaults to :math:`L_2`
        pairwise distance.
    margin : float, optional
        Minimum desired margin between positive and negative
        distances (default ``1.0``).
    swap : bool, optional
        Enable the Balntas-2016 anchor-swap heuristic: replace
        :math:`d(a, n)` with :math:`\min\!\big(d(a, n), d(p, n)\big)`
        so the harder negative drives the gradient (default ``False``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-triplet tensor.

    Notes
    -----
    Per-triplet loss:

    .. math::

        L_i = \max\!\big(0,\; d(a_i, p_i) - d(a_i, n_i) + \text{margin}\big)

    The Lucid module wrapper
    :class:`lucid.nn.TripletMarginWithDistanceLoss` forwards into
    this function; both surfaces are valid entry-points.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import (
    ...     triplet_margin_with_distance_loss,
    ...     pairwise_distance,
    ... )
    >>> def manhattan(a, b):
    ...     return pairwise_distance(a, b, p=1.0)
    >>> a = lucid.tensor([[1.0, 0.0]])
    >>> p = lucid.tensor([[1.0, 0.1]])
    >>> n = lucid.tensor([[0.0, 1.0]])
    >>> triplet_margin_with_distance_loss(a, p, n, distance_function=manhattan)
    Tensor(0.2)
    """
    from lucid.nn.functional.activations import pairwise_distance

    df: object = distance_function
    if df is None:

        def df(a: Tensor, b: Tensor) -> Tensor:
            """Derivative helper used inside the loss-function gradient computation."""
            return pairwise_distance(a, b, p=2.0)

    d_ap: Tensor = df(anchor, positive)
    d_an: Tensor = df(anchor, negative)
    if swap:
        d_pn: Tensor = df(positive, negative)
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
    r"""Cosine embedding loss for pairwise similarity learning.

    Encourages "similar" pairs (label :math:`y = 1`) to align in
    direction and "dissimilar" pairs (:math:`y = -1`) to be at
    least ``margin`` cosine-units apart.  Operates purely on
    angular (direction) information — magnitudes are normalised
    away, which is useful when the relevant signal is the
    *direction* of embeddings (e.g., word vectors, learned
    representations).

    Parameters
    ----------
    x1 : Tensor
        First embedding of shape :math:`(N, D)`.
    x2 : Tensor
        Second embedding of the same shape.
    y : Tensor
        Label tensor of shape :math:`(N,)` with values :math:`\pm 1`.
    margin : float, optional
        Minimum desired cosine gap for dissimilar pairs, typically
        in :math:`[-1, 1]` (default ``0.0``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-pair tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-pair loss:

    .. math::

        L_i = \begin{cases}
            1 - \cos(x_1^{(i)}, x_2^{(i)}) & y_i = +1 \\
            \max\!\big(0,\; \cos(x_1^{(i)}, x_2^{(i)}) - \text{margin}\big) & y_i = -1
        \end{cases}

    With ``margin = 0``, dissimilar pairs are only penalised when
    they have positive cosine similarity — i.e., the loss is
    satisfied as long as the angle between them exceeds :math:`90°`.
    Increasing ``margin`` toward 1 demands more separation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import cosine_embedding_loss
    >>> x1 = lucid.tensor([[1.0, 0.0]])
    >>> x2 = lucid.tensor([[0.5, 0.5]])
    >>> y = lucid.tensor([1.0])
    >>> cosine_embedding_loss(x1, x2, y)
    Tensor(0.2928...)
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
    r"""Pairwise ranking hinge loss.

    Trains a scoring function so that for each pair :math:`(x_1, x_2)`
    the *signed* score gap :math:`y\,(x_1 - x_2)` exceeds the
    ``margin``.  Used for learning-to-rank (search, recommendation),
    Bradley-Terry style preference modelling, and reward-model
    training for RLHF — wherever the supervision signal is a
    pairwise preference rather than a target value.

    Parameters
    ----------
    x1 : Tensor
        Scores for the first item of each pair, any shape.
    x2 : Tensor
        Scores for the second item, same shape as ``x1``.
    y : Tensor
        Pairwise preference label :math:`\pm 1`: :math:`+1` if
        :math:`x_1` should rank higher, :math:`-1` otherwise.
    margin : float, optional
        Required minimum score gap (default ``0.0``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-pair tensor.

    Notes
    -----
    Per-pair loss:

    .. math::

        L_i = \max\!\big(0,\; -y_i\,(x_1^{(i)} - x_2^{(i)}) + \text{margin}\big)

    Pairs already satisfying the margin (:math:`y(x_1 - x_2) \ge \text{margin}`)
    contribute zero loss and zero gradient — the hinge structure
    naturally focuses learning on the violating pairs, akin to a
    pairwise SVM.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import margin_ranking_loss
    >>> s1 = lucid.tensor([2.0, 0.5])
    >>> s2 = lucid.tensor([1.0, 1.0])
    >>> y = lucid.tensor([1.0, 1.0])
    >>> margin_ranking_loss(s1, s2, y, margin=1.0)
    Tensor(0.75)
    """
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
    r"""Hinge embedding loss.

    Designed for similarity learning on pre-computed distances:
    given a (typically non-negative) score ``x`` representing a
    pairwise distance, push positive pairs (label :math:`+1`)
    toward small distances and negative pairs (label :math:`-1`)
    above a fixed ``margin``.  Common in Siamese network training
    and energy-based dissimilarity models.

    Parameters
    ----------
    x : Tensor
        Per-pair score (distance) tensor, any shape.
    y : Tensor
        Label tensor :math:`\pm 1` with the same shape as ``x``.
    margin : float, optional
        Margin enforced for negative pairs (default ``1.0``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = \begin{cases}
            x_i & y_i = +1 \\
            \max(0,\; \text{margin} - x_i) & y_i = -1
        \end{cases}

    The positive branch simply minimises the distance; the negative
    branch is a one-sided hinge that pushes apart only those pairs
    whose distance is below the margin — pairs already far apart
    contribute nothing.  This asymmetric structure prevents the
    loss from collapsing all embeddings into a single point.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hinge_embedding_loss
    >>> dist = lucid.tensor([0.2, 0.8])
    >>> y = lucid.tensor([1.0, -1.0])
    >>> hinge_embedding_loss(dist, y, margin=1.0)
    Tensor(0.2)
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
    r"""Poisson negative log-likelihood loss for count regression.

    The maximum-likelihood objective when targets are non-negative
    integer counts modelled as :math:`y \sim \mathrm{Poisson}(\lambda)`.
    Standard for forecasting tasks (web clicks, event counts,
    biological cell counts) where the variance scales with the
    mean.  Unlike :func:`mse_loss`, this loss respects the
    heteroscedasticity inherent in count data.

    Parameters
    ----------
    x : Tensor
        Predicted Poisson rate.  By default (``log_input=True``)
        treated as :math:`\log \lambda` for numerical stability;
        set ``log_input=False`` to pass the rate :math:`\lambda`
        directly.
    target : Tensor
        Observed counts, broadcast-compatible with ``x``.
    log_input : bool, optional
        Whether ``x`` is :math:`\log \lambda` (default) or
        :math:`\lambda`.  The log-form avoids exponentiating an
        unbounded prediction in inner loops.
    full : bool, optional
        Include the Stirling approximation term
        :math:`\log(y!) \approx y\log y - y + \tfrac{1}{2}\log(2\pi y)`
        in the loss.  Has no effect on gradients (constant in
        ``x``) but yields the correct log-likelihood value.  Not
        currently added — kept for API parity.
    eps : float, optional
        Small constant added before :math:`\log` when
        ``log_input=False`` (default ``1e-8``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss (constant-in-:math:`x` terms dropped):

    .. math::

        L_i = \begin{cases}
            e^{x_i} - y_i\,x_i & \text{log\_input = True} \\
            x_i - y_i \log(x_i + \varepsilon) & \text{log\_input = False}
        \end{cases}

    Gradient w.r.t. :math:`x` is :math:`e^x - y` (log-input form)
    or :math:`1 - y/(x + \varepsilon)` (rate form).  Both push
    :math:`\lambda` toward :math:`y` in expectation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import poisson_nll_loss
    >>> log_lam = lucid.tensor([0.0, 1.0, 2.0])
    >>> y = lucid.tensor([1.0, 2.0, 5.0])
    >>> poisson_nll_loss(log_lam, y, log_input=True)
    Tensor(0.7299...)
    """
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
    r"""Gaussian negative log-likelihood for heteroscedastic regression.

    Maximum-likelihood objective when the prediction is a
    *distribution* :math:`\mathcal{N}(\mu, \sigma^2)` over the
    target rather than a point estimate.  Training a network with
    two heads (one for :math:`\mu`, one for :math:`\sigma^2`)
    against this loss recovers calibrated predictive uncertainty
    — useful for active learning, decision-aware regression, and
    Bayesian deep ensembles.

    The variance ``var`` is clamped below by ``eps`` to prevent
    division by zero and runaway log-terms when the network
    initially predicts near-zero variance.

    Parameters
    ----------
    x : Tensor
        Predicted means :math:`\mu`, any shape.
    target : Tensor
        Observed values :math:`y`, broadcast-compatible with ``x``.
    var : Tensor
        Predicted variances :math:`\sigma^2 > 0`, broadcast-
        compatible with ``x``.
    full : bool, optional
        Include the constant :math:`\tfrac{1}{2}\log(2\pi)` term in
        the loss value.  Has no effect on gradients; useful only
        for reporting log-likelihoods.  Not currently added.
    eps : float, optional
        Lower bound applied to ``var`` for numerical stability
        (default ``1e-6``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss (constant terms dropped):

    .. math::

        L_i = \tfrac{1}{2}\!\left(\log \sigma_i^2 + \frac{(y_i - \mu_i)^2}{\sigma_i^2}\right)

    The first term penalises over-confidence (small variance), the
    second term rewards accuracy weighted by precision.  Together
    they give the model a clean trade-off: when it cannot reduce
    :math:`(y-\mu)^2`, increasing :math:`\sigma^2` decreases the
    loss — this is what produces calibrated uncertainty estimates.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import gaussian_nll_loss
    >>> mu = lucid.tensor([0.0, 1.0])
    >>> y = lucid.tensor([0.5, 1.0])
    >>> var = lucid.tensor([1.0, 0.25])
    >>> gaussian_nll_loss(mu, y, var)
    Tensor(-0.2218...)
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
    r"""Connectionist Temporal Classification (CTC) loss.

    The standard training objective for unaligned sequence
    prediction — used in speech recognition, handwriting
    recognition, and any task where the input sequence is much
    longer than the target and no per-frame alignment is provided.
    Introduced by Graves et al. 2006.

    Internally marginalises over every valid alignment of a
    :math:`T`-frame prediction onto an :math:`S`-symbol target by
    inserting "blank" symbols and allowing each target symbol to
    span one or more frames, computing the negative log of the
    total path probability via dynamic programming.

    Parameters
    ----------
    log_probs : Tensor
        Log-probabilities of shape :math:`(T, N, C)` where
        :math:`T` is the input sequence length, :math:`N` is the
        batch size, and :math:`C` is the number of classes
        (including the blank).  Typically produced by
        :func:`~lucid.nn.functional.log_softmax` over the class
        axis.
    targets : Tensor
        Target indices, shape :math:`(N, S)` (padded) or
        :math:`(\sum_i \text{target\_lengths}_i,)` (concatenated).
        ``int32``.
    input_lengths : Tensor
        Effective input lengths :math:`(N,)`, ``int32``.  Enables
        padding-aware batching.
    target_lengths : Tensor
        Effective target lengths :math:`(N,)`, ``int32``.
    blank : int, optional
        Index of the blank symbol (default ``0``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.  Under
        ``"mean"``, the per-sample loss is averaged across the
        batch.
    zero_infinity : bool, optional
        When ``True``, infinite losses (which arise when a target
        cannot fit in the available input frames) and their
        gradients are set to zero, effectively skipping those
        samples (default ``False``).

    Returns
    -------
    Tensor
        Scalar (``"mean"`` / ``"sum"``) or per-sample tensor of
        shape :math:`(N,)`.

    Notes
    -----
    The CTC objective is the negative log of the total alignment
    probability:

    .. math::

        L = -\log \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})}
            \prod_{t=1}^{T} p_t(\pi_t),

    where :math:`\mathcal{B}` is the "many-to-one" alignment map
    that collapses repeats and removes blanks.  The forward DP runs
    in log-domain (Accelerate arithmetic on CPU); the GPU stream
    currently falls back to CPU.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import ctc_loss, log_softmax
    >>> # T=4 frames, N=1 batch, C=3 classes (blank=0)
    >>> logits = lucid.randn(4, 1, 3)
    >>> log_p = log_softmax(logits, dim=2)
    >>> targets = lucid.tensor([[1, 2]], dtype=lucid.int32)
    >>> il = lucid.tensor([4], dtype=lucid.int32)
    >>> tl = lucid.tensor([2], dtype=lucid.int32)
    >>> ctc_loss(log_p, targets, il, tl)  # doctest: +SKIP
    Tensor(...)
    """
    from lucid._dispatch import _unwrap

    # Flatten targets to 1-D int32 if needed.
    tgt_impl = _unwrap(targets)
    if len(list(tgt_impl.shape)) > 1:
        tgt_impl = _C_engine.reshape(tgt_impl, [-1])

    # Ensure integer dtype for lengths and targets.
    def _to_i32(impl: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
        if getattr(impl, "dtype", None) != _C_engine.I32:
            return _C_engine.astype(impl, _C_engine.I32)
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
    r"""Multi-class hinge (margin) loss — Crammer-Singer SVM objective.

    A non-probabilistic alternative to :func:`cross_entropy` for
    multi-class classification: instead of fitting a softmax
    distribution, it requires the true-class score to exceed every
    other class score by at least ``margin``.  Frequently used in
    structured prediction and as a drop-in for hinge-style losses
    in metric learning.

    Parameters
    ----------
    x : Tensor
        Class scores of shape :math:`(N, C)`.
    target : Tensor
        Integer class indices of shape :math:`(N,)`.
    p : int, optional
        Power applied to each hinge term — ``1`` for the standard
        hinge loss, ``2`` for the smoother squared-hinge variant
        (default ``1``).
    margin : float, optional
        Required minimum score gap between the true class and
        every competitor (default ``1.0``).
    weight : Tensor or None, optional
        Per-class weight vector of shape :math:`(C,)`.  Each sample
        contribution is scaled by the weight of its *true* class.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-sample tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-sample loss:

    .. math::

        L_i = \frac{1}{C} \sum_{j \ne t_i}
              \max\!\big(0,\; \text{margin} - x_{i, t_i} + x_{i, j}\big)^p

    Samples whose true-class score already dominates all
    competitors by ``margin`` produce zero loss and zero gradient
    — like the binary SVM hinge, only the *support vectors*
    contribute to the update.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multi_margin_loss
    >>> scores = lucid.tensor([[2.0, 0.5, 0.1]])
    >>> target = lucid.tensor([0], dtype=lucid.int32)
    >>> multi_margin_loss(scores, target)
    Tensor(0.0)
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
    r"""Multi-label hinge loss for set-valued targets.

    The multi-label counterpart of :func:`multi_margin_loss`: each
    sample can belong to *several* classes (the "positives"), and
    the objective requires every positive class score to exceed
    every non-positive (negative) class score by at least 1.  Used
    for set prediction tasks such as image tagging where labels
    are not mutually exclusive.

    Targets are encoded as a fixed-width index list with ``-1`` as
    a padding sentinel — entries up to the first ``-1`` mark the
    positive classes for that sample.

    Parameters
    ----------
    x : Tensor
        Class scores of shape :math:`(N, C)` or :math:`(C,)`.
    target : Tensor
        Same shape as ``x``.  Non-negative entries are positive
        class indices; ``-1`` entries are ignored.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-sample tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-sample loss, summed over positive labels :math:`t` and
    non-positive labels :math:`j`:

    .. math::

        L_i = \frac{1}{C} \sum_{t \in P_i} \sum_{j \notin P_i}
              \max\!\big(0,\; 1 - x_{i, t} + x_{i, j}\big),

    where :math:`P_i` is the set of positive labels for sample
    :math:`i`.  Equivalently, it is the average of multi-class
    hinge losses obtained by treating each positive label as
    *the* correct one against the full set of non-positives.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multilabel_margin_loss
    >>> scores = lucid.tensor([[1.0, 0.5, -0.3, 0.2]])
    >>> target = lucid.tensor([[0, 1, -1, -1]], dtype=lucid.int32)
    >>> multilabel_margin_loss(scores, target)
    Tensor(0.85)
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
    r"""Logistic (softplus) loss for binary classification with ±1 labels.

    A "soft" variant of the binary hinge loss: instead of the
    piecewise-linear :math:`\max(0, 1 - y\,x)`, it uses the smooth
    surrogate :math:`\log(1 + e^{-y\,x})` — which is the
    negative-log-likelihood of a logistic model with labels in
    :math:`\{-1, +1\}`.  Equivalent to
    :func:`binary_cross_entropy_with_logits` with the ``{0, 1}``
    labels re-coded as :math:`\{-1, +1\}`.

    The implementation evaluates :math:`\mathrm{softplus}(-y\,x)`,
    which is numerically stable for large ``|x|`` (no overflow,
    no log of near-zero values).

    Parameters
    ----------
    input : Tensor
        Raw scores (logits), any shape.
    target : Tensor
        Target tensor of the same shape, conventionally holding
        :math:`\pm 1` (any real values are accepted).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = \log\!\big(1 + \exp(-y_i\,x_i)\big).

    Unlike the (non-smooth) hinge, every sample contributes a
    non-zero gradient — even correctly classified ones — but the
    contribution decays exponentially as :math:`y\,x` grows.  This
    softness improves optimisation behaviour with first-order
    methods at the cost of a slightly less sparse solution.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import soft_margin_loss
    >>> x = lucid.tensor([2.0, -1.0])
    >>> y = lucid.tensor([1.0, -1.0])
    >>> soft_margin_loss(x, y)
    Tensor(0.2284...)
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
    r"""Per-class logistic loss averaged over labels (multi-label BCE).

    The standard objective for multi-label classification with
    *independent* per-class probabilities: each class gets its own
    binary logistic regression head, and the total loss is the
    mean of the per-class binary cross-entropies.  Mathematically
    equivalent to applying
    :func:`binary_cross_entropy_with_logits` per class and
    averaging across the class axis.

    Computed via the numerically stable identity
    :math:`\log \sigma(x) = -\mathrm{softplus}(-x)`, which avoids
    overflow / underflow for large ``|x|``.

    Parameters
    ----------
    input : Tensor
        Raw logits of shape :math:`(N, C)`.
    target : Tensor
        Target probabilities (typically binary) of shape :math:`(N, C)`.
    weight : Tensor or None, optional
        Per-class weight broadcast against the per-class loss
        tensor before averaging.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-sample tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-sample loss, averaged across the :math:`C` classes:

    .. math::

        L_i = -\frac{1}{C} \sum_c \Big[
            t_{i,c}\,\log \sigma(x_{i,c})
            + (1 - t_{i,c})\,\log(1 - \sigma(x_{i,c}))
        \Big]

    Because the per-class predictions are independent (no softmax
    coupling), the gradient through each class is exactly that of
    a single binary logistic regression — convenient for highly
    multi-label problems where the active label set is sparse.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multilabel_soft_margin_loss
    >>> logits = lucid.tensor([[2.0, -1.0, 0.5]])
    >>> target = lucid.tensor([[1.0, 0.0, 1.0]])
    >>> multilabel_soft_margin_loss(logits, target)
    Tensor(0.3567...)
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
