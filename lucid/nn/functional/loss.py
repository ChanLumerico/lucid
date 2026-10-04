"""
nn.functional loss functions.
"""

import math
from typing import TYPE_CHECKING, Sequence, cast

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._types import Reduction, ReductionKL

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


def _accumulation_dtype(t: Tensor) -> _lucid.dtype:
    """The dtype a loss over ``t`` is computed and summed in.

    Half precision is widened to float32.  A float16 sum overflows at
    65504, so a mean over a large batch came out ``inf / inf = nan`` even
    though every term and the answer were small; and constants a loss
    needs (an ``eps`` of 1e-12, a gradient floor of 1e12) are not float16
    numbers at all.  The result is rounded back to the input's dtype.
    """
    if t.dtype in (_lucid.float16, _lucid.bfloat16):
        return _lucid.float32
    return t.dtype


def _reduce_in(t: Tensor, reduction: str, out_dtype: _lucid.dtype) -> Tensor:
    """Reduce per-element losses held in their accumulation dtype, then
    round the result back to ``out_dtype``."""
    _validate_reduction(reduction)
    if reduction == "mean":
        t = t.mean()
    elif reduction == "sum":
        t = t.sum()
    return t if t.dtype == out_dtype else t.to(dtype=out_dtype)


def _host_any(mask: Tensor) -> bool:
    """Whether any element of a CPU ``mask`` is set.

    For a guard on a CPU tensor, which is on the host already, so reading
    it costs no sync.  The read runs outside any active compile trace,
    like the embedding table's range check: it raises or passes, and a
    host read inside a trace would mark the trace unsupported.
    """
    tracer = _C_engine.compile.current_tracer()
    _C_engine.compile.set_current_tracer(None)
    try:
        return bool(mask.any().item())
    finally:
        _C_engine.compile.set_current_tracer(tracer)


def mse_loss(x: Tensor, target: Tensor, reduction: Reduction = "mean") -> Tensor:
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
    tensor(0.25)
    """
    _validate_reduction(reduction)
    red: int = _REDUCTION_MAP[reduction]
    return _wrap(_C_engine.nn.mse_loss(_unwrap(x), _unwrap(target), red))


def l1_loss(x: Tensor, target: Tensor, reduction: Reduction = "mean") -> Tensor:
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
    tensor(0.5)
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
    x: Tensor, target: Tensor, beta: float = 1.0, reduction: Reduction = "mean"
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
    tensor(1.312)
    """
    # ``huber_loss`` alone is the wrong function: Huber's quadratic region
    # is ``0.5 x²`` where smooth L1's is ``0.5 x² / beta``, and its linear
    # tail is ``beta(|x| - 0.5 beta)`` where smooth L1's is ``|x| - 0.5
    # beta``.  The two differ by a factor of ``beta`` everywhere — which is
    # 1 at the default, so the default was right and every other ``beta``
    # was scaled.  The ``1/beta`` this restores is the one the docstring
    # above has always described.
    if beta == 0.0:
        # The degenerate limit is plain L1, as the reference also answers.
        return l1_loss(x, target, reduction=reduction)
    return huber_loss(x, target, delta=beta, reduction=reduction) / beta


def huber_loss(
    x: Tensor, target: Tensor, delta: float = 1.0, reduction: Reduction = "mean"
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
    tensor(2.312)
    """
    _validate_reduction(reduction)
    red: int = _REDUCTION_MAP[reduction]
    return _wrap(_C_engine.nn.huber_loss(_unwrap(x), _unwrap(target), delta, red))


def _refuse_or_poison(
    target: Tensor,
    safe: Tensor,
    counted: Tensor | None,
    scale: Tensor | None,
    dtype: _lucid.dtype,
) -> Tensor | None:
    """Deal with class indices outside the class range.

    ``safe`` is ``target`` clamped to the class range, so the two differ
    exactly where an index was out of range; ``counted`` is ``False`` at
    the ``ignore_index`` positions, which are allowed to be.

    A CPU target is on the host already, so it is checked for free and an
    ``IndexError`` raised, as the reference does; ``scale`` comes back as
    it was.  A Metal target is not read back — a host read in every loss
    call stalls every training step — so ``scale`` (a per-sample factor,
    ``None`` for all ones) comes back NaN at each such position instead,
    and a bad label poisons the loss rather than being scored as class 0
    or ``C - 1``; ``dtype`` is the factor's dtype when ``scale`` is
    ``None``.  The CPU read runs outside any active compile trace, like the
    embedding table's range check: it is a guard, not a value.
    """
    usable: Tensor = safe == target
    if counted is not None:
        usable = usable | ~counted
    if target.device == "cpu":
        tracer = _C_engine.compile.current_tracer()
        _C_engine.compile.set_current_tracer(None)
        try:
            first: int | None = (
                None
                if bool(usable.all().item())
                else int(target[~usable].reshape(-1)[0].item())
            )
        finally:
            _C_engine.compile.set_current_tracer(tracer)
        if first is not None:
            raise IndexError(f"Target {first} is out of bounds.")
        return scale
    ones: Tensor = scale if scale is not None else _lucid.ones_like(safe, dtype=dtype)
    return _lucid.where(usable, ones, _lucid.full_like(ones, math.nan))


def _class_nll(
    log_p: Tensor,
    target: Tensor,
    weight: Tensor | None,
    ignore_index: int | None,
    reduction: str,
    label_smoothing: float,
    op: str,
) -> Tensor:
    """Negative log-likelihood of integer class targets — the shared body of
    :func:`cross_entropy` and :func:`nll_loss`.

    ``log_p`` holds log-probabilities of shape ``(N, C, *)`` and ``target``
    class indices of shape ``(N, *)``.  Each sample is scaled by the weight
    of its class, ``ignore_index`` samples are dropped, and ``"mean"``
    divides by the total weight of the samples kept.
    """
    num_classes: int = int(log_p.shape[1])
    tgt: Tensor = target.to(dtype=_lucid.int32)

    # Clamped before every gather, not masked after it.
    #
    # ``ignore_index`` defaults to -100, and those sentinels used to be
    # handed straight to ``gather``, which read that far outside the
    # logits and returned whatever was in memory; on the CPU, once
    # ``gather`` learned to bounds-check, the same call raised.  The class
    # weight was gathered with the raw index too, and with a rank-matched
    # ``gather`` that refused a K-d target outright.
    #
    # The value read at a clamped position is discarded by the mask below,
    # so which valid index is used does not matter; that it is valid does.
    safe: Tensor = _lucid.clip(tgt, 0, num_classes - 1)
    nll: Tensor = -_lucid.gather(log_p, 1, safe.unsqueeze(1)).squeeze(1)

    # ``keep`` is 1 for a sample that counts and 0 for an ignored one — and
    # NaN for an out-of-range target on Metal, so a bad label poisons the
    # loss rather than being scored as class 0 or C - 1.
    keep: Tensor | None = None
    counted: Tensor | None = None
    if ignore_index is not None:
        counted = tgt != ignore_index
        keep = counted.to(dtype=log_p.dtype)
    keep = _refuse_or_poison(tgt, safe, counted, keep, log_p.dtype)

    # Each sample's share of the mean: its class weight times ``keep``.
    # ``None`` when every sample counts once.
    sample_weight: Tensor | None = None
    if weight is not None:
        sample_weight = _lucid.index_select(weight, 0, safe.reshape(-1)).reshape(
            list(tgt.shape)
        )
    if keep is not None:
        sample_weight = keep if sample_weight is None else sample_weight * keep
    if sample_weight is not None:
        nll = nll * sample_weight

    if label_smoothing > 0.0:
        # Smoothing term — uniform distribution NLL = -mean over classes
        # of log_softmax, which is -sum/C.  When weight is set, the uniform
        # term is weighted by (mean class weight) following the reference
        # framework's behaviour.
        smooth: Tensor = -log_p.mean(dim=1)  # (N, ...)
        if weight is not None:
            # Weighted-uniform: −Σ_c w_c · log_softmax / C
            log_p_weighted: Tensor = log_p * weight.reshape(
                [1, num_classes] + [1] * (log_p.ndim - 2)
            )
            smooth = -log_p_weighted.sum(dim=1) / num_classes
        if keep is not None:
            smooth = smooth * keep
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth

    if reduction == "none":
        return nll
    # Half precision is summed in float32.  A float16 count of kept samples
    # is inf past 65504, and so is the summed loss, so a mean over a long
    # sequence batch came out inf / inf = NaN though every term was small.
    out_dtype = nll.dtype
    acc = _accumulation_dtype(nll)
    if acc != out_dtype:
        nll = nll.to(dtype=acc)
        if sample_weight is not None:
            sample_weight = sample_weight.to(dtype=acc)
    reduced: Tensor
    if reduction == "sum":
        reduced = nll.sum()
    elif sample_weight is None:
        reduced = nll.mean()
    else:
        reduced = _weighted_mean(nll, sample_weight)
    return reduced if acc == out_dtype else reduced.to(dtype=out_dtype)


def _weighted_mean(nll: Tensor, sample_weight: Tensor) -> Tensor:
    """``nll.sum() / sample_weight.sum()`` — the mean of :func:`_class_nll`,
    whose divisor is the total weight of the samples kept."""
    total: Tensor = nll.sum()
    denom: Tensor = sample_weight.sum()
    # A batch with nothing kept — all padding, as one micro-batch of an
    # MLM or seq2seq run can be — has the value 0 / 0 = NaN, as in the
    # reference.  Its gradient is 0, also as in the reference: dividing by
    # the zero itself sent inf to every masked position, inf * 0 = NaN
    # reached every parameter, and one such batch under gradient
    # accumulation ruined the whole step.
    empty: Tensor = denom == 0.0
    safe_denom: Tensor = _lucid.where(empty, _lucid.ones_like(denom), denom)
    return _lucid.where(empty, _lucid.full_like(total, math.nan), total / safe_denom)


def cross_entropy(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
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
        shape matching ``x``.  An index outside :math:`[0, C)` that is
        not ``ignore_index`` raises ``IndexError`` for a CPU tensor; on
        Metal, where reading the target back would stall every step, it
        makes the loss NaN instead.
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
    tensor(0.3597)
    """
    _validate_reduction(reduction)
    from lucid.nn.functional.activations import log_softmax as _log_softmax

    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing!r}")

    # Class dim is 1 for both (N, C) and (N, C, *) inputs.
    log_p: Tensor = _log_softmax(x, dim=1)
    return _class_nll(
        log_p, target, weight, ignore_index, reduction, label_smoothing, "cross_entropy"
    )


def nll_loss(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
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
        :math:`(N, d_1, \dots, d_k)`.  An index outside :math:`[0, C)`
        that is not ``ignore_index`` raises ``IndexError`` for a CPU
        tensor and makes the loss NaN on Metal.
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
    tensor(0.3597)
    """
    _validate_reduction(reduction)
    return _class_nll(x, target, weight, ignore_index, reduction, 0.0, "nll_loss")


def binary_cross_entropy(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Binary cross-entropy between predicted probabilities and targets.

    The standard objective for binary classification and for
    multi-label classification with independent class predictions.
    Operates on *probabilities* in :math:`(0, 1)` — typically the
    output of a :func:`~lucid.nn.functional.sigmoid`.  When working
    with raw logits, prefer :func:`binary_cross_entropy_with_logits`
    for numerical stability.

    Each logarithm is clamped below at :math:`-100`, so a probability
    of exactly ``0`` or ``1`` costs at most ``100`` instead of an
    infinite or undefined loss, and the input gradient's denominator
    :math:`p(1-p)` is floored at :math:`\varepsilon = 10^{-12}`, so the
    gradient stays finite at the boundaries too.

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

        L_i = -\big(y_i\,\max(\log p_i, -100)
              + (1 - y_i)\,\max(\log(1 - p_i), -100)\big)

    Gradient w.r.t. ``x`` is
    :math:`(p_i - y_i) / \max(p_i(1 - p_i), \varepsilon)`, which grows
    without bound as :math:`p_i \to 0` or :math:`1` until the floor
    caps it — the reason the logits-form (which yields a clean
    :math:`\sigma(x) - y` gradient) is preferred when training
    stability is critical.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import binary_cross_entropy
    >>> p = lucid.tensor([0.9, 0.2, 0.7])
    >>> y = lucid.tensor([1.0, 0.0, 1.0])
    >>> binary_cross_entropy(p, y)
    tensor(0.2284)
    """
    _validate_reduction(reduction)
    # The probability used to be clamped to [1e-12, 1 - 1e-12], and in
    # float32 ``1 - 1e-12`` rounds to 1.0: a sigmoid of a logit above ~17
    # is exactly 1, ``log(1 - 1)`` is -inf, and ``0 * -inf`` made the loss
    # NaN.  The clamp also zeroed the gradient at both ends.
    #
    # The definition now is the reference's: each log term clamped at
    # -100 in the value, and an input gradient of
    # ``(p - y) / max(p (1 - p), eps)``.  Where ``p (1 - p) >= eps`` the
    # plain expression is differentiated as written (which also keeps the
    # second derivative); elsewhere — within ~1e-12 of 0 or 1, where the
    # logs are clamped or infinite — the value is computed off the graph
    # and the floored gradient is attached as a straight-through term.
    #
    # Half precision is computed in float32 and rounded back: ``eps`` is 0
    # in float16, and the floored gradient (1e12) is not a float16 number.
    out_dtype = x.dtype
    acc = _accumulation_dtype(x)
    p: Tensor = x.to(dtype=acc)
    y: Tensor = target.to(dtype=acc)
    eps: float = 1e-12
    p_d: Tensor = p.detach()
    inside: Tensor = p_d * (1.0 - p_d) >= eps
    p_in: Tensor = _lucid.where(inside, p, _lucid.full_like(p_d, 0.5))
    interior: Tensor = -(y * p_in.log() + (1.0 - y) * (1.0 - p_in).log())
    log_p: Tensor = p_d.log().clamp(min=-100.0)
    log_q: Tensor = (1.0 - p_d).log().clamp(min=-100.0)
    grad_edge: Tensor = (p_d - y.detach()) / eps
    edge: Tensor = -(y * log_p + (1.0 - y) * log_q) + (p - p_d) * grad_edge
    bce: Tensor = _lucid.where(inside, interior, edge)
    if weight is not None:
        bce = bce * weight
    return _reduce_in(bce, reduction, out_dtype)


def binary_cross_entropy_with_logits(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
    reduction: Reduction = "mean",
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
    The numerically stable form is

    .. math::

        L_i = (1 - y_i)\,x_i + \operatorname{softplus}(-x_i),

    equivalent to :math:`-(y\log\sigma(x) + (1-y)\log(1-\sigma(x)))`
    but free of overflow, and smooth at :math:`x = 0` (the
    :math:`\max(x, 0) + \log(1 + e^{-|x|})` spelling has the same value
    but a kink in each piece there).  With ``pos_weight``:

    .. math::

        L_i = (1 - y_i)\,x_i
              + \big(1 + (w^{+} - 1) y_i\big)\operatorname{softplus}(-x_i).

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
    tensor(0.3048)
    """
    _validate_reduction(reduction)
    # ``max(x, 0) - x y + log(1 + exp(-|x|))`` has the right value and the
    # wrong derivative at x = 0: the subgradients there are clamp' = 1 and
    # sign(0) = 0, so d/dx came out 1 - y instead of 1/2 - y, and a
    # zero-initialised head got no gradient at all from its positive
    # labels.  ``(1 - y) x + softplus(-x)`` is the same function written
    # with one smooth piece, and softplus is stable for any |x|.
    out_dtype = x.dtype
    acc = _accumulation_dtype(x)
    xa: Tensor = x.to(dtype=acc)
    y: Tensor = target.to(dtype=acc)
    softplus_neg: Tensor = _wrap(_C_engine.softplus(_unwrap(-xa)))
    if pos_weight is None:
        loss: Tensor = (1.0 - y) * xa + softplus_neg
    else:
        loss = (1.0 - y) * xa + (1.0 + (pos_weight - 1.0) * y) * softplus_neg
    if weight is not None:
        loss = loss * weight
    return _reduce_in(loss, reduction, out_dtype)


def kl_div(
    x: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduction: ReductionKL = "mean",
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

    with :math:`0 \log 0 = 0`, so a target with zero entries (one-hot,
    sparse) contributes nothing there rather than ``nan``.  Globally:

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
    tensor(0.02391)
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
        # ``target * (log(target) - x)`` is ``0 * -inf = nan`` wherever the
        # target is 0 — every off-class entry of a one-hot or sparse
        # distillation target.  ``xlogy`` takes ``0 log 0`` as 0, the limit
        # the divergence is defined with.
        kl = _C_engine.sub(
            _unwrap(_lucid.xlogy(target, target)), _C_engine.mul(ti, xi)
        )
    if reduction == "mean":
        return _wrap(_C_engine.mean(kl, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(kl, [], False))
    if reduction == "batchmean":
        total: _C_engine.TensorImpl = _C_engine.sum(kl, [], False)
        batch_size: int = int(x.shape[0])
        return _wrap(total) / batch_size
    return _wrap(kl)


def _apply_reduction(t: _C_engine.TensorImpl, reduction: Reduction) -> Tensor:
    """Apply reduction to a batch of per-sample losses."""
    # Nine losses reduce here, and an unknown string used to fall through
    # to "none": ``reduction="avg"`` returned the unreduced tensor.
    _validate_reduction(reduction)
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
    reduction: Reduction = "mean",
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
    tensor(0.)
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
    reduction: Reduction = "mean",
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
    tensor(0.)
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
    reduction: Reduction = "mean",
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
    tensor(0.2929)
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
    reduction: Reduction = "mean",
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
    tensor(0.75)
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
    reduction: Reduction = "mean",
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
    tensor(0.2)
    """
    xi = _unwrap(x)
    yi = _unwrap(y)
    _C_engine.full(xi.shape, 1.0, xi.dtype, xi.device)
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
    reduction: Reduction = "mean",
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
        in the loss, where ``target > 1`` (at 0 and 1 the true
        :math:`\log(y!)` is 0).  Has no effect on gradients (constant in
        ``x``) but yields the correct log-likelihood value.
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
    tensor(-0.2976)
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

    if full:
        # ``full`` was accepted and did nothing: passing it changed the
        # answer by exactly zero.  The term it names is Stirling's
        # approximation to the ``log(target!)`` that the non-full form
        # drops as constant in the parameters —
        #
        #     target·log(target) − target + ½·log(2π·target)
        #
        # applied only where ``target > 1``, since at 0 and 1 the true
        # ``log(target!)`` is 0 and the approximation is not.
        one = _C_engine.full(ti.shape, 1.0, ti.dtype, ti.device)
        safe = _C_engine.maximum(ti, one)  # keeps log finite at target == 0
        stirling = _C_engine.add(
            _C_engine.sub(_C_engine.mul(safe, _C_engine.log(safe)), safe),
            _C_engine.mul(
                _C_engine.full(ti.shape, 0.5, ti.dtype, ti.device),
                _C_engine.log(
                    _C_engine.mul(
                        _C_engine.full(ti.shape, 2.0 * math.pi, ti.dtype, ti.device),
                        safe,
                    )
                ),
            ),
        )
        zero = _C_engine.zeros(ti.shape, ti.dtype, ti.device)
        loss = _C_engine.add(
            loss, _C_engine.where(_C_engine.greater(ti, one), stirling, zero)
        )

    return _apply_reduction(loss, reduction)


def gaussian_nll_loss(
    x: Tensor,
    target: Tensor,
    var: Tensor | float,
    full: bool = False,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
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
    initially predicts near-zero variance.  The clamp acts on the value
    only: the gradient reaches ``var`` as if it had not been clamped, so
    a variance head stuck below ``eps`` is still pushed back up.

    Parameters
    ----------
    x : Tensor
        Predicted means :math:`\mu`, any shape.
    target : Tensor
        Observed values :math:`y`, broadcast-compatible with ``x``.
    var : Tensor or float
        Predicted variances :math:`\sigma^2 \ge 0`.  Either the shape of
        ``x``; or that shape without its last dimension — one variance
        per sample, e.g. ``(N,)`` for an ``(N, D)`` input — which is
        unsqueezed to broadcast over that dimension; or the shape of
        ``x`` with exactly one dimension of size 1.  A float is one
        variance for every element.  Any other shape raises
        ``ValueError``.  A negative variance raises ``ValueError`` for a
        CPU tensor; on Metal, where reading it back would stall the
        step, it makes the loss NaN there instead.
    full : bool, optional
        Include the constant :math:`\tfrac{1}{2}\log(2\pi)` term in
        the loss value.  Has no effect on gradients; it makes the value
        an actual negative log-likelihood rather than one shifted by a
        constant.
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
    tensor(-0.2841)
    """
    if isinstance(var, (int, float)):
        if var < 0:
            raise ValueError("var has negative entry/entries")
        var = _lucid.full_like(x, float(var))
    # The reference's shape rules.  A per-sample variance ``(N,)`` against
    # an ``(N, D)`` input used to broadcast along the *last* axis — one
    # variance per feature — and ``(N,)`` against ``(N, D != N)`` failed.
    if tuple(var.shape) != tuple(x.shape):
        if tuple(x.shape[:-1]) == tuple(var.shape):
            var = var.unsqueeze(-1)
        elif x.ndim == var.ndim and (
            sum(v for d, v in zip(x.shape, var.shape) if d != v) == 1
        ):
            pass
        else:
            raise ValueError("var is of incorrect size")
    _validate_reduction(reduction)

    negative: Tensor = var < 0.0
    if var.device == "cpu" and _host_any(negative):
        raise ValueError("var has negative entry/entries")

    # Clamped at ``eps`` in value, not in gradient: the reference clamps a
    # copy under ``no_grad``, so the gradient flows to ``var`` unchanged.
    # ``maximum(var, eps)`` gave every variance below ``eps`` a zero
    # gradient, and a variance head that collapsed there stayed there.
    var_d: Tensor = var.detach()
    var_c: Tensor = _lucid.where(var_d >= eps, var, (var - var_d) + eps)
    loss: Tensor = 0.5 * (var_c.log() + (x - target) ** 2 / var_c)
    if full:
        # The omitted constant of the Gaussian log-density, 0.5 * log(2 pi)
        # per element.  It does not change the gradient, but it is what makes
        # the returned number an actual negative log-likelihood rather than
        # one shifted by a constant — which matters the moment the value is
        # compared against another model's or reported as a likelihood.
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    if var.device != "cpu":
        loss = _lucid.where(negative, _lucid.full_like(loss, math.nan), loss)
    return _apply_reduction(_unwrap(loss), reduction)


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor | Sequence[int],
    target_lengths: Tensor | Sequence[int],
    blank: int = 0,
    reduction: Reduction = "mean",
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
        ``"mean"``, each sample's loss is divided by its target
        length (at least 1) and the results are averaged across the
        batch, as the reference framework defines it.
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
    that collapses repeats and removes blanks.  The forward and
    backward recursions run in the log domain on the CPU; on metal
    the inputs make a round trip.  The gradient with respect to
    ``log_probs`` follows the reference framework's formula, which is
    the true gradient with respect to the logits once a
    :func:`~lucid.nn.functional.log_softmax` sits in front — the way
    the loss is meant to be fed.  It is differentiable once.

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
    _validate_reduction(reduction)

    def _as_lengths(v: Tensor | Sequence[int]) -> Tensor:
        if isinstance(v, _lucid.Tensor):
            return v
        return _lucid.tensor(list(v), dtype=_lucid.int64)

    input_lengths = _as_lengths(input_lengths)
    target_lengths = _as_lengths(target_lengths)

    tgt_impl = _unwrap(targets)
    if len(list(tgt_impl.shape)) > 1:
        # Padded (N, S): row b's first target_lengths[b] entries are its
        # target.  The kernel reads one concatenated list, so the rows are
        # gathered into one — flattening the padding in with them made every
        # sample after a short one read the padding as its target.
        width = int(tgt_impl.shape[1])
        rows = [int(n) for n in cast(list[int], target_lengths.tolist())]
        keep = [b * width + i for b, n in enumerate(rows) for i in range(n)]
        flat = _wrap(_C_engine.reshape(tgt_impl, [-1]))
        tgt_impl = _unwrap(flat[_lucid.tensor(keep, dtype=_lucid.int64)])

    # Ensure integer dtype for lengths and targets.
    def _to_i32(impl: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
        if getattr(impl, "dtype", None) != _C_engine.I32:
            return _C_engine.astype(impl, _C_engine.I32)
        return impl

    tgt_impl = _to_i32(tgt_impl)
    il_impl = _to_i32(_unwrap(input_lengths))
    tl_impl = _to_i32(_unwrap(target_lengths))
    # Targets and lengths usually live on the CPU whatever the device of
    # log_probs; the kernel reads all four from one device, and a metal
    # log_probs beside CPU integers failed with a bad_variant_access.
    device = log_probs.device
    tgt_impl, il_impl, tl_impl = (
        _unwrap(_wrap(t).to(device)) for t in (tgt_impl, il_impl, tl_impl)
    )

    loss_t = _C_engine.nn.ctc_loss(
        _unwrap(log_probs), tgt_impl, il_impl, tl_impl, blank, zero_infinity
    )
    if reduction == "mean":
        per_sample = _wrap(loss_t)
        lengths = target_lengths.to(per_sample.dtype).to(per_sample.device)
        return (per_sample / lengths.clamp(min=1.0)).mean()
    return _apply_reduction(loss_t, reduction)


def multi_margin_loss(
    x: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Tensor | None = None,
    reduction: Reduction = "mean",
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
        Integer class indices of shape :math:`(N,)`.  An index outside
        :math:`[0, C)` raises ``IndexError`` for a CPU tensor and makes
        the loss NaN on Metal.
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
    tensor(0.)
    """
    num_classes: int = int(x.shape[1])
    tgt: Tensor = target.to(dtype=_lucid.int32).reshape(-1)
    safe: Tensor = _lucid.clip(tgt, 0, num_classes - 1)
    poison: Tensor | None = _refuse_or_poison(tgt, safe, None, None, x.dtype)
    tgt_col: Tensor = safe.reshape([-1, 1])  # (N, 1)

    # margin - x[i, y_i] + x[i, j] for every j, hinged and raised to p.
    correct: Tensor = _lucid.gather(x, 1, tgt_col)  # (N, 1)
    hinge: Tensor = (margin - correct + x).relu()
    if p > 1:
        hinge = hinge ** float(p)

    if weight is not None:
        # The weight of each sample's true class.  This was a gather of a
        # (1, C) weight with an (N, 1) index, which the CPU refused for any
        # N > 1 ("index is larger than the operand on a non-gathered axis").
        hinge = hinge * _lucid.index_select(weight, 0, safe).reshape([-1, 1])

    # The true class is not one of its own competitors.
    classes: Tensor = _lucid.arange(num_classes, device=x.device).reshape([1, -1])
    is_target: Tensor = classes == tgt_col.to(dtype=classes.dtype)
    hinge = _lucid.where(is_target, _lucid.zeros_like(hinge), hinge)

    loss_n: Tensor = hinge.sum(dim=1) / num_classes  # (N,)
    if poison is not None:
        loss_n = loss_n * poison
    return _apply_reduction(_unwrap(loss_n), reduction)


def multilabel_margin_loss(
    x: Tensor,
    target: Tensor,
    reduction: Reduction = "mean",
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
    positive classes for that sample; that entry and every one after it
    are ignored, whatever they hold.

    Parameters
    ----------
    x : Tensor
        Class scores of shape :math:`(N, C)` or :math:`(C,)`.
    target : Tensor
        Same shape as ``x``, any integer dtype.  The entries before the
        first negative one are the positive class indices.  A listed
        class outside :math:`[0, C)` raises ``IndexError`` for a CPU
        tensor and makes that sample's loss NaN on Metal.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar, or per-sample tensor of shape :math:`(N,)` (a 0-d tensor
        for a 1-D ``x``) under ``"none"``.

    Notes
    -----
    Per-sample loss, summed over positive labels :math:`t` and
    non-positive labels :math:`j`:

    .. math::

        L_i = \frac{1}{C} \sum_{t \in P_i} \sum_{j \notin P_i}
              \max\!\big(0,\; 1 - x_{i, t} + x_{i, j}\big),

    where :math:`P_i` is the list of positive labels for sample
    :math:`i` (a class listed twice is counted twice).  Equivalently,
    it is the average of multi-class hinge losses obtained by treating
    each positive label as *the* correct one against the full set of
    non-positives.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multilabel_margin_loss
    >>> scores = lucid.tensor([[1.0, 0.5, -0.3, 0.2]])
    >>> target = lucid.tensor([[0, 1, -1, -1]])
    >>> multilabel_margin_loss(scores, target)
    tensor(0.275)
    """
    unbatched: bool = x.ndim == 1
    xb: Tensor = x.reshape([1, -1]) if unbatched else x
    # Any integer dtype is taken: a ``lucid.tensor([...ints...])`` target
    # is int64, and an explicitly int32 one is just as usual.
    tgt: Tensor = (target.reshape([1, -1]) if unbatched else target).to(
        dtype=_lucid.int32
    )
    num_classes: int = int(xb.shape[1])

    # The labels of a sample are its entries up to the first negative one.
    # Every column used to be read with ``index >= 0``, so a label after
    # the first -1 still counted as a positive.
    listed: Tensor = _lucid.cumprod((tgt >= 0).to(dtype=_lucid.int32), dim=1) == 1
    safe: Tensor = _lucid.clip(tgt, 0, num_classes - 1)
    # A listed class outside [0, C) raises for a CPU target and makes the
    # sample's loss NaN on Metal (its count is NaN), as for cross_entropy.
    src: Tensor = listed.to(dtype=xb.dtype)
    poisoned: Tensor | None = _refuse_or_poison(tgt, safe, listed, src, xb.dtype)
    if poisoned is not None:
        src = poisoned

    # How many times each class is listed.  A class listed twice counts
    # twice as a positive, as in the reference; it is a target class (not
    # one of the negatives) once listed at all.
    counts: Tensor = _lucid.scatter_add(
        _lucid.zeros_like(xb), 1, safe.to(dtype=_lucid.int64), src
    )
    negative: Tensor = (counts == 0.0).to(dtype=xb.dtype)

    # hinge[i, t, j] = max(0, 1 - x[i, t] + x[i, j]), for each positive t
    # (weighted by its count) against each negative j.
    hinge: Tensor = (1.0 - xb.unsqueeze(2) + xb.unsqueeze(1)).relu()
    pairs: Tensor = hinge * counts.unsqueeze(2) * negative.unsqueeze(1)
    loss_n: Tensor = pairs.sum(dim=[1, 2]) / num_classes  # (N,)
    if unbatched:
        loss_n = loss_n.reshape([])
    return _apply_reduction(_unwrap(loss_n), reduction)


# ── P3 fills: soft_margin_loss / multilabel_soft_margin_loss ───────────────


def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    reduction: Reduction = "mean",
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
    tensor(0.2201)
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
    reduction: Reduction = "mean",
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
    tensor(0.3048)
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
