"""Cross-task shared utilities used by multiple model families.

These helpers are task-agnostic and may be imported by any model
regardless of its output task (classification, detection, segmentation …).
"""

from typing import TYPE_CHECKING, cast

import lucid
import lucid.nn as nn
import lucid.optim as optim
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def make_divisible(
    v: float,
    divisor: int = 8,
    min_value: int | None = None,
) -> int:
    """Round ``v`` to the nearest multiple of ``divisor``.

    The result is at least ``min_value`` (falls back to ``divisor`` when
    *None*).  The 0.9 × v guard prevents the value from being rounded
    down excessively — if the adjusted value is more than 10 % below ``v``
    an extra ``divisor`` is added.

    This is the canonical implementation used by MobileNet, EfficientNet,
    SE-ResNet, SK-ResNet, ResNeSt and any other family that needs channel
    counts aligned to a power-of-two-friendly grid.

    Parameters
    ----------
    v : float
        Raw (possibly non-integer) channel count to round.
    divisor : int, optional
        Alignment granularity.  Default ``8``.
    min_value : int, optional
        Hard lower bound on the result.  When ``None`` (default), falls
        back to ``divisor``.

    Returns
    -------
    int
        Channel count rounded to a multiple of ``divisor``, never below
        ``min_value`` / ``divisor``.

    Examples
    --------
    >>> from lucid.models._utils._common import make_divisible
    >>> make_divisible(33, divisor=8)
    32
    >>> make_divisible(35, divisor=8)
    32
    >>> make_divisible(3, divisor=8, min_value=8)
    8
    """
    min_val: int = min_value if min_value is not None else divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def transform_input_imagenet_to_tf(x: Tensor) -> Tensor:
    r"""Re-normalise ImageNet-standardised input into the ``(x-0.5)/0.5`` range.

    The Inception-v3 and GoogLeNet ImageNet checkpoints are ports of the
    original TensorFlow weights, which were trained on inputs scaled to
    :math:`[-1, 1]`.  Callers, however, feed ImageNet-standardised images —
    that is what the families' bundled preprocessing presets emit — so the
    channel statistics have to be converted inside the model:

    .. math::

        x'_c = x_c \cdot \frac{\sigma_c}{0.5} + \frac{\mu_c - 0.5}{0.5}

    with the usual ImageNet :math:`\mu = (0.485, 0.456, 0.406)` and
    :math:`\sigma = (0.229, 0.224, 0.225)`.  Skipping it leaves every channel
    off by a factor of ~2.2 in scale plus an offset, a large distribution
    shift for a BatchNorm network and a silent drop in top-1.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    chans = [
        x[:, c : c + 1] * (std[c] / 0.5) + ((mean[c] - 0.5) / 0.5) for c in range(3)
    ]
    return lucid.cat(chans, dim=1)


def init_cnn_fan_out(model: nn.Module, *, linear_std: float | None = None) -> None:
    r"""Reference CNN initialisation: He/MSRA fan-out convs, unit BN, small FC.

    ``kaiming_normal_(mode="fan_out", nonlinearity="relu")`` on every
    convolution, ones/zeros on every norm layer, and — when ``linear_std`` is
    given — ``N(0, linear_std²)`` with zero bias on every ``Linear`` (the
    MobileNet convention; leave it *None* to keep the framework default for
    fully-connected layers).

    Lucid's own ``Conv2d.reset_parameters`` falls back to
    ``kaiming_uniform(a=sqrt(5))``, which is a *different distribution with a
    different gain* — it is the generic framework default, not the one the
    ResNet/VGG/MobileNet line of papers trains with.  Starting from it changes
    the early-training trajectory of any from-scratch run.

    Parameters
    ----------
    model : nn.Module
        Module to initialise in place.  Every submodule is visited.
    linear_std : float or None, optional, keyword-only, default=None
        Standard deviation for ``Linear`` weights.  ``None`` leaves fully
        connected layers at the framework default; pass a value (0.01 in
        MobileNet) to draw them from ``N(0, linear_std^2)`` with zero bias.

    Returns
    -------
    None
        ``model`` is modified in place.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
        ):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear) and linear_std is not None:
            nn.init.normal_(m.weight, mean=0.0, std=linear_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def zero_init_last_bn(
    model: nn.Module, block_types: tuple[type, ...], attr: str = "bn3"
) -> None:
    r"""Zero the final BatchNorm gamma of every residual block.

    "Zero-initialize the last BN in each residual branch, so the block starts
    as an identity" (He et al., *Bag of Tricks*, and the reference ResNet's
    ``zero_init_residual``).  Improves early-training stability and is what the
    ``zero_init_residual`` config flag is supposed to switch on.

    Parameters
    ----------
    model : nn.Module
        Module to initialise in place.
    block_types : tuple of type
        Residual-block classes to match; a submodule is treated as a block
        when it is an instance of any of them.
    attr : str, optional, default="bn3"
        Attribute name of the block's final normalisation layer.  Blocks
        that do not carry it, or whose layer has no weight, are skipped.

    Returns
    -------
    None
        ``model`` is modified in place.
    """
    for m in model.modules():
        if isinstance(m, block_types):
            bn = getattr(m, attr, None)
            if bn is not None and getattr(bn, "weight", None) is not None:
                nn.init.zeros_(bn.weight)


def init_transformer_trunc_normal(
    model: nn.Module,
    std: float = 0.02,
    *,
    init_tokens: bool = True,
    zero_head: str | None = None,
) -> None:
    r"""Reference ViT-lineage initialisation: ``trunc_normal_(std)`` + zero bias.

    Every ``Linear`` (and ``Conv2d`` acting as a patch embedding) is drawn from
    a truncated normal at ``std``, biases are zeroed, and LayerNorms are left at
    unit weight / zero bias.  ViT, Swin, ConvNeXt, CvT, PVT, InceptionNeXt,
    EfficientFormer and MaxViT all specify this; without it they start from
    Lucid's generic ``kaiming_uniform(a=sqrt(5))``, whose scale is far larger
    and which the papers' LR schedules are not tuned for.

    Parameters
    ----------
    model : nn.Module
        Module to initialise in place.
    std : float, optional, default=0.02
        Standard deviation of the truncated normal.
    init_tokens : bool, optional, keyword-only, default=True
        Also draw bare ``Parameter`` tokens — class tokens, positional
        tables, distillation tokens — from the same distribution.  They are
        not owned by any layer, so the framework default never touches them
        and they would otherwise start at exact zeros.
    zero_head : str or None, optional, keyword-only, default=None
        Attribute name of a classification head to zero after
        initialisation, matching the papers' fine-tuning recipe.  ``None``
        leaves every head at its drawn value.

    Returns
    -------
    None
        ``model`` is modified in place.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    if init_tokens:
        # Class tokens and positional tables are bare Parameters, so the module
        # sweep above never reaches them.  Left at exact zeros (a common
        # oversight) a CLS token carries no signal to break symmetry with, and
        # every reference draws these from the same truncated normal.
        marks = (
            "cls_token",
            "class_token",
            "pos_embed",
            "position_embed",
            "dist_token",
        )
        for name, prm in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if any(mk in leaf for mk in marks):
                nn.init.trunc_normal_(prm, std=std)

    if zero_head is not None:
        # ViT and its descendants zero the classification head so training
        # starts from a uniform posterior.
        for name, prm in model.named_parameters():
            if name.startswith(zero_head):
                nn.init.zeros_(prm)


def reject_unavailable_pretrained(factory_name: str, *, alternative: str = "") -> None:
    """Refuse ``pretrained=True`` where no weights exist for that factory.

    ``PretrainedModel.from_pretrained`` resolves a registered name by calling
    ``factory(pretrained=True)`` and returns whatever comes back — it never
    checks that weights were actually loaded.  A factory that accepts the flag
    and ignores it therefore hands back a randomly initialised model that the
    caller believes is pretrained, which is a worse failure than refusing:
    it is silent, and it surfaces as mysteriously poor accuracy rather than
    as an error.

    Parameters
    ----------
    factory_name : str
        Name used in the message, so the caller knows which call to change.
    alternative : str, optional
        A sibling factory that *does* publish weights, mentioned in the
        message when given.
    """
    hint = (
        f"  Use ``{alternative}`` for the checkpointed variant." if alternative else ""
    )
    raise NotImplementedError(
        f"No pretrained weights are published for ``{factory_name}``; "
        f"``{factory_name}(pretrained=True)`` cannot be honoured.{hint}"
    )


def truncated_svd_linear(layer: nn.Linear, rank: int) -> nn.Sequential:
    r"""Factorise a trained ``Linear`` into two smaller ones (Fast R-CNN §3.1).

    A layer :math:`W \in \mathbb{R}^{u \times v}` is replaced by its rank-``t``
    truncated SVD, :math:`W \approx U_t \Sigma_t V_t^{\top}`, split across two
    layers: the first applies :math:`\Sigma_t V_t^{\top}` with no bias, the
    second applies :math:`U_t` and carries the original bias.  Parameter
    count goes from :math:`uv` to :math:`t(u + v)`, so any
    :math:`t < uv / (u + v)` is a saving.

    §3.1 uses this on ``fc6``/``fc7`` to cut detection time — for a detector
    that spends most of its forward pass in the per-RoI fully-connected
    layers, "compressing them with truncated SVD reduces detection time by
    more than 30% with only a small drop in mAP".

    This is a **post-training** transformation: it approximates weights that
    already exist and is not something to train through.  It also changes
    the parameter layout, so the result no longer accepts the original
    checkpoint.

    Args:
        layer: A trained :class:`~lucid.nn.Linear`.
        rank:  ``t``.  Must lie in ``[1, min(u, v)]``; the upper end
            reproduces the layer exactly and saves nothing.

    Returns:
        A :class:`~lucid.nn.Sequential` of two ``Linear`` layers computing
        the same function up to the truncation error.

    Raises:
        ValueError: If ``rank`` is outside the representable range, since a
            silently clamped rank would return a layer that is either
            unexpectedly exact or unexpectedly degenerate.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.models._utils._common import truncated_svd_linear
    >>> compressed = truncated_svd_linear(nn.Linear(64, 32), rank=8)
    >>> len(compressed)
    2
    """
    out_features = int(layer.weight.shape[0])
    in_features = int(layer.weight.shape[1])
    limit = min(out_features, in_features)
    if not 1 <= rank <= limit:
        raise ValueError(
            f"rank must lie in [1, {limit}] for a {out_features}x{in_features} "
            f"layer; got {rank}.  Above {limit} the factorisation has no more "
            "singular values to keep, and the compression would silently be "
            "an exact copy."
        )

    with lucid.no_grad():
        u, s, vh = lucid.linalg.svd(layer.weight)

    first = nn.Linear(in_features, rank, bias=False)
    second = nn.Linear(rank, out_features, bias=layer.bias is not None)
    with lucid.no_grad():
        # Sigma folds into the first factor, so the second is a plain
        # projection and the bias stays where it was.
        first.weight[:] = vh[:rank] * s[:rank].reshape(-1, 1)
        second.weight[:] = u[:, :rank]
        if layer.bias is not None and second.bias is not None:
            second.bias[:] = layer.bias

    return nn.Sequential(first, second)


def fit_linear_svm(
    features: Tensor,
    labels: Tensor,
    *,
    c: float = 1e-3,
    steps: int = 200,
    lr: float = 0.01,
) -> tuple[Tensor, Tensor]:
    r"""Fit one binary linear SVM by minimising hinge loss with L2.

    Solves

    .. math::

        \min_{w, b}\;\; \tfrac{1}{2}\lVert w \rVert^2
        + C \sum_i \max(0,\, 1 - y_i (w^{\top} x_i + b))

    by gradient descent.  R-CNN §2.3 trains one of these per class on
    features cached from a frozen network, taking positives from
    ground-truth boxes only and negatives from proposals below IoU 0.3.

    This is the optimisation, not the pipeline.  The paper's accuracy comes
    substantially from **hard negative mining** — repeatedly rescoring the
    negative pool and refitting on the highest-scoring mistakes — which
    needs the dataset, not just one feature matrix.  A caller reproducing
    R-CNN drives that loop and calls this at each round.

    Args:
        features: ``(N, D)`` feature matrix, one row per example.
        labels:   ``(N,)`` with ``+1`` for positives and ``-1`` for
            negatives.  Zeros are rejected rather than silently treated as
            negatives.
        c:        Penalty on the hinge term, the ``C`` above.
        steps:    Gradient steps.
        lr:       Learning rate.

    Returns:
        ``(w, b)`` — ``w`` of shape ``(D,)`` and ``b`` a scalar tensor.
        Score a new feature with ``x @ w + b``; positive means the class.

    Raises:
        ValueError: If ``labels`` holds anything other than +1 / -1, or if
            it contains only one sign — a single-class fit has no margin to
            maximise and would return an arbitrary separator.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._common import fit_linear_svm
    >>> x = lucid.tensor([[2.0, 0.0], [-2.0, 0.0]])
    >>> w, b = fit_linear_svm(x, lucid.tensor([1.0, -1.0]))
    >>> bool((x @ w + b).tolist()[0] > 0)
    True
    """
    values = cast(list[float], labels.reshape(-1).tolist())
    if any(v not in (1.0, -1.0) for v in values):
        raise ValueError(
            "fit_linear_svm expects labels of exactly +1 and -1; got values "
            f"{sorted(set(values))[:5]}.  Zero-or-one labels are a common "
            "mix-up and would train against the wrong margin."
        )
    if len(set(values)) < 2:
        raise ValueError(
            "fit_linear_svm needs both positive and negative examples; the "
            f"labels are all {values[0]:+.0f}.  With one class there is no "
            "margin to maximise and the returned separator would be arbitrary."
        )

    d = int(features.shape[1])
    dev = features.device.type
    # Kept 2-D internally because ``matmul`` requires it; the caller gets the
    # flat ``(D,)`` vector they would write ``x @ w`` with.
    w = nn.Parameter(lucid.zeros(d, 1, device=dev))
    b = nn.Parameter(lucid.zeros(1, device=dev))
    opt = optim.SGD([w, b], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        scores = (features @ w).reshape(-1) + b
        hinge = F.relu(1.0 - labels * scores).sum()
        loss = 0.5 * (w * w).sum() + c * hinge
        loss.backward()
        opt.step()

    return w.detach().reshape(-1), b.detach().reshape(())
