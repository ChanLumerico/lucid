"""Registry factories for LeNet (LeCun et al., 1998).

The original paper specifies a single architecture (tanh activations +
average pooling).  Modern ReLU / max-pool reimplementations are not
paper-defined variants — get them via ``create_model("lenet_5",
activation="relu", pooling="max")`` instead.
"""

from dataclasses import replace
from typing import Any, cast

from lucid.models._registry import register_model
from lucid.models.vision.lenet._config import LeNetConfig
from lucid.models.vision.lenet._model import LeNet, LeNetForImageClassification

_CFG_5 = LeNetConfig()  # paper original (tanh + avg-pool)


# ── Backbone ──────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="lenet",
    model_type="lenet",
    model_class=LeNet,
    default_config=_CFG_5,
)
def lenet_5(pretrained: bool = False, **overrides: object) -> LeNet:
    r"""LeNet-5 feature-extracting backbone (no classification head).

    Builds a :class:`LeNet` with the paper-cited LeCun 1998 topology:
    three convolutions (C1: 1→6, C3: 6→16, C5: 16→120) interleaved with
    two :math:`2\times2` sub-sampling layers, using :math:`\tanh`
    activations and parameter-free average pooling.  Approximately
    44 k convolutional parameters (≈60 k including the F6 + output
    layers of the classifier variant).  Designed for :math:`1\times32
    \times32` grayscale digit inputs.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`LeNetConfig` to
        customise individual fields.  Use ``activation="relu"`` and
        ``pooling="max"`` for the modern ReLU + max-pool reimplementation,
        or ``in_channels=3`` for RGB inputs.

    Returns
    -------
    LeNet
        Backbone with the LeNet-5 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See LeCun et al., "Gradient-Based Learning Applied to Document
    Recognition", Proc. IEEE 86(11):2278–2324, 1998, Figure 2.  The
    canonical metric is MNIST test-set error: the original paper reports
    0.95% with the full distortion-augmented training recipe.  No
    paper-cited "tiny / large" LeNet variants exist (H11) — pass
    ``activation``/``pooling`` overrides to get the common modernised
    flavours.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.lenet import lenet_5
    >>> model = lenet_5()
    >>> x = lucid.randn(1, 1, 32, 32)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 120, 1, 1)
    (1, 120, 1, 1)
    """
    cfg = replace(_CFG_5, **cast(dict[str, Any], overrides)) if overrides else _CFG_5
    return LeNet(cfg)


# ── Classifier ────────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="lenet",
    model_type="lenet",
    model_class=LeNetForImageClassification,
    default_config=_CFG_5,
)
def lenet_5_cls(
    pretrained: bool = False, **overrides: object
) -> LeNetForImageClassification:
    r"""LeNet-5 image classifier (C1-S2-C3-S4-C5 + F6 + linear head).

    Builds a :class:`LeNetForImageClassification` with the paper-cited
    LeCun 1998 configuration: three convolutions, two sub-sampling
    layers, a hidden F6 layer (120 → 84), and a final linear projection
    to ``config.num_classes`` (default 10 for MNIST).  Approximately
    60 k parameters total.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`LeNetConfig` to
        retarget ``num_classes`` (e.g. ``num_classes=100`` for
        CIFAR-100) or to enable the modern ReLU + max-pool variant via
        ``activation="relu"``, ``pooling="max"``.

    Returns
    -------
    LeNetForImageClassification
        Classifier with the LeNet-5 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See LeCun et al., "Gradient-Based Learning Applied to Document
    Recognition", Proc. IEEE 86(11):2278–2324, 1998, Figure 2.  The
    canonical MNIST test-set error is 0.95% with the original distortion-
    augmented training recipe.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.lenet import lenet_5_cls
    >>> model = lenet_5_cls()
    >>> x = lucid.randn(2, 1, 32, 32)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    cfg = replace(_CFG_5, **cast(dict[str, Any], overrides)) if overrides else _CFG_5
    return LeNetForImageClassification(cfg)
