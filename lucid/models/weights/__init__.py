"""``lucid.models.weights`` — per-family pretrained-weight enums.

Discovery namespace for every model family's ``<Variant>Weights`` enum.
Each enum is *declared* inside its model package
(``lucid/models/vision/<family>/_weights.py``) so the declaration sits
right next to the architecture it targets; this module then re-exports
all of them under a single short import path so users don't have to
remember which sub-package each family lives in::

    from lucid.models.weights import AlexNetWeights, ResNet18Weights
    from lucid.models import alexnet_cls
    m = alexnet_cls(weights=AlexNetWeights.IMAGENET1K_V1)

Deliberately separate from :mod:`lucid.weights` — that package hosts the
*infrastructure* (``WeightEntry``, ``WeightsEnum``, ``HUB_BASE``,
``register_weights``, hub download, loading); this package only re-exports
*concrete* enums.  And deliberately separate from the top-level
:mod:`lucid.models` namespace — that namespace already carries every
model class + factory, so piling on tens of additional ``*Weights``
symbols would bury discovery.  All ``<Variant>Weights`` enums are
reachable here and *only* here:

* ✅ ``from lucid.models.weights import AlexNetWeights``
* ❌ ``from lucid.models import AlexNetWeights`` (not exported)

The dependency direction is one-way (``models.weights`` → per-family
``_weights.py`` → :mod:`lucid.weights`); the per-family declarations
register themselves with :mod:`lucid.weights` for the discovery
registry on import, so importing :mod:`lucid.models.weights` (or any
member of it) also populates :func:`lucid.weights.list_pretrained`.
"""

# 2012 — AlexNet (Krizhevsky 2014 single-stream OWT after NIPS 2012)
from lucid.models.vision.alexnet._weights import AlexNetWeights

# 2015 — ResNet (He et al.)
from lucid.models.vision.resnet._weights import ResNet18Weights

__all__ = [
    # ── Vision (2012) AlexNet ─────────────────────────────────────────
    "AlexNetWeights",
    # ── Vision (2015) ResNet ──────────────────────────────────────────
    "ResNet18Weights",
]
