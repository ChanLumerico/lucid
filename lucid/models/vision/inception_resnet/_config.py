"""Inception-ResNet v2 configuration (Szegedy et al., 2016)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="Inception-ResNet",
    citation=(
        'Szegedy, Christian, et al. "Inception-v4, Inception-ResNet and '
        'the Impact of Residual Connections on Learning." Proceedings of '
        "the AAAI Conference on Artificial Intelligence, 2017."
    ),
    theory=r"""
    Inception-ResNet hybridises the two dominant architectural ideas
    of the early ResNet era: the *multi-branch Inception module* and
    the *residual shortcut*.  Each Inception sub-network is wrapped in
    an identity skip connection, so the block computes

    .. math::

        y = x + \alpha \cdot \mathcal{F}_{\text{inception}}(x),

    where :math:`\alpha` is a small fixed scale factor (typically
    0.10–0.30) applied to the Inception branch before addition.  This
    *residual scaling* trick was introduced specifically for
    Inception-ResNet because the unscaled residual branches
    occasionally caused training to diverge on networks with many
    filters per Inception module.

    Three block families — Block35, Block17, Block8 — mirror
    Inception v3's A/B/C topologies at the 35×35, 17×17, and 8×8
    resolutions respectively, but each is now a residual unit.  The
    network's stem is shared with Inception v4 (an aggressive
    multi-branch downsampler).  Reduction-A and Reduction-B blocks
    perform spatial reduction between groups of residual Inception
    blocks while doubling the channel budget.

    The empirical message of the paper is precise and influential:
    residual connections *do not* improve final accuracy beyond what
    a comparably-sized non-residual Inception v4 achieves, but they
    *dramatically* accelerate convergence — Inception-ResNet v2
    reaches the same accuracy roughly twice as fast.  Final top-5
    error on ImageNet is 3.08%, with the residual variants holding a
    convergence-speed edge over their non-residual siblings throughout
    training.
    """,
)
@dataclass(frozen=True)
class InceptionResNetConfig(ModelConfig):
    """Configuration for Inception-ResNet v2.

    Uses the same stem as Inception-v4, followed by:
      - Mixed_5b (192 → 320)
      - 10× Block35  (scale_a, default 0.17)
      - Mixed_6a / Reduction-A (320 → 1088)
      - 20× Block17  (scale_b, default 0.10)
      - Mixed_7a / Reduction-B (1088 → 2080)
      -  9× Block8   (scale_c, default 0.20) + 1× Block8 (no ReLU)
      - Conv2d_7b 1×1 projection (2080 → 1536)
      - AdaptiveAvgPool → Dropout → FC(1536, num_classes)

    ``scale_a`` — residual scale for Block35 (paper suggests 0.17).
    ``scale_b`` — residual scale for Block17 (paper suggests 0.10).
    ``scale_c`` — residual scale for Block8  (paper suggests 0.20).
    ``dropout`` — head dropout rate (0.2 in the paper).
    """

    model_type: ClassVar[str] = "inception_resnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.2
    scale_a: float = 0.17  # Block35 (Inception-A)
    scale_b: float = 0.10  # Block17 (Inception-B)
    scale_c: float = 0.20  # Block8  (Inception-C)
