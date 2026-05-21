"""CSPNet configuration dataclass (Wang et al., 2019)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="CSPNet",
    citation=(
        'Wang, Chien-Yao, et al. "CSPNet: A New Backbone that can '
        'Enhance Learning Capability of CNN." Proceedings of the '
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition "
        "Workshops, 2020, pp. 390–391."
    ),
    theory=r"""
    CSPNet ("Cross-Stage Partial Network") is a generic backbone
    transformation that can be wrapped around almost any
    dense-or-residual stage — ResNet, ResNeXt, DenseNet — to cut
    its compute and memory while *improving* its accuracy.  The
    core observation is that during back-propagation the gradient
    flowing through a dense block is *duplicated* many times across
    repeated dense connections, which both wastes computation and
    can confuse the optimisation by re-using the same gradient
    information in many places.

    The CSP transformation factors each stage as follows.  The
    input feature map is split along the channel dimension into
    two halves:

    .. math::

        x = [x', x''], \qquad
        x' \in \mathbb{R}^{H \times W \times C/2}, \;
        x'' \in \mathbb{R}^{H \times W \times C/2}.

    Only :math:`x''` is fed through the original dense / residual
    block to produce :math:`y'' = \mathcal{F}(x'')`; the other half
    :math:`x'` *bypasses* the block entirely.  At the end of the
    stage the two halves are concatenated and projected:

    .. math::

        y = \mathrm{Transition}\big( [x', \mathcal{F}(x'')] \big).

    Because :math:`x'` never enters the dense block, the duplicated
    gradient is split into two truncated paths that share no
    intermediate activations — which both reduces redundant gradient
    information *and* halves the FLOPs of the heavy block.  Applied
    to a ResNet-50 / ResNeXt-50 / DenseNet-201 backbone, CSP cuts
    compute by 10–20% while improving ImageNet top-1 by a fraction
    of a percent.  In Lucid the :class:`CSPNetConfig` parameterises
    the CSPResNet variants — four stages of CSP-wrapped bottleneck
    blocks with configurable per-stage depths and channel widths.
    """,
)
@dataclass(frozen=True)
class CSPNetConfig(ModelConfig):
    """Unified config for CSPResNet variants.

    ``layers`` — per-stage block counts (4 stages total).
    ``channels`` — base channel widths per stage before CSP branching.
    """

    model_type: ClassVar[str] = "cspnet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 3, 5, 2)
    channels: tuple[int, ...] = (64, 128, 256, 512)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "channels", tuple(self.channels))
