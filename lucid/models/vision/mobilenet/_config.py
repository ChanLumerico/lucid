"""MobileNet v1 configuration (Howard et al., 2017)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="MobileNet",
    citation=(
        'Howard, Andrew G., et al. "MobileNets: Efficient Convolutional '
        'Neural Networks for Mobile Vision Applications." arXiv preprint '
        "arXiv:1704.04861, 2017."
    ),
    theory=r"""
    MobileNet is built around the *depthwise separable convolution*, a
    factorisation that decomposes a standard convolution into two much
    cheaper stages: a per-channel **depthwise** :math:`3 \times 3`
    spatial filter followed by a :math:`1 \times 1` **pointwise**
    convolution that mixes channels.  For an input of :math:`M`
    channels and an output of :math:`N` channels with kernel size
    :math:`D_K`, the cost drops from
    :math:`D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F` to

    .. math::

        D_K \cdot D_K \cdot M \cdot D_F \cdot D_F
        \;+\; M \cdot N \cdot D_F \cdot D_F,

    a reduction of roughly :math:`1/N + 1/D_K^2`.  For
    :math:`D_K = 3` and the typical :math:`N` of a few hundred,
    that is an 8–9× FLOPs saving with only a small accuracy loss
    on ImageNet.

    Two scalar hyper-parameters expose a smooth accuracy–latency
    trade-off curve at inference time without retraining the whole
    family.  The **width multiplier** :math:`\alpha \in (0, 1]`
    uniformly thins every layer (input and output channels become
    :math:`\alpha M` and :math:`\alpha N`), giving four standard
    variants at :math:`\alpha \in \{1.0, 0.75, 0.5, 0.25\}`.  The
    **resolution multiplier** :math:`\rho` rescales the input spatial
    size — it does not change the model itself, only the feature
    maps flowing through it, and is therefore not enforced inside
    the network.

    Together these two knobs let MobileNet-v1 cover a wide spectrum
    of mobile/embedded budgets from sub-mW edge devices up to
    desktop-class deployment, while keeping a single training recipe
    and a single :class:`MobileNetV1Config` schema.
    """,
)
@dataclass(frozen=True)
class MobileNetV1Config(ModelConfig):
    """Configuration for MobileNet v1.

    ``width_mult`` — uniform channel multiplier (α in the paper).
      1.0 → full model; 0.75 / 0.5 / 0.25 → slimmer variants.

    ``resolution_mult`` — spatial resolution multiplier (ρ).
      Typically applied to the input, not the architecture itself.
      Kept here for documentation parity; not enforced inside the model.

    ``dropout`` — classifier dropout (0.001 in the original paper).
    """

    model_type: ClassVar[str] = "mobilenet_v1"

    num_classes: int = 1000
    in_channels: int = 3
    width_mult: float = 1.0
    dropout: float = 0.001
