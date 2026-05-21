"""ZFNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="ZFNet",
    citation=(
        'Zeiler, Matthew D., and Rob Fergus. "Visualizing and Understanding '
        'Convolutional Networks." European Conference on Computer Vision '
        "(ECCV), 2014."
    ),
    theory=r"""
    ZFNet is the ILSVRC-2013 winner and is best understood as an
    AlexNet that was *tuned by looking at what the network had
    learned*.  Zeiler and Fergus introduced the *deconvolutional
    network* visualisation technique — projecting individual feature-
    map activations back to the input pixel space — and used the
    resulting diagnostic images to reshape the early layers of the
    network.

    Two specific changes distinguish ZFNet from AlexNet.  The first
    convolution was reduced from an :math:`11\times11` stride-4 kernel
    to a :math:`7\times7` stride-2 kernel, addressing the observation
    that AlexNet's first layer learned a mixture of extremely
    high-frequency and dead filters.  The second convolution's stride
    was simultaneously dropped from 2 to a more moderate value, giving
    denser low-level feature coverage and avoiding aliasing artefacts
    that the visualisations had revealed.  The remaining layers retain
    AlexNet's topology.

    These modest topological changes — together with the *methodology*
    of using visualisation to drive architectural choices — were
    influential out of proportion to their parameter cost.  ZFNet
    achieved a top-5 error of 11.7% on ImageNet, and the
    deconvolutional-network analysis it popularised is the conceptual
    ancestor of every later interpretability tool from Grad-CAM to
    feature inversion.
    """,
)
@dataclass(frozen=True)
class ZFNetConfig(ModelConfig):
    """Configuration for ZFNet (Zeiler & Fergus, 2013).

    ZFNet is an AlexNet variant with modified first two conv layers:
    - Conv1: 7×7 stride=2 pad=1 (vs AlexNet's 11×11 stride=4)
    - Conv2: 5×5 stride=2 (vs AlexNet's 5×5 stride=1)
    """

    model_type: ClassVar[str] = "zfnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5
