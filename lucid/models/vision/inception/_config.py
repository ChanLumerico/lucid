"""Inception v3 configuration (Szegedy et al., 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="Inception",
    citation=(
        'Szegedy, Christian, et al. "Rethinking the Inception Architecture '
        'for Computer Vision." Proceedings of the IEEE Conference on '
        "Computer Vision and Pattern Recognition (CVPR), 2016, "
        "pp. 2818–2826."
    ),
    theory=r"""
    Inception v3 is the third generation of the Inception family and
    crystallises a set of *design principles* that the authors
    extracted from earlier experiments — most notably that one should
    avoid representational bottlenecks (do not aggressively shrink
    spatial resolution in early layers) and should *factorise* large
    convolutions to gain both parameter efficiency and additional
    nonlinearity.

    The headline factorisation idea is to replace a single
    :math:`n\times n` convolution with an :math:`n\times1` followed by
    a :math:`1\times n` convolution.  For :math:`n=7` this trades
    :math:`49 C^2` parameters for :math:`14 C^2` while adding an
    intermediate ReLU and matching the original receptive field
    exactly along a separable manifold.  Inception v3 uses three
    different block topologies (Inception-A, B, C) at three spatial
    resolutions (35×35, 17×17, 8×8), each tailored to the receptive-
    field budget at that stage of the network.

    Two further innovations debuted in this paper are *label
    smoothing*, which replaces the one-hot target with a softer
    distribution :math:`(1 - \epsilon)\,\mathbf{1}_y +
    \epsilon/K` to discourage overconfident predictions, and
    *RMSProp* with a carefully tuned learning-rate schedule for
    ImageNet-scale training.  Inception v3 reaches a top-5 error of
    3.5% on ImageNet validation with only 24 M parameters,
    illustrating that thoughtful architectural surgery can outperform
    brute scaling.
    """,
)
@dataclass(frozen=True)
class InceptionConfig(ModelConfig):
    """Configuration for Inception v3.

    ``aux_logits`` enables the auxiliary classifier that attaches after the
    second InceptionC block (used during training with weight 0.4).
    ``dropout`` — main head dropout rate (0.5 in the paper).
    ``version`` — only ``"v3"`` is supported in this module.
    """

    model_type: ClassVar[str] = "inception_v3"

    num_classes: int = 1000
    in_channels: int = 3
    version: str = "v3"
    aux_logits: bool = False
    dropout: float = 0.5
