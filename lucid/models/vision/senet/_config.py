"""SENet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="SENet",
    citation=(
        'Hu, Jie, et al. "Squeeze-and-Excitation Networks." Proceedings '
        "of the IEEE Conference on Computer Vision and Pattern "
        "Recognition, 2018, pp. 7132–7141."
    ),
    theory=r"""
    SENet introduces the **Squeeze-and-Excitation block**, a tiny
    drop-in channel-attention module that recalibrates the relative
    importance of each feature-map channel after every convolutional
    stage.  The block runs in three phases.

    The **squeeze** phase reduces each channel to a single scalar by
    global average pooling over the spatial dimensions:

    .. math::

        z_c = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W}
              u_c(i, j),

    yielding a channel descriptor :math:`z \in \mathbb{R}^{C}` that
    summarises the global response of every channel.

    The **excitation** phase passes :math:`z` through a small
    two-layer fully-connected bottleneck with a reduction ratio
    :math:`r` (typically 16) and a sigmoid output:

    .. math::

        s = \sigma\big( W_2 \,\delta( W_1 z ) \big),
        \qquad W_1 \in \mathbb{R}^{C/r \times C}, \;
        W_2 \in \mathbb{R}^{C \times C/r},

    where :math:`\delta` is ReLU and :math:`\sigma` is the
    sigmoid.  The output :math:`s \in (0, 1)^{C}` is interpreted
    as per-channel gating weights.

    Finally the **scale** phase modulates the original feature
    map channel-wise:

    .. math::

        \tilde{u}_c = s_c \cdot u_c.

    Plugging this block into the end of every residual unit of a
    ResNet-50 (giving SE-ResNet-50) adds only ~10% extra parameters
    and negligible FLOPs while reducing ImageNet top-5 error by
    roughly 1.5 percentage points — a result that won the ILSVRC
    2017 classification challenge.  In Lucid :class:`SENetConfig`
    parameterises every SE-ResNet variant (18 / 34 / 50 / 101 / 152)
    with the same ``layers`` and ``block_type`` knobs as ResNet plus
    the SE-specific ``reduction`` ratio.
    """,
)
@dataclass(frozen=True)
class SENetConfig(ModelConfig):
    """Unified config for all SE-ResNet variants (Hu et al., 2017).

    Paper: "Squeeze-and-Excitation Networks"

    ``block_type`` selects BasicBlock (SE-ResNet-18/34) or Bottleneck
    (SE-ResNet-50/101/152).  ``layers`` is the per-stage repetition count.
    ``reduction`` is the channel reduction ratio in the SE block (default 16).
    """

    model_type: ClassVar[str] = "senet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    reduction: int = 16
    block_type: str = "bottleneck"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
