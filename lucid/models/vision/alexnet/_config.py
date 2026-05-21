"""AlexNet configuration (Krizhevsky, Sutskever & Hinton, 2012)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="AlexNet",
    citation=(
        'Krizhevsky, Alex, et al. "ImageNet Classification with Deep '
        'Convolutional Neural Networks." Advances in Neural Information '
        "Processing Systems, 2012."
    ),
    theory=r"""
    AlexNet is the architecture that re-ignited deep learning, winning
    the ILSVRC-2012 ImageNet classification challenge with a top-5
    error of 15.3% — more than ten percentage points below the
    runner-up — and demonstrating for the first time that a large
    convolutional network trained end-to-end on raw pixels could
    decisively beat hand-engineered feature pipelines.

    The architecture stacks five convolutional layers followed by
    three fully-connected layers, taking :math:`3\times224\times224`
    RGB inputs and producing 1000-way softmax logits.  Three
    contributions made training such a deep network tractable on
    2012-era hardware: (i) the *rectified linear unit*
    :math:`\phi(x) = \max(0, x)` replaced saturating nonlinearities,
    cutting training time by several factors; (ii) *dropout* with
    :math:`p=0.5` in the two large 4096-dim fully-connected layers
    regularised the ≈60 M parameters against overfitting; and
    (iii) *local response normalisation* together with overlapping
    max-pooling provided implicit lateral inhibition between feature
    maps.

    Training was split across two GPUs in a model-parallel
    configuration — a practical necessity given the 3 GB memory
    budget of a single GTX 580 at the time, and the origin of the
    "two-stream" diagram in the paper.  AlexNet's success established
    the now-standard recipe of deep ConvNet + ReLU + dropout + heavy
    data augmentation + SGD with momentum, and every subsequent
    ImageNet-scale vision model descends from it directly.
    """,
)
@dataclass(frozen=True)
class AlexNetConfig(ModelConfig):
    """Configuration for AlexNet.

    The canonical AlexNet takes 3×224×224 inputs and produces 4096-dim
    embeddings before the final classifier.

    ``dropout`` controls the two dropout layers inside the classifier
    (0.5 in the original paper).
    """

    model_type: ClassVar[str] = "alexnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5
