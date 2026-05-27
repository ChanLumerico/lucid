"""``lucid.utils.transforms`` — image transform library.

A Lucid-native take on ``torchvision.transforms`` v2: a composable
pipeline of transforms that operate on :class:`lucid.Tensor` images
(``(C, H, W)`` or ``(B, C, H, W)``).  Because ``lucid/`` may not import
numpy / PIL (H4), transforms are tensor-in / tensor-out; image decoding
(file → tensor) lives outside this package.

Two entry points:

* **Class transforms** — :class:`Compose` a list of :class:`Transform`
  objects (:class:`Resize`, :class:`CenterCrop`, :class:`Normalize`,
  :class:`Rescale`, …).
* **Functional** — stateless ops under
  :mod:`lucid.utils.transforms.functional` (``F.resize`` / ``F.normalize``
  / …) that the classes wrap.

Presets (:class:`ImageClassification`) bundle a task's canonical
inference pipeline; pretrained weights ship their preprocessing this
way (see :mod:`lucid.weights`).

>>> import lucid.utils.transforms as T
>>> tf = T.Compose([T.Resize(256), T.CenterCrop(224),
...                 T.Normalize(mean=(0.485, 0.456, 0.406),
...                             std=(0.229, 0.224, 0.225))])
>>> y = tf(image)        # image: lucid.Tensor (3, H, W) in [0, 1]
"""

from lucid.utils.transforms import functional
from lucid.utils.transforms._base import Compose, Transform
from lucid.utils.transforms._datatypes import BoundingBoxes, Image, Mask
from lucid.utils.transforms._geometric import (
    CenterCrop,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
)
from lucid.utils.transforms._photometric import (
    ColorJitter,
    Normalize,
    RandomErasing,
    Rescale,
)
from lucid.utils.transforms._presets import ImageClassification

__all__ = [
    # Base
    "Transform",
    "Compose",
    # Typed targets (multi-target dispatch)
    "Image",
    "Mask",
    "BoundingBoxes",
    # Geometric — deterministic
    "Resize",
    "CenterCrop",
    "Pad",
    # Geometric — random
    "RandomCrop",
    "RandomResizedCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    # Photometric — deterministic
    "Normalize",
    "Rescale",
    # Photometric — random
    "ColorJitter",
    "RandomErasing",
    # Presets
    "ImageClassification",
    # Functional submodule
    "functional",
]
