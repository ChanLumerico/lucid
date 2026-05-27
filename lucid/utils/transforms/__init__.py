"""``lucid.utils.transforms`` — image transform library.

An Albumentations-compatible transform library (matching class names +
constructor signatures) built tensor-native on Lucid (no numpy / PIL,
per H4).  Transforms consume / produce :class:`lucid.Tensor` images
``(C, H, W)`` or ``(B, C, H, W)``; image *decoding* (file → tensor)
lives outside this package.

Two entry points:

* **Class transforms** — :class:`Compose` a list of :class:`Transform`
  objects.  Each is applied with probability ``p`` and moves every
  typed target (:class:`Image` / :class:`Mask` / :class:`BoundingBoxes`
  / :class:`Keypoints`) consistently.
* **Functional** — stateless ops under
  :mod:`lucid.utils.transforms.functional`.

Presets (:class:`ImageClassification`) bundle a task's canonical
inference pipeline; pretrained weights ship preprocessing this way.

>>> import lucid.utils.transforms as T
>>> tf = T.Compose([
...     T.SmallestMaxSize(256),
...     T.CenterCrop(224, 224),
...     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
... ])
>>> y = tf(image)
"""

from lucid.utils.transforms import functional
from lucid.utils.transforms._base import (
    Compose,
    GeometricTransform,
    PhotometricTransform,
    Transform,
)
from lucid.utils.transforms._datatypes import (
    BoundingBoxes,
    Image,
    Keypoints,
    Mask,
)
from lucid.utils.transforms._geometric import (
    CenterCrop,
    HorizontalFlip,
    LongestMaxSize,
    RandomCrop,
    RandomResizedCrop,
    Resize,
    SmallestMaxSize,
    VerticalFlip,
)
from lucid.utils.transforms._interpolation import Interpolation
from lucid.utils.transforms._photometric import ColorJitter, Normalize
from lucid.utils.transforms._presets import ImageClassification
from lucid.utils.transforms._spatial import (
    Affine,
    Flip,
    Perspective,
    RandomRotate90,
    Rotate,
    ShiftScaleRotate,
    Transpose,
)

__all__ = [
    # Base hierarchy
    "Transform",
    "GeometricTransform",
    "PhotometricTransform",
    "Compose",
    # Interpolation
    "Interpolation",
    # Typed targets
    "Image",
    "Mask",
    "BoundingBoxes",
    "Keypoints",
    # Geometric — resize / crop
    "Resize",
    "SmallestMaxSize",
    "LongestMaxSize",
    "CenterCrop",
    "RandomCrop",
    "RandomResizedCrop",
    # Geometric — flips / transpose / rot90
    "HorizontalFlip",
    "VerticalFlip",
    "Flip",
    "Transpose",
    "RandomRotate90",
    # Geometric — affine warps
    "Rotate",
    "ShiftScaleRotate",
    "Affine",
    "Perspective",
    # Photometric
    "Normalize",
    "ColorJitter",
    # Presets
    "ImageClassification",
    # Functional submodule
    "functional",
]
