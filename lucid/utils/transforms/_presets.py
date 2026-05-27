"""Ready-made transform presets.

Presets bundle the canonical inference pipeline for a task so pretrained
weights can ship their exact preprocessing.  :class:`ImageClassification`
is the ImageNet-style preset consumed by ``lucid.models.*._weights``
(via :class:`~lucid.weights.WeightEntry`).
"""

from typing import cast

from lucid._tensor import Tensor
from lucid.utils.transforms._base import Compose, Empty, Transform, _NoParams
from lucid.utils.transforms._geometric import CenterCrop, Resize
from lucid.utils.transforms._interpolation import Interpolation
from lucid.utils.transforms._photometric import Normalize


class ImageClassification(_NoParams, Transform[Empty]):
    r"""Standard ImageNet classification preprocessing preset.

    ``Resize(resize_size)`` (shorter side) → ``CenterCrop(crop_size)`` →
    ``Normalize(mean, std)``, packaged as one callable.  Delegates to an
    internal :class:`~lucid.utils.transforms.Compose`, so it also
    threads multi-target samples through its (geometric) inner stages.

    Parameters
    ----------
    crop_size : int
        Square center-crop side fed to the model.
    resize_size : int, optional, default=256
        Shorter-side length before cropping.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"
        Resize interpolation mode.

    Notes
    -----
    Input is assumed already scaled to ``[0, 1]`` (prepend
    :class:`~lucid.utils.transforms.Rescale` for uint8).
    """

    _IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    _IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        crop_size: int,
        *,
        resize_size: int = 256,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.mean = mean if mean is not None else self._IMAGENET_MEAN
        self.std = std if std is not None else self._IMAGENET_STD
        self.interpolation = interpolation
        self._pipeline = Compose(
            [
                Resize(resize_size, interpolation=interpolation),
                CenterCrop(crop_size),
                Normalize(self.mean, self.std),
            ]
        )

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return cast(Tensor, self._pipeline(img))

    def __call__(self, inputs: object) -> object:
        # Delegate wholesale so multi-target samples thread through the
        # inner (geometric) stages correctly.
        return self._pipeline(inputs)

    def __repr__(self) -> str:
        return (
            f"ImageClassification(crop_size={self.crop_size}, "
            f"resize_size={self.resize_size}, mean={self.mean}, "
            f"std={self.std}, interpolation={self.interpolation})"
        )
