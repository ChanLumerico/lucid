"""Ready-made transform presets.

Presets bundle the canonical inference pipeline for a task so pretrained
weights can ship their exact preprocessing.  :class:`ImageClassification`
is the ImageNet-style preset consumed by
``lucid.models.*._weights`` (via :class:`~lucid.weights.WeightEntry`).
"""

from lucid._tensor import Tensor
from lucid.utils.transforms._base import Compose, Transform
from lucid.utils.transforms._geometric import CenterCrop, Resize
from lucid.utils.transforms._photometric import Normalize


class ImageClassification(Transform):
    r"""Standard ImageNet classification preprocessing preset.

    Equivalent to torchvision's ``ImageClassification`` preset:
    ``Resize(resize_size)`` (shorter side) → ``CenterCrop(crop_size)`` →
    ``Normalize(mean, std)``.

    Parameters
    ----------
    crop_size : int
        Side length of the square center crop fed to the model.
    resize_size : int, optional, default=256
        Shorter-side length before cropping.
    mean : tuple of float, optional
        Per-channel mean; defaults to ImageNet ``(0.485, 0.456, 0.406)``.
    std : tuple of float, optional
        Per-channel std; defaults to ImageNet ``(0.229, 0.224, 0.225)``.
    interpolation : str, optional, default="bilinear"
        Resize interpolation mode.

    Notes
    -----
    Input is assumed to be a float :class:`lucid.Tensor` already scaled
    to ``[0, 1]`` (prepend :class:`~lucid.utils.transforms.Rescale` when
    starting from uint8).  Accepts ``(C, H, W)`` or ``(B, C, H, W)`` and
    preserves the input rank.

    Examples
    --------
    >>> from lucid.utils.transforms import ImageClassification
    >>> tf = ImageClassification(crop_size=224, resize_size=256)
    >>> tf(image).shape
    (3, 224, 224)
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
        interpolation: str = "bilinear",
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

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        return self._pipeline(img)

    def __repr__(self) -> str:
        return (
            f"ImageClassification(crop_size={self.crop_size}, "
            f"resize_size={self.resize_size}, mean={self.mean}, "
            f"std={self.std}, interpolation={self.interpolation!r})"
        )
