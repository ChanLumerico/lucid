"""Inference preprocessing transforms bundled with pretrained weights.

Every :class:`~lucid.weights._base.WeightEntry` carries the exact
preprocessing pipeline its checkpoint was trained with, so loading a
pretrained model also hands the caller the matching transform —
mirroring torchvision's ``weights.transforms()`` ergonomics.

Transforms operate on **Lucid tensors only** (no PIL / numpy): image
*decoding* (file → tensor) is the caller's responsibility and lives
outside :mod:`lucid.weights`.  A transform's input is a float
:class:`lucid.Tensor` with pixel values already scaled to ``[0, 1]``,
shaped ``(C, H, W)`` or ``(B, C, H, W)``.
"""

import abc

import lucid
import lucid.nn.functional as F
from lucid._tensor import Tensor


class Transform(abc.ABC):
    """Abstract base for an inference preprocessing pipeline.

    Subclasses implement :meth:`__call__`, mapping a raw image tensor
    to the normalized tensor the model expects.
    """

    @abc.abstractmethod
    def __call__(self, img: Tensor) -> Tensor:
        """Apply the transform to ``img`` and return the result.

        Parameters
        ----------
        img : Tensor
            Float image tensor in ``[0, 1]``, shaped ``(C, H, W)`` or
            ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Preprocessed tensor ready to feed the model.
        """
        raise NotImplementedError


class ImageClassification(Transform):
    r"""Standard ImageNet-style classification preprocessing.

    Reproduces the torchvision ``ImageClassification`` preset: resize
    the shorter side to ``resize_size`` (preserving aspect ratio),
    center-crop to ``crop_size`` × ``crop_size``, then normalize each
    channel with ``mean`` / ``std``.

    Parameters
    ----------
    crop_size : int
        Side length of the square center crop fed to the model
        (commonly ``224``).
    resize_size : int, optional, default=256
        Target length of the shorter image side before cropping.
    mean : tuple of float, optional
        Per-channel mean subtracted during normalization.  Defaults to
        the ImageNet statistics ``(0.485, 0.456, 0.406)``.
    std : tuple of float, optional
        Per-channel standard deviation.  Defaults to the ImageNet
        statistics ``(0.229, 0.224, 0.225)``.
    interpolation : str, optional, default="bilinear"
        Resize interpolation mode, forwarded to
        :func:`lucid.nn.functional.interpolate`.

    Notes
    -----
    Input pixel values are assumed to already be scaled to ``[0, 1]``
    (i.e. divided by 255).  The transform accepts an unbatched
    ``(C, H, W)`` tensor or a batched ``(B, C, H, W)`` tensor and
    returns the same rank it was given.

    Examples
    --------
    >>> from lucid.weights import ImageClassification
    >>> tf = ImageClassification(crop_size=224, resize_size=256)
    >>> x = tf(image)            # image: lucid.Tensor (3, H, W) in [0, 1]
    >>> x.shape
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

    def __call__(self, img: Tensor) -> Tensor:
        unbatched = img.ndim == 3
        x = img[None] if unbatched else img

        x = self._resize_shorter_side(x)
        x = self._center_crop(x)
        x = self._normalize(x)

        return x[0] if unbatched else x

    def _resize_shorter_side(self, x: Tensor) -> Tensor:
        """Resize so the shorter spatial side equals ``resize_size``."""
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h <= w:
            new_h = self.resize_size
            new_w = int(round(w * self.resize_size / h))
        else:
            new_w = self.resize_size
            new_h = int(round(h * self.resize_size / w))

        align = False if self.interpolation in ("bilinear", "bicubic") else None
        return F.interpolate(
            x, size=(new_h, new_w), mode=self.interpolation, align_corners=align
        )

    def _center_crop(self, x: Tensor) -> Tensor:
        """Crop a centered ``crop_size`` × ``crop_size`` window."""
        h, w = int(x.shape[-2]), int(x.shape[-1])
        crop = self.crop_size
        top = max((h - crop) // 2, 0)
        left = max((w - crop) // 2, 0)
        return x[..., top : top + crop, left : left + crop]

    def _normalize(self, x: Tensor) -> Tensor:
        """Subtract per-channel mean and divide by per-channel std."""
        c = int(x.shape[-3])
        mean = lucid.tensor(list(self.mean), dtype=x.dtype).reshape(1, c, 1, 1)
        std = lucid.tensor(list(self.std), dtype=x.dtype).reshape(1, c, 1, 1)
        return (x - mean) / std
