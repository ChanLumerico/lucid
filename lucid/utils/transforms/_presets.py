"""Ready-made transform presets — Lucid's pretrained-pipeline contract.

A *preset* is a high-level, task-specific preprocessing pipeline that
knows how to serialise itself to a JSON-friendly dict and reconstruct
back — the canonical handoff between a :class:`~lucid.weights.WeightEntry`
and the actual transform sequence a pretrained model expects.

Design borrows from both Hugging Face ``transformers`` (model-family
``ImageProcessor`` + ``preprocessor_config.json``) and timm
(``default_cfg`` flat dict + builder).  Each preset is a class
(HF-style); the standard task presets (:class:`ImageClassification`,
:class:`ImageClassificationAugment`, :class:`Detection`,
:class:`Segmentation`, :class:`Pose`) cover the common cases without
per-model boilerplate (timm-style).  When a specific model needs
custom preprocessing (e.g. Inception's ``299×299`` bicubic), subclass
the matching task preset and override what differs.

Round-trip contract::

    cfg = preset.to_dict()                # JSON-friendly
    restored = AutoTransformsPreset.from_dict(cfg)
    assert type(restored) is type(preset)
    assert restored.to_dict() == cfg

Every preset is also a regular :class:`~lucid.utils.transforms.Transform`
— call it on an image tensor or a multi-target sample dict, both work.

Examples
--------
>>> import lucid
>>> from lucid.utils.transforms import (
...     ImageClassification, AutoTransformsPreset,
... )
>>> preset = ImageClassification(crop_size=224, resize_size=256)
>>> y = preset(lucid.rand(3, 300, 400))           # tensor in, tensor out
>>> cfg = preset.to_dict()                         # round-trip
>>> back = AutoTransformsPreset.from_dict(cfg)
>>> back.to_dict() == cfg
True
"""

import abc
from typing import ClassVar, cast, override

from lucid._tensor import Tensor
from lucid.utils.transforms._autoaugment import (
    AutoAugment,
    RandAugment,
    TrivialAugmentWide,
)
from lucid.utils.transforms._base import (
    BboxParams,
    Compose,
    Empty,
    Transform,
    TransformLike,
    _NoParams,
)
from lucid.utils.transforms._erasing import RandomErasing
from lucid.utils.transforms._geometric import (
    CenterCrop,
    HorizontalFlip,
    LongestMaxSize,
    RandomResizedCrop,
    SmallestMaxSize,
)
from lucid.utils.transforms._crop import PadIfNeeded
from lucid.utils.transforms._interpolation import Interpolation
from lucid.utils.transforms._photometric import ColorJitter, Normalize

# ── registry + auto-resolver ────────────────────────────────────────


_PRESET_REGISTRY: dict[str, type[TransformsPreset]] = {}


def _register_preset(cls: type[TransformsPreset]) -> type[TransformsPreset]:
    """Class decorator: register ``cls`` under its ``preset_type`` key."""
    key = cls.preset_type
    if key in _PRESET_REGISTRY and _PRESET_REGISTRY[key] is not cls:
        raise ValueError(
            f"preset key {key!r} already registered to "
            f"{_PRESET_REGISTRY[key].__qualname__}; conflict with "
            f"{cls.__qualname__}"
        )
    _PRESET_REGISTRY[key] = cls
    return cls


# ── base class ──────────────────────────────────────────────────────


_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class TransformsPreset(_NoParams, Transform[Empty], abc.ABC):
    r"""Abstract base for task-specific preprocessing presets.

    A preset is a thin wrapper over an internal :class:`Compose`
    pipeline (assembled in :meth:`__init__` from the preset's task-
    specific defaults) plus a serialisation contract.  Subclasses
    must:

    1. Set the :attr:`preset_type` class variable to a unique short
       string used in ``config.json``.
    2. Declare the constructor's typed kwargs (no ``**kwargs`` —
       enforces the schema explicitly).
    3. Build ``self._pipeline`` (a :class:`Compose`) inside
       :meth:`__init__`.
    4. Implement :meth:`_init_kwargs` returning the dict of constructor
       arguments needed to reconstruct an identical instance.

    Attributes
    ----------
    preset_type : ClassVar[str]
        Short identifier (e.g. ``"ImageClassification"``) used as the
        ``preprocessor_type`` key in :meth:`to_dict` / :meth:`from_dict`.
    """

    preset_type: ClassVar[str]

    _pipeline: Compose

    # ── serialisation contract ──────────────────────────────────────

    @abc.abstractmethod
    def _init_kwargs(self) -> dict[str, object]:
        """Return the constructor kwargs that reconstruct ``self``.

        The values must be JSON-friendly (``int`` / ``float`` / ``str``
        / ``list`` / ``dict`` / ``bool`` / ``None``).
        """

    def to_dict(self) -> dict[str, object]:
        r"""Serialise the preset to a JSON-friendly dict.

        Mirrors Hugging Face's ``preprocessor_config.json`` shape:
        ``{"preprocessor_type": <class-key>, "init_kwargs": {…}}``.

        Returns
        -------
        dict
            Two-key dict suitable for :func:`json.dump` and
            :meth:`AutoTransformsPreset.from_dict` round-trip.
        """
        return {
            "preprocessor_type": self.preset_type,
            "init_kwargs": self._init_kwargs(),
        }

    @classmethod
    def from_dict(cls, cfg: dict[str, object]) -> TransformsPreset:
        r"""Reconstruct a preset instance from a :meth:`to_dict` payload.

        When called on the base :class:`TransformsPreset`, dispatches
        to the registered subclass named by ``cfg["preprocessor_type"]``;
        when called on a concrete subclass, verifies the key matches
        and constructs that subclass directly.

        Parameters
        ----------
        cfg : dict
            ``{"preprocessor_type": ..., "init_kwargs": {...}}``.

        Returns
        -------
        TransformsPreset
            New instance of the named subclass.

        Raises
        ------
        KeyError
            If ``preprocessor_type`` is not in the registry.
        ValueError
            If ``cfg`` is missing required keys or ``preprocessor_type``
            disagrees with the class :meth:`from_dict` was called on.
        """
        key = cfg.get("preprocessor_type")
        kwargs = cfg.get("init_kwargs", {})
        if not isinstance(key, str):
            raise ValueError(
                f"from_dict: missing or non-string 'preprocessor_type' in {cfg!r}"
            )
        if not isinstance(kwargs, dict):
            raise ValueError(
                f"from_dict: 'init_kwargs' must be a dict, got {type(kwargs).__name__}"
            )
        target = _PRESET_REGISTRY.get(key)
        if target is None:
            raise KeyError(
                f"unknown preset type {key!r}; registered: "
                f"{sorted(_PRESET_REGISTRY)}"
            )
        if cls is not TransformsPreset and target is not cls:
            raise ValueError(
                f"from_dict on {cls.__qualname__} got cfg for {target.__qualname__}"
            )
        return target(**kwargs)

    # ── Transform interface (delegates to the inner pipeline) ───────

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return cast(Tensor, self._pipeline(img))

    @override
    def __call__(self, inputs: object) -> object:
        # Wholesale delegation so multi-target samples thread through
        # the inner stages with consistent params per call.
        return self._pipeline(inputs)

    @override
    def __repr__(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs().items())
        return f"{type(self).__name__}({kw})"


class AutoTransformsPreset:
    r"""Resolver mirroring Hugging Face's ``AutoImageProcessor``.

    Stateless utility — no instances are constructed; the classmethods
    look up the registry by ``preprocessor_type`` and dispatch.

    Examples
    --------
    >>> from lucid.utils.transforms import AutoTransformsPreset
    >>> cfg = {
    ...     "preprocessor_type": "ImageClassification",
    ...     "init_kwargs": {"crop_size": 224, "resize_size": 256},
    ... }
    >>> preset = AutoTransformsPreset.from_dict(cfg)
    >>> type(preset).__name__
    'ImageClassification'
    """

    @staticmethod
    def from_dict(cfg: dict[str, object]) -> TransformsPreset:
        """Reconstruct a preset from a :meth:`TransformsPreset.to_dict` payload.

        Dispatches on the ``preprocessor_type`` key to the registered
        :class:`TransformsPreset` subclass and forwards ``init_kwargs``
        to its constructor.  Symmetric with :meth:`TransformsPreset.to_dict`
        — the round-trip ``from_dict(p.to_dict())`` produces a preset
        with identical behaviour.

        Parameters
        ----------
        cfg : dict
            ``{"preprocessor_type": <class-key>, "init_kwargs": {...}}``
            payload — typically loaded from a pretrained model's
            ``preprocessor_config.json``.

        Returns
        -------
        TransformsPreset
            Fresh instance of the named subclass.

        Raises
        ------
        ValueError
            If ``preprocessor_type`` is missing or names an unknown
            preset; if ``init_kwargs`` is not a dict.
        """
        return TransformsPreset.from_dict(cfg)

    @staticmethod
    def registered() -> list[str]:
        """Return the sorted list of registered preset type keys.

        Useful for discovery / introspection — every returned name is
        a legal value of the ``preprocessor_type`` field accepted by
        :meth:`from_dict`.
        """
        return sorted(_PRESET_REGISTRY)


# ── concrete presets ────────────────────────────────────────────────


@_register_preset
class ImageClassification(TransformsPreset):
    r"""Standard ImageNet classification *inference* preset.

    Pipeline: ``SmallestMaxSize(resize_size)`` → ``CenterCrop(crop_size)``
    → ``Normalize(mean, std)``.  This is the torchvision /
    Albumentations canonical eval pipeline shipped with most
    pretrained image-classification weights.

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
    :class:`~lucid.utils.transforms.ToFloat` for uint8).
    """

    preset_type: ClassVar[str] = "ImageClassification"

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
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        self._pipeline = Compose(
            [
                SmallestMaxSize(resize_size, interpolation=interpolation),
                CenterCrop(crop_size, crop_size),
                Normalize(self.mean, self.std, max_pixel_value=1.0),
            ]
        )

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (
                interp.value if isinstance(interp, Interpolation) else interp
            ),
        }


def _parse_auto_augment(spec: str) -> TransformLike:
    r"""Resolve a timm-style ``auto_augment`` spec to a Transform instance.

    Accepted forms::

        "ta_wide"                       → TrivialAugmentWide()
        "ta"                            → TrivialAugmentWide(num_magnitude_bins=10)
        "ra"                            → RandAugment()  (defaults: n=2, m=9)
        "ra-mM"                         → RandAugment(magnitude=M)
        "ra-nN"                         → RandAugment(num_ops=N)
        "ra-mM-nN" / "ra-nN-mM"         → RandAugment(num_ops=N, magnitude=M)
        "aa_imagenet" / "aa_cifar10"    → AutoAugment(policy=...)
        "aa_svhn"

    Mirrors the reference-framework convention so user code that
    already supplies ``"ra-m9"`` etc. works without translation.
    """
    if spec == "ta_wide":
        return TrivialAugmentWide(p=1.0)
    if spec == "ta":
        return TrivialAugmentWide(num_magnitude_bins=10, p=1.0)
    if spec.startswith("ra"):
        parts = spec.split("-")
        m, n = 9, 2
        for part in parts[1:]:
            if part.startswith("m"):
                m = int(part[1:])
            elif part.startswith("n"):
                n = int(part[1:])
            else:
                raise ValueError(
                    f"unrecognised RandAugment fragment {part!r} in spec {spec!r}"
                )
        return RandAugment(num_ops=n, magnitude=m, p=1.0)
    if spec.startswith("aa_"):
        policy = spec[3:]
        return AutoAugment(policy=policy, p=1.0)
    raise ValueError(
        f"unknown auto_augment spec {spec!r}; expected one of "
        '"ta", "ta_wide", "ra[-mM][-nN]", "aa_imagenet", "aa_cifar10", "aa_svhn"'
    )


@_register_preset
class ImageClassificationAugment(TransformsPreset):
    r"""Standard ImageNet classification *training* preset (augmentation).

    Pipeline (stages 4–7 conditional, all share the same Normalize stats)::

        1. RandomResizedCrop(crop_size, scale, ratio)
        2. HorizontalFlip(p=hflip_prob)            — skipped if hflip_prob = 0
        3. AutoAugment / RandAugment / TAW         — skipped if auto_augment = None
        4. ColorJitter(brightness=contrast=...)    — skipped if color_jitter = 0
        5. Normalize(mean, std, max_pixel_value=1)
        6. RandomErasing(p=random_erasing)         — skipped if random_erasing = 0

    Mirrors the reference-framework training recipe shipped with most
    classification models.  Setting ``auto_augment`` and/or
    ``random_erasing`` reaches the strong-augmentation recipe used by
    timm / torchvision references.

    Parameters
    ----------
    crop_size : int
        Square crop fed to the model.
    scale : tuple of float, optional, default=(0.08, 1.0)
        ``RandomResizedCrop`` scale range.
    ratio : tuple of float, optional, default=(0.75, 1.3333)
        ``RandomResizedCrop`` aspect-ratio range.
    color_jitter : float, optional, default=0.4
        Brightness / contrast / saturation jitter strength.  Zero
        disables ``ColorJitter`` entirely.
    hflip_prob : float, optional, default=0.5
        Horizontal-flip probability.  Zero disables the flip.
    auto_augment : str or None, optional, default=None
        AutoAugment-family policy.  See :func:`_parse_auto_augment`
        for accepted spec strings.  ``None`` disables this stage —
        recommended unless you already have a tuned recipe.
    random_erasing : float, optional, default=0.0
        :class:`~lucid.utils.transforms.RandomErasing` probability
        applied *after* :class:`Normalize` (mirrors reference
        ordering).  ``0.0`` disables the stage.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"
    """

    preset_type: ClassVar[str] = "ImageClassificationAugment"

    def __init__(
        self,
        crop_size: int,
        *,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        color_jitter: float = 0.4,
        hflip_prob: float = 0.5,
        auto_augment: str | None = None,
        random_erasing: float = 0.0,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        if not 0.0 <= random_erasing <= 1.0:
            raise ValueError(f"random_erasing must be in [0, 1], got {random_erasing}")
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
        self.color_jitter = color_jitter
        self.hflip_prob = hflip_prob
        self.auto_augment = auto_augment
        self.random_erasing = random_erasing
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation

        stages: list[TransformLike] = [
            RandomResizedCrop(
                crop_size,
                crop_size,
                scale=scale,
                ratio=ratio,
                interpolation=interpolation,
                p=1.0,
            ),
        ]
        if hflip_prob > 0.0:
            stages.append(HorizontalFlip(p=hflip_prob))
        if auto_augment is not None:
            stages.append(_parse_auto_augment(auto_augment))
        if color_jitter > 0.0:
            stages.append(
                ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=0.0,
                    p=1.0,
                )
            )
        stages.append(Normalize(self.mean, self.std, max_pixel_value=1.0))
        if random_erasing > 0.0:
            stages.append(RandomErasing(p=random_erasing))
        self._pipeline = Compose(stages)

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "scale": list(self.scale),
            "ratio": list(self.ratio),
            "color_jitter": self.color_jitter,
            "hflip_prob": self.hflip_prob,
            "auto_augment": self.auto_augment,
            "random_erasing": self.random_erasing,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (
                interp.value if isinstance(interp, Interpolation) else interp
            ),
        }


@_register_preset
class Detection(TransformsPreset):
    r"""Object-detection preset — coordinates ride with the image.

    Pipeline: ``LongestMaxSize(max_size)`` → ``PadIfNeeded(max_size,
    max_size)`` → ``Normalize(mean, std)`` plus a
    :class:`BboxParams(min_area, min_visibility)` policy on the
    enclosing :class:`Compose` so out-of-frame boxes (and their
    labels) drop automatically after the pipeline runs.

    Parameters
    ----------
    max_size : int, optional, default=1333
        Longest-side target after resize; canonical detection
        recipes use ``800 / 1333``.  ``PadIfNeeded`` then squares the
        canvas to ``(max_size, max_size)`` so batching works.
    min_area : float, optional, default=1.0
        Drop post-pipeline boxes whose absolute pixel area is below
        this — see :class:`BboxParams`.
    min_visibility : float, optional, default=0.0
        Drop boxes whose visible fraction (area / original area)
        falls below this.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"
    """

    preset_type: ClassVar[str] = "Detection"

    def __init__(
        self,
        *,
        max_size: int = 1333,
        min_area: float = 1.0,
        min_visibility: float = 0.0,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        self.max_size = max_size
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        self._pipeline = Compose(
            [
                LongestMaxSize(max_size, interpolation=interpolation),
                PadIfNeeded(max_size, max_size, value=0.0),
                Normalize(self.mean, self.std, max_pixel_value=1.0),
            ],
            bbox_params=BboxParams(min_area=min_area, min_visibility=min_visibility),
        )

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "max_size": self.max_size,
            "min_area": self.min_area,
            "min_visibility": self.min_visibility,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (
                interp.value if isinstance(interp, Interpolation) else interp
            ),
        }


@_register_preset
class Segmentation(TransformsPreset):
    r"""Semantic-segmentation preset — image + mask share geometry.

    Pipeline: ``SmallestMaxSize(resize_size)`` → ``CenterCrop(crop_size)``
    → ``Normalize`` applied to the image only.  Mask travels through
    the geometric stages with nearest-neighbour interpolation (label
    preservation guaranteed by every
    :class:`~lucid.utils.transforms._base.GeometricTransform`'s
    ``_apply_mask`` hook) and is *not* normalised.

    Parameters
    ----------
    crop_size : int
        Square crop fed to the model.
    resize_size : int, optional, default=520
        Shorter-side length before cropping; canonical for FCN /
        DeepLab style recipes.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"
        Image resize interpolation.  Masks always use nearest.
    """

    preset_type: ClassVar[str] = "Segmentation"

    def __init__(
        self,
        crop_size: int,
        *,
        resize_size: int = 520,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        self._pipeline = Compose(
            [
                SmallestMaxSize(resize_size, interpolation=interpolation),
                CenterCrop(crop_size, crop_size),
                Normalize(self.mean, self.std, max_pixel_value=1.0),
            ]
        )

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (
                interp.value if isinstance(interp, Interpolation) else interp
            ),
        }


@_register_preset
class Pose(TransformsPreset):
    r"""Keypoint / pose-estimation preset.

    Pipeline: ``SmallestMaxSize(resize_size)`` → ``CenterCrop(crop_size)``
    → ``Normalize``.  Keypoint coordinates ride along through every
    geometric stage's ``_apply_keypoints`` hook; extra columns
    (visibility / score / angle) pass through unchanged.

    Parameters
    ----------
    crop_size : int
        Square crop fed to the model.
    resize_size : int, optional, default=256
        Shorter-side length before cropping.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"
    """

    preset_type: ClassVar[str] = "Pose"

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
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        self._pipeline = Compose(
            [
                SmallestMaxSize(resize_size, interpolation=interpolation),
                CenterCrop(crop_size, crop_size),
                Normalize(self.mean, self.std, max_pixel_value=1.0),
            ]
        )

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (
                interp.value if isinstance(interp, Interpolation) else interp
            ),
        }
