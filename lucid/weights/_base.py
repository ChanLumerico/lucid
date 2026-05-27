"""Core types for the :mod:`lucid.weights` pretrained-weight system.

This module defines the two primitives every per-model weight
declaration is built from:

* :class:`WeightEntry` — an immutable manifest for one concrete
  checkpoint: where to download it (``url`` + ``sha256``), how many
  output classes it has, the preprocessing it expects, and a free-form
  ``meta`` dict carrying provenance + benchmark numbers.
* :class:`WeightsEnum` — an :class:`enum.Enum` subclass whose members
  *are* :class:`WeightEntry` values.  Each architecture declares one
  ``WeightsEnum`` (e.g. ``ResNet18Weights``) listing its tagged
  variants (``IMAGENET1K_V1``, ``IMAGENET1K_V2``, …) plus a ``DEFAULT``
  alias.  This mirrors the torchvision ``Weights`` enum convention so
  the call-site ergonomics are familiar.

Deliberately, :class:`WeightEntry` does **not** carry the Lucid model
:class:`~lucid.models._base.ModelConfig`.  Named factories
(``resnet_18`` …) already pin their own config, so embedding it here
would only create a circular dependency between :mod:`lucid.weights`
and :mod:`lucid.models`.  The architecture config lives on the Hub in
each checkpoint's ``config.json`` for discoverability instead.
"""

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.weights._transforms import Transform


@dataclass(frozen=True)
class WeightEntry:
    r"""Immutable manifest describing one pretrained checkpoint.

    A :class:`WeightEntry` is the value held by each
    :class:`WeightsEnum` member.  It is everything the runtime needs to
    fetch, verify, and load a checkpoint — but nothing about the model
    architecture itself (that is fixed by the factory that consumes the
    entry).

    Parameters
    ----------
    url : str
        Direct download URL of the ``model.safetensors`` blob.  Lucid
        hosts these on the Hugging Face Hub under the ``lucid-dl`` org
        (``.../resolve/main/<TAG>/model.safetensors``) but no
        provider-specific knowledge is encoded here — any HTTPS URL
        works.
    sha256 : str
        Hex-encoded SHA-256 digest of the file at ``url``.  Verified
        after download and on every cache hit; a mismatch forces a
        re-download (the cached copy is presumed corrupt).
    num_classes : int
        Number of output classes the checkpoint's head produces (e.g.
        ``1000`` for ImageNet-1k).  Lets callers sanity-check the entry
        against a model's configured ``num_classes`` before loading.
    transforms : Transform
        Preprocessing pipeline the weights were trained with — applied
        to inputs at inference time so ``resnet_18(pretrained=True)``
        "just works".  See :mod:`lucid.weights._transforms`.
    meta : dict
        Free-form provenance + metrics.  Conventional keys: ``source``
        (e.g. ``"torchvision/ResNet18_Weights.IMAGENET1K_V1"``),
        ``license``, ``recipe``, ``metrics`` (nested
        ``{dataset: {metric: value}}``), ``num_params``, ``gflops``,
        ``file_size_mb``.  Rendered into the Hub ``config.json`` +
        model card by the conversion tool.

    Notes
    -----
    Frozen (``frozen=True``) so entries can be shared by reference and
    used as :class:`enum.Enum` values.  Two members carrying the *same*
    ``WeightEntry`` instance collapse into an enum alias — this is how
    ``DEFAULT = IMAGENET1K_V1`` works (see :class:`WeightsEnum`).
    """

    url: str
    sha256: str
    num_classes: int
    transforms: "Transform"
    meta: dict[str, object] = field(default_factory=dict)


class WeightsEnum(enum.Enum):
    r"""Base class for per-architecture pretrained-weight enums.

    Each architecture subclasses :class:`WeightsEnum` and lists its
    tagged checkpoints as members whose values are :class:`WeightEntry`
    instances::

        class ResNet18Weights(WeightsEnum):
            IMAGENET1K_V1 = WeightEntry(url=..., sha256=..., ...)
            DEFAULT = IMAGENET1K_V1

    Members expose the underlying entry's fields directly
    (``ResNet18Weights.IMAGENET1K_V1.url``) plus the tag name
    (``.tag``), so call sites read naturally.  Assigning
    ``DEFAULT = IMAGENET1K_V1`` makes ``DEFAULT`` an *alias* of the
    canonical member (standard :mod:`enum` behaviour for duplicate
    values), exactly mirroring the torchvision convention where
    ``DEFAULT`` tracks the strongest available weights.

    Notes
    -----
    The :class:`enum.Enum` value of each member is a frozen
    :class:`WeightEntry`.  Accessing ``.value`` returns that entry; the
    convenience properties below forward to it.
    """

    @property
    def entry(self) -> WeightEntry:
        """The underlying :class:`WeightEntry` value."""
        value = self.value
        if not isinstance(value, WeightEntry):
            raise TypeError(
                f"{type(self).__name__}.{self.name} value is "
                f"{type(value).__name__}, expected WeightEntry"
            )
        return value

    @property
    def tag(self) -> str:
        """The variant tag (member name), e.g. ``"IMAGENET1K_V1"``.

        For the ``DEFAULT`` alias this resolves to the canonical
        member's name (the one it aliases), not the literal string
        ``"DEFAULT"``.
        """
        return self.name

    @property
    def url(self) -> str:
        """Download URL of this checkpoint (see :attr:`WeightEntry.url`)."""
        return self.entry.url

    @property
    def sha256(self) -> str:
        """Expected SHA-256 of this checkpoint."""
        return self.entry.sha256

    @property
    def num_classes(self) -> int:
        """Output-class count of this checkpoint's head."""
        return self.entry.num_classes

    @property
    def meta(self) -> dict[str, object]:
        """Provenance + metrics dict (see :attr:`WeightEntry.meta`)."""
        return self.entry.meta

    def transforms(self) -> "Transform":
        """Return the preprocessing pipeline the weights expect.

        Returns
        -------
        Transform
            Callable transform; apply to a :class:`lucid.Tensor` image
            before feeding the model.

        Examples
        --------
        >>> weights = ResNet18Weights.IMAGENET1K_V1
        >>> preprocess = weights.transforms()
        >>> x = preprocess(image)        # image: lucid.Tensor (C, H, W)
        >>> logits = model(x[None])
        """
        return self.entry.transforms
