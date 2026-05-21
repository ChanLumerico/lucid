"""Structural-typing protocols for the model-zoo family contract.

This module is a *static-typing companion* to the nominal hierarchy in
:mod:`lucid.models._base`.  The base classes (``ModelConfig`` /
``PretrainedModel``) own the canonical implementation; the protocols
here describe the **shape** that any class playing a family role must
expose.  Three motivations:

1. **Static checking.**  Type checkers (mypy, pyright, IDE language
   servers) can verify that a Config / Backbone / Task-wrapper class
   *structurally* satisfies the contract — independent of where it sits
   in the inheritance graph.
2. **Duck-typed extension.**  Third-party packages may define their
   own model families without inheriting from ``ModelConfig`` /
   ``PretrainedModel`` (e.g. adapter layers, vendor SDKs).  As long as
   the class has the right attributes, the framework treats it as a
   first-class family.
3. **Runtime structural checks.**  Protocols here are marked
   ``@runtime_checkable`` so the static validator and other
   introspection tools can use ``isinstance(cls, ModelConfigProtocol)``
   instead of a hard-coded ``issubclass(cls, ModelConfig)`` check.

These protocols are **advisory** — they document the contract that the
nominal hierarchy enforces.  They do not replace ``ModelConfig`` or
``PretrainedModel``.  See ``arch-models-family-contract`` for the full
spec.
"""

from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

# ---------------------------------------------------------------------------
# Building-block protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class HasModelType(Protocol):
    r"""Building-block protocol: class carries a ``model_type`` ClassVar.

    The simplest of the family-contract pieces.  Every concrete family
    Config (``ResNetConfig``, ``BertConfig``, …) overrides this ClassVar
    to a unique short identifier (``"resnet"``, ``"bert"``) used by the
    registry, ``config.json`` persistence, and ``AutoConfig`` dispatch.

    Intermediate abstract bases — ``ModelConfig`` itself,
    ``LanguageModelConfig``, ``DiffusionModelConfig``,
    ``GenerativeModelConfig`` — deliberately keep ``model_type ==
    "base"``; they are not family Configs and won't satisfy stricter
    protocols (e.g. :class:`ModelConfigProtocol`) that compose this
    one.

    Attributes
    ----------
    model_type : ClassVar[str]
        Persistent family identifier — embedded in checkpoint
        ``config.json`` and used by directory-based loading to dispatch
        to the correct registry entry.

    Notes
    -----
    Component of the composite :class:`ModelConfigProtocol`.  Use this
    protocol alone in signatures that only need the family identifier
    (e.g. logging utilities, registry lookups) without also requiring
    the full ``@model_family_meta`` machinery.

    Examples
    --------
    Every documented family Config carries the attribute:

    >>> from lucid.models.vision.resnet import ResNetConfig
    >>> from lucid.models._protocols import HasModelType
    >>> isinstance(ResNetConfig, HasModelType)
    True

    A plain class without the ClassVar does not satisfy it:

    >>> class Plain: pass
    >>> isinstance(Plain, HasModelType)
    False
    """

    model_type: ClassVar[str]


@runtime_checkable
class HasFamilyMeta(Protocol):
    r"""Building-block protocol: class has been decorated with
    ``@model_family_meta`` and so carries the docs-facing metadata
    triple at ``__model_family_meta__``.

    The decorator attaches a :class:`ModelFamilyMeta` instance holding
    ``canonical_name`` / ``citation`` / ``theory`` strings.  The docs
    site reads these to render family-root index cards, sidebar
    labels, and paper headers.  The Lucid contract requires every
    concrete family Config to be wrapped.

    Attributes
    ----------
    __model_family_meta__ : ClassVar[object]
        Frozen :class:`lucid.models._meta.ModelFamilyMeta` instance set
        by ``@model_family_meta``.  The annotation is ``object`` here
        because :class:`Protocol` cannot import its concrete dataclass
        without creating a circular dependency; use
        ``ModelFamilyMeta`` in real signatures that need the typed
        fields.

    Notes
    -----
    Component of :class:`ModelConfigProtocol`.  Adding this protocol
    to a function signature signals "must be a docs-discoverable
    family Config" without committing to the rest of the contract
    (registry id, dataclass-ness).

    Examples
    --------
    >>> from lucid.models.vision.vit import ViTConfig
    >>> from lucid.models._protocols import HasFamilyMeta
    >>> isinstance(ViTConfig, HasFamilyMeta)
    True

    A bare class without the decorator fails the check:

    >>> class Bare: model_type = "bare"
    >>> isinstance(Bare, HasFamilyMeta)
    False
    """

    __model_family_meta__: ClassVar[object]


@runtime_checkable
class HasConfigClass(Protocol):
    r"""Building-block protocol: model class points at its family's
    Config via the ``config_class`` ClassVar.

    Every concrete :class:`PretrainedModel` subclass — backbone or
    task wrapper — declares ``config_class: ClassVar[type[...]] =
    <FamilyConfig>``.  The base ``PretrainedModel.__init__`` uses the
    attribute to validate the runtime config argument and
    :meth:`from_pretrained` uses it to reconstruct the config from
    JSON on disk.

    Attributes
    ----------
    config_class : ClassVar[type]
        Pointer to the family's :class:`ModelConfig` subclass.  Set
        on every concrete model class; left as ``None`` on the
        abstract :class:`PretrainedModel` base so a missing override
        raises a clear ``TypeError`` at first instantiation.

    Notes
    -----
    Component of :class:`PretrainedModelProtocol`.  Refine to
    :class:`BackboneProtocol` or :class:`TaskWrapperProtocol` in
    signatures that need finer distinctions.

    Examples
    --------
    >>> from lucid.models.vision.resnet import ResNet
    >>> from lucid.models._protocols import HasConfigClass
    >>> isinstance(ResNet, HasConfigClass)
    True
    >>> ResNet.config_class.__name__
    'ResNetConfig'
    """

    config_class: ClassVar[type]


# ---------------------------------------------------------------------------
# Composite role protocols — what *the docs site* and *the framework* expect
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelConfigProtocol(HasModelType, HasFamilyMeta, Protocol):
    r"""Composite structural contract for a family Config class.

    A class satisfies this protocol when it simultaneously
    1. declares a non-empty ``model_type`` ClassVar
       (:class:`HasModelType`),
    2. has been decorated with ``@model_family_meta`` so that
       ``__model_family_meta__`` is set (:class:`HasFamilyMeta`),
    3. is a ``@dataclass`` — the ``__dataclass_fields__`` ClassVar
       below is the witness that the dataclass transform ran.

    Use this protocol in any function that accepts arbitrary family
    Configs without coupling to the nominal :class:`ModelConfig`
    base — third-party model packages may satisfy it via duck
    typing.

    Attributes
    ----------
    model_type : ClassVar[str]
        Inherited from :class:`HasModelType` — see that protocol.
    __model_family_meta__ : ClassVar[object]
        Inherited from :class:`HasFamilyMeta` — see that protocol.
    __dataclass_fields__ : ClassVar[dict[str, Any]]
        Standard attribute set by :func:`dataclasses.dataclass`;
        used here as the runtime witness that the class went through
        the dataclass transform.

    Notes
    -----
    The protocol is :func:`runtime_checkable`, so the static
    validator (``tools/validate_model_zoo.py --runtime``) can use
    ``isinstance(cls, ModelConfigProtocol)`` to catch refactor drift
    (e.g. a decorator that silently strips
    ``__model_family_meta__``).  See ``arch-models-family-contract``
    for the full 5-slot family layout.

    Examples
    --------
    Every documented family Config satisfies the contract:

    >>> from lucid.models.vision.resnet import ResNetConfig
    >>> from lucid.models.text.bert import BertConfig
    >>> from lucid.models._protocols import ModelConfigProtocol
    >>> all(isinstance(c, ModelConfigProtocol) for c in (ResNetConfig, BertConfig))
    True

    A duck-typed third-party Config with the right shape also
    qualifies — no need to inherit from :class:`ModelConfig`:

    >>> from typing import ClassVar
    >>> class Stranger:
    ...     model_type: ClassVar[str] = "stranger"
    ...     __model_family_meta__: ClassVar[object] = object()
    ...     __dataclass_fields__: ClassVar[dict] = {}
    >>> isinstance(Stranger, ModelConfigProtocol)
    True
    """

    __dataclass_fields__: ClassVar[dict[str, Any]]


@runtime_checkable
class PretrainedModelProtocol(HasConfigClass, Protocol):
    r"""Composite structural contract for a family's *model* class —
    either the backbone (``ResNet``, ``BertModel``) or a task wrapper
    (``ResNetForImageClassification``, ``BertForMaskedLM``).

    The protocol requires (i) the ``config_class`` pointer (from
    :class:`HasConfigClass`) and (ii) the two methods every
    framework-consumed model exposes: ``__init__(self, config)`` and
    ``forward(...)``.  Concrete framework subclasses additionally
    inherit :class:`lucid.nn.Module` — so ``parameters()`` /
    ``state_dict()`` / ``train()`` / ``eval()`` come for free — but
    those are intentionally *not* part of the structural contract:
    the protocol is meant to be checkable against the **class object
    alone**, before instantiation, so the static validator can use
    it.

    Attributes
    ----------
    config_class : ClassVar[type]
        Inherited from :class:`HasConfigClass`.

    Notes
    -----
    Refinements: :class:`BackboneProtocol` for direct-model classes
    (no ``For<Task>`` in name); :class:`TaskWrapperProtocol` for
    ``*For<Task>`` heads.  Use the refined protocol in signatures
    that need to distinguish — e.g. a feature extractor utility
    should take a :class:`BackboneProtocol`, while a fine-tuning
    helper might take a :class:`TaskWrapperProtocol`.

    Examples
    --------
    >>> from lucid.models.vision.resnet import (
    ...     ResNet, ResNetForImageClassification,
    ... )
    >>> from lucid.models._protocols import PretrainedModelProtocol
    >>> isinstance(ResNet, PretrainedModelProtocol)
    True
    >>> isinstance(ResNetForImageClassification, PretrainedModelProtocol)
    True
    """

    def __init__(self, config: Any) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class BackboneProtocol(PretrainedModelProtocol, Protocol):
    r"""Refinement of :class:`PretrainedModelProtocol` for the family's
    *direct* (backbone) model — the class whose name does **not**
    contain ``For<Task>`` (e.g. ``ResNet``, ``BertModel``, ``ViT``,
    ``DDPMModel``).

    Currently identical in shape to the base protocol; the dedicated
    type exists so static checkers can distinguish backbones from
    task wrappers at call sites that demand one or the other.  A
    feature-extraction utility, for example, should require a
    backbone — not an ``ImageClassification`` head — to avoid
    accidentally pulling logits.

    Notes
    -----
    Naming rule: any public class in ``_model.py`` whose name lacks
    the ``For<CapWord>`` pattern is treated as the backbone.  See
    ``arch-models-family-contract`` slot 2.

    Examples
    --------
    >>> from lucid.models.vision.resnet import ResNet
    >>> from lucid.models._protocols import BackboneProtocol, TaskWrapperProtocol
    >>> isinstance(ResNet, BackboneProtocol)
    True
    >>> isinstance(ResNet, TaskWrapperProtocol)
    True  # all backbones structurally satisfy the wrapper protocol too;
          # nominal naming is what distinguishes them
    """


@runtime_checkable
class TaskWrapperProtocol(PretrainedModelProtocol, Protocol):
    r"""Refinement of :class:`PretrainedModelProtocol` for ``*For<Task>``
    head classes (e.g. ``ResNetForImageClassification``,
    ``BertForMaskedLM``).

    Implementations are expected to return a task-specific ``*Output``
    dataclass from :meth:`forward` — :class:`ImageClassificationOutput`
    for classifiers, :class:`MaskedLMOutput` for MLM heads, etc.  The
    output type is not encoded in the protocol because Python's
    structural typing cannot constrain method *return* shapes
    without a Generic parameter, which would complicate the
    runtime-checkable surface.

    Notes
    -----
    Naming rule: the class name ends with ``For<CapWord>`` matching
    a key in ``_TASK_SUFFIX_MAP`` (see build script).  The validator
    flags ``For<Unknown>`` suffixes with MZ050 so new task types
    must be explicitly registered.

    Examples
    --------
    >>> from lucid.models.vision.resnet import ResNetForImageClassification
    >>> from lucid.models._protocols import TaskWrapperProtocol
    >>> isinstance(ResNetForImageClassification, TaskWrapperProtocol)
    True
    """


@runtime_checkable
class OutputDataclassProtocol(Protocol):
    r"""Forward-return dataclass shape — any dataclass whose name ends
    with ``Output`` (e.g. :class:`ImageClassificationOutput`,
    :class:`MaskedLMOutput`).  Carries at least an ``logits`` /
    family-specific tensor field plus an optional ``loss``.

    The exact field set differs across tasks (logits / boxes / masks
    / samples / hidden_states / past_key_values / …), so the protocol
    intentionally requires only the structural ``__dataclass_fields__``
    witness — no concrete field constraint.

    Attributes
    ----------
    __dataclass_fields__ : ClassVar[dict[str, Any]]
        Standard attribute set by :func:`dataclasses.dataclass`.

    Notes
    -----
    Output dataclasses are hidden from the docs sidebar
    (``_model.py`` 4-slot rule, slot 4) because they are *return
    types* rather than user-facing entry points.  The protocol is
    still useful for return-type hints in framework utilities (e.g.
    metrics computation, loss aggregation) that accept any output
    shape.

    Examples
    --------
    >>> from lucid.models import ImageClassificationOutput
    >>> from lucid.models._protocols import OutputDataclassProtocol
    >>> isinstance(ImageClassificationOutput, OutputDataclassProtocol)
    True
    """

    __dataclass_fields__: ClassVar[dict[str, Any]]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


ConfigT = TypeVar("ConfigT", bound=ModelConfigProtocol)
r"""Type variable for family-Config-parameterised generics.

Use in signatures that want to *track* which Config a function or
helper is bound to, without paying the cost of making
:class:`PretrainedModel` itself :class:`typing.Generic` (Python
forbids parameterising :class:`ClassVar` by a :class:`TypeVar`, which
blocks the obvious ``config_class: ClassVar[type[ConfigT]]``
declaration on the base).

The variable is bound to :class:`ModelConfigProtocol` so structural
duck-typed Configs are accepted alongside nominal subclasses.

Examples
--------
A small loader that round-trips any family Config preserves the
caller's concrete type:

>>> from pathlib import Path
>>> from lucid.models._protocols import ConfigT
>>> def load_config(cls: type[ConfigT], path: str) -> ConfigT:
...     return cls.load(path)
"""
