"""Decorator-based metadata for model-family Config classes.

Each family's ``ModelConfig`` subclass is wrapped with
``@model_family_meta(canonical_name=..., citation=..., theory=...)`` to attach
the three docs-facing fields used by the API docs site:

- ``canonical_name`` — paper-style display name (``"ResNet"``, ``"BERT"``)
- ``citation``       — MLA-formatted citation of the defining paper
- ``theory``         — rST-formatted theoretical description (math OK)

The decorator stores the metadata on the class as ``__model_family_meta__``
for runtime introspection.  The docs build script parses the decorator
statically from each family's ``_config.py`` source — it does not rely on
importing the class — so the metadata is the single source of truth for
both runtime and docs.

Convention rationale: see ``arch-models-canonical-name``.
"""

from dataclasses import dataclass
from typing import Callable, TypeVar

from lucid.models._base import ModelConfig

_C = TypeVar("_C", bound=type[ModelConfig])


@dataclass(frozen=True)
class ModelFamilyMeta:
    r"""Immutable container for a family's docs-facing metadata.

    Attributes
    ----------
    canonical_name : str
        Paper-style family display name (e.g. ``"ResNet"``,
        ``"ConvNeXt"``, ``"BERT"``).
    citation : str
        MLA-formatted citation of the family's defining paper.
    theory : str
        rST-formatted theoretical description.  May include
        ``:math:`...`` inline math and ``.. math::`` blocks; rendered
        through the same pipeline as docstring extended-description
        sections on the API docs site.
    """

    canonical_name: str
    citation: str
    theory: str


def model_family_meta(
    *,
    canonical_name: str,
    citation: str,
    theory: str,
) -> Callable[[_C], _C]:
    r"""Class decorator: attach paper / theory / display metadata to a
    ``ModelConfig`` subclass.

    The decorator stores a :class:`ModelFamilyMeta` instance on the
    decorated class as the ``__model_family_meta__`` attribute, so
    runtime code may introspect family metadata without parsing source.

    The docs site does **not** rely on the runtime attribute — instead
    it parses the decorator call statically from each family's
    ``_config.py`` file (Python ``ast`` module).  This means:

    - Adding a new family is purely a source-level operation; nothing
      needs to be re-imported or re-registered at runtime.
    - The decorator arguments **must be string literals** (or implicit
      concatenations of literals).  f-strings, expressions, and
      references to module-level constants are not recognised by the
      static parser.

    Parameters
    ----------
    canonical_name : str, keyword-only
        Paper-style family display name (``"ResNet"``, ``"BERT"``).
    citation : str, keyword-only
        MLA-formatted citation of the defining paper.  Use ``\"...\"``
        around the title to embed double-quotes.
    theory : str, keyword-only
        rST-formatted theoretical description.  Prefer raw triple-quoted
        strings (``r\"\"\"...\"\"\"``) so LaTeX escapes survive verbatim.

    Returns
    -------
    Callable[[_C], _C]
        The decorator function.  The returned class is *the same class
        object*, with ``__model_family_meta__`` set.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from lucid.models._base import ModelConfig
    >>> from lucid.models._meta import model_family_meta
    >>> @model_family_meta(
    ...     canonical_name="ResNet",
    ...     citation=(
    ...         "He, Kaiming, et al. \"Deep Residual Learning for Image "
    ...         "Recognition.\" CVPR, 2016, pp. 770-778."
    ...     ),
    ...     theory=r'''
    ...     Deep residual network — adds an identity shortcut around
    ...     every convolution stack:
    ...
    ...     .. math::
    ...
    ...         y = F(x, \{W_i\}) + x
    ...     ''',
    ... )
    ... @dataclass(frozen=True)
    ... class ResNetConfig(ModelConfig):
    ...     model_type: ClassVar[str] = "resnet"
    """

    meta = ModelFamilyMeta(
        canonical_name=canonical_name,
        citation=citation,
        theory=theory,
    )

    def _wrap(cls: _C) -> _C:
        cls.__model_family_meta__ = meta  # type: ignore[attr-defined]
        return cls

    return _wrap
