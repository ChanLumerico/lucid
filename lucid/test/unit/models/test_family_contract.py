"""Automatic protocol-conformance tests for the Lucid model zoo.

Walks every family directory under ``lucid/models/<domain>/`` and asserts
that the exported Config class structurally satisfies
``ModelConfigProtocol`` and the public model classes satisfy
``PretrainedModelProtocol``.  Runs on every ``pytest`` invocation — so
any future family that drops a slot is caught the moment its tests run,
without anyone having to remember to invoke ``tools/validate_model_zoo``.

Companion to the static AST validator at ``tools/validate_model_zoo.py``:
the validator catches *file-level* / *decorator-absence* mistakes,
these tests catch *runtime structural* drift (e.g. a refactor that
silently strips ``__model_family_meta__``).

Contract spec: ``obsidian/architecture/arch-models-family-contract.md``
"""

import importlib
import inspect
import re
from pathlib import Path
from types import ModuleType

import pytest

from lucid.models._protocols import (
    ModelConfigProtocol,
    PretrainedModelProtocol,
)
from lucid.models._registry import is_model

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = REPO_ROOT / "lucid" / "models"
DOMAINS = ("vision", "text", "generative")

# Abstract intermediate Config bases — these legitimately keep
# ``model_type == "base"`` and are not themselves family configs.
_INTERMEDIATE_CONFIG_NAMES = {
    "ModelConfig",
    "LanguageModelConfig",
    "DiffusionModelConfig",
    "GenerativeModelConfig",
}


def _discover_families() -> list[tuple[str, str]]:
    families: list[tuple[str, str]] = []
    for domain in DOMAINS:
        dom_dir = MODELS_DIR / domain
        if not dom_dir.is_dir():
            continue
        for fam in sorted(dom_dir.iterdir()):
            if not fam.is_dir() or fam.name.startswith("_"):
                continue
            families.append((domain, fam.name))
    return families


_FAMILIES = _discover_families()


def _import_family(domain: str, family: str):
    mod_path = f"lucid.models.{domain}.{family}"
    return importlib.import_module(mod_path)


@pytest.mark.parametrize(
    "domain,family", _FAMILIES, ids=[f"{d}/{f}" for d, f in _FAMILIES]
)
def test_family_config_satisfies_protocol(domain: str, family: str) -> None:
    """Every family must export a ``<Family>Config`` class that
    structurally satisfies :class:`ModelConfigProtocol` (model_type
    ClassVar set, ``@model_family_meta`` applied, and is a dataclass)."""
    mod = _import_family(domain, family)
    config_classes = [
        obj
        for name in dir(mod)
        if isinstance(obj := getattr(mod, name), type)
        and name.endswith("Config")
        and name not in _INTERMEDIATE_CONFIG_NAMES
    ]
    assert config_classes, (
        f"family {domain}/{family}: no <Family>Config class exported "
        f"from __init__.py"
    )
    for cfg in config_classes:
        assert isinstance(cfg, ModelConfigProtocol), (
            f"{cfg.__name__} ({domain}/{family}): does not satisfy "
            f"ModelConfigProtocol.  Required attributes: "
            f"model_type (ClassVar[str]), __model_family_meta__ "
            f"(set by @model_family_meta), __dataclass_fields__ "
            f"(from @dataclass).  See arch-models-family-contract.md."
        )
        # Sanity: not the abstract default.
        assert cfg.model_type != "base", (
            f"{cfg.__name__} ({domain}/{family}): model_type is still "
            f"'base' — must override with a unique family identifier."
        )


@pytest.mark.parametrize(
    "domain,family", _FAMILIES, ids=[f"{d}/{f}" for d, f in _FAMILIES]
)
def test_family_models_satisfy_protocol(domain: str, family: str) -> None:
    """Every public model class in a family (backbone or task wrapper)
    must satisfy :class:`PretrainedModelProtocol` — i.e. declare
    ``config_class``, ``__init__(self, config)`` and ``forward(...)``.
    """
    mod = _import_family(domain, family)
    model_classes = []
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if not isinstance(obj, type):
            continue
        # Skip Configs, Output dataclasses, and protocols themselves.
        if name.endswith("Config") or name.endswith("Output"):
            continue
        # Heuristic: only classes whose home module is in this family.
        home = getattr(obj, "__module__", "")
        if not home.startswith(f"lucid.models.{domain}.{family}"):
            continue
        # The protocol governs the 5-slot family-canonical classes that
        # take a Config object.  Internal backbone building blocks (e.g.
        # ``DDPMUNet``, ``ViTEmbeddings``) are legitimately exported as
        # composition primitives and don't carry the protocol surface;
        # skip anything that lacks ``config_class`` (which the protocol
        # itself requires).  This avoids the prior false-positive on
        # ``DDPMUNet`` while still pinning the actual model classes.
        if not hasattr(obj, "config_class"):
            continue
        model_classes.append(obj)

    if not model_classes:
        pytest.skip(
            f"family {domain}/{family}: no public model class exported "
            f"— nothing to check (legitimate for re-export-only __init__)."
        )
    for cls in model_classes:
        assert isinstance(cls, PretrainedModelProtocol), (
            f"{cls.__name__} ({domain}/{family}): does not satisfy "
            f"PretrainedModelProtocol.  Required attributes: "
            f"config_class (ClassVar[type]), __init__(self, config), "
            f"forward(...).  See arch-models-family-contract.md."
        )


def test_family_count_matches_directory_scan() -> None:
    """Guards against discovery silently dropping families — e.g. a
    family that doesn't expose anything in ``__init__.py`` would still
    be a directory but invisible to the parametrised tests above."""
    actual = sum(
        1
        for domain in DOMAINS
        if (MODELS_DIR / domain).is_dir()
        for fam in (MODELS_DIR / domain).iterdir()
        if fam.is_dir() and not fam.name.startswith("_")
    )
    assert len(_FAMILIES) == actual, (
        f"family discovery saw {len(_FAMILIES)}, raw directory scan "
        f"sees {actual} — investigate."
    )


def _misspelt_versions(canonical_name: str) -> list[str]:
    """Version tokens in a display name that are not written ``-vN``.

    The zoo writes a family's version as a lowercase ``v`` joined by a
    hyphen — ``Dreamer-v3``, ``MobileNet-v2``, ``PVT-v2`` — so the
    sidebar reads consistently regardless of how each paper styled its
    own title (``DreamerV3``, ``MobileNet V2``, ``PVTv2`` are all in the
    literature).

    Only ``v``-prefixed tokens count.  A bare trailing numeral is part of
    the published name rather than a version suffix — ``GPT-2`` is what
    the model is called, and ``Mask2Former`` is not versioned at all.
    """
    bad: list[str] = []
    for match in re.finditer(r"[Vv]\s*\d+", canonical_name):
        token = match.group()
        preceded_by_hyphen = (
            match.start() > 0 and canonical_name[match.start() - 1] == "-"
        )
        if not (token.startswith("v") and preceded_by_hyphen):
            bad.append(token)
    return bad


@pytest.mark.parametrize(
    "domain,family", _FAMILIES, ids=[f"{d}/{f}" for d, f in _FAMILIES]
)
def test_family_version_suffix_is_hyphen_v(domain: str, family: str) -> None:
    """Versioned families display as ``<Name>-vN`` on the docs site."""
    module = _import_family(domain, family)
    config = next(
        (getattr(module, n) for n in module.__all__ if n.endswith("Config")), None
    )
    assert config is not None, f"{domain}/{family} exports no Config"
    meta = getattr(config, "__model_family_meta__", None)
    canonical = getattr(meta, "canonical_name", None) or ""
    bad = _misspelt_versions(canonical)
    assert not bad, (
        f"{domain}/{family} canonical_name {canonical!r} writes its version "
        f"as {bad} — the zoo's convention is '-vN' (lowercase v, hyphen "
        f"joined), e.g. 'Dreamer-v3'."
    )


def test_the_version_check_can_fail() -> None:
    """Guards the test above — every family already conforms, so a check
    that matched nothing would look identical to one that works."""
    assert _misspelt_versions("MobileNet V2") == ["V2"]
    assert _misspelt_versions("DreamerV3") == ["V3"]
    assert _misspelt_versions("PVTv2") == ["v2"]  # right letter, no hyphen
    assert _misspelt_versions("Dreamer-v3") == []
    assert _misspelt_versions("GPT-2") == []  # not a version suffix
    assert _misspelt_versions("Mask2Former") == []


def _stray_exported_functions(module: ModuleType) -> list[str]:
    """Names in ``module.__all__`` that are functions with no business
    being public.

    A family's public functions are its **model factories**, plus the
    occasional **config builder** (EfficientDet exposes one, keyed by the
    compound-scaling coefficient).  Anything else — a training objective,
    a dispatch helper — is an implementation detail of the family.

    Kept as a helper rather than inlined so the test below can be pointed
    at a module built to fail.
    """
    stray: list[str] = []
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name, None)
        if not inspect.isfunction(obj):
            continue
        if is_model(name):
            continue
        if getattr(obj, "__module__", "").endswith("._config"):
            continue
        stray.append(name)
    return stray


@pytest.mark.parametrize(
    "domain,family", _FAMILIES, ids=[f"{d}/{f}" for d, f in _FAMILIES]
)
def test_family_exports_no_internal_helpers(domain: str, family: str) -> None:
    """A family's ``__all__`` must not leak its own machinery.

    This is a docs bug before it is an API bug.  The site renders a
    family leaf's module-level functions as its factory list, so an
    exported helper appears among the pretrained entries as though you
    could load weights with it — which is how ``free_bits_kl`` was
    spotted sitting next to the twelve DreamerV3 factories.

    Classes are exempt: they render in their own section, so a genuinely
    public building block (``DDPMUNet``, the SDE hierarchy) is fine
    there.
    """
    module = _import_family(domain, family)
    stray = _stray_exported_functions(module)
    assert not stray, (
        f"lucid.models.{domain}.{family} exports {stray} — not registered "
        f"factories and not config builders.  Drop them from __init__.py "
        f"(callers inside the family should import from the private "
        f"module directly), or register them if they really are factories."
    )


def test_the_export_check_can_fail() -> None:
    """Guards the test above — with every family already clean, a check
    that silently matched nothing would pass just as convincingly."""
    fake = ModuleType("lucid.models.generative.fake")

    def free_bits_kl() -> None: ...

    free_bits_kl.__module__ = "lucid.models.generative.fake._objectives"
    fake.free_bits_kl = free_bits_kl  # type: ignore[attr-defined]
    fake.__all__ = ["free_bits_kl"]  # type: ignore[attr-defined]

    assert _stray_exported_functions(fake) == ["free_bits_kl"]
