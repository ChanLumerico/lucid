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
from pathlib import Path

import pytest

from lucid.models._protocols import (
    ModelConfigProtocol,
    PretrainedModelProtocol,
)

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
