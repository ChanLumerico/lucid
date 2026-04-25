import importlib
import pytest

_PUBLIC_MODULES = (
    "lucid",
    "lucid.types",
    "lucid.error",
    "lucid._tensor",
    "lucid._utils",
    "lucid._backend",
    "lucid._backend.metal",
    "lucid._func",
    "lucid.autograd",
    "lucid.random",
    "lucid.linalg",
    "lucid.nn",
    "lucid.optim",
    "lucid.models",
    "lucid.transforms",
    "lucid.visual",
    "lucid.weights",
    "lucid.data",
    "lucid.datasets",
    "lucid.einops",
)


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES, ids=lambda m: m)
def test_public_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
