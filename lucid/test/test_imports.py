from .core import ModuleImportBase


class TestPublicModulesImport(ModuleImportBase):
    modules = (
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
