"""Per-directory pytest config for ``lucid/test/unit/models``.

Skips collection of tests that depend on model families which are
declared in the test tree but not yet implemented in ``lucid.models``.
The tests stay in the repo so they're picked up automatically once the
corresponding model module lands.
"""

import importlib

# (test file → list of model modules it imports at the top)
_PENDING_MODULES = {
    "test_models_inception.py": (
        "lucid.models.vision.inception_v4",
        "lucid.models.vision.inception_resnet",
    ),
    "test_models_mobilenet_v4.py": ("lucid.models.vision.mobilenet_v4",),
}

collect_ignore: list[str] = []
for _file, _modules in _PENDING_MODULES.items():
    for _mod in _modules:
        try:
            importlib.import_module(_mod)
        except ModuleNotFoundError:
            collect_ignore.append(_file)
            break
