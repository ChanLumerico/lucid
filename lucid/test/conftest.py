import importlib
import pytest


try:
    importlib.import_module("lucid")
    _LUCID_IMPORT_ERROR = None
except Exception as exc:
    _LUCID_IMPORT_ERROR = exc


@pytest.fixture(scope="session", autouse=True)
def require_lucid_package() -> None:
    if _LUCID_IMPORT_ERROR is not None:
        pytest.skip(
            f"lucid import failed during collection: {_LUCID_IMPORT_ERROR}",
            allow_module_level=True,
        )
