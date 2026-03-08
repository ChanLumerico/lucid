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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    _ = config
    training_test_file = "test_training.py"
    prior_items: list[pytest.Item] = []
    trailing_items: list[pytest.Item] = []

    for item in items:
        file_path, _, _ = item.location
        if file_path.endswith(training_test_file):
            trailing_items.append(item)
        else:
            prior_items.append(item)

    if trailing_items:
        items[:] = prior_items + trailing_items
