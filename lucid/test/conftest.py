import importlib
import numpy as np
import pytest

try:
    importlib.import_module("lucid")
    _LUCID_IMPORT_ERROR = None
except Exception as exc:
    _LUCID_IMPORT_ERROR = exc


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "parity: lucid↔torch numerical parity test")
    config.addinivalue_line(
        "markers", "slow: slow test (model forwards, long trajectories)"
    )
    config.addinivalue_line("markers", "smoke: quick sanity test")


@pytest.fixture(scope="session", autouse=True)
def require_lucid_package() -> None:
    if _LUCID_IMPORT_ERROR is not None:
        pytest.skip(
            f"lucid import failed during collection: {_LUCID_IMPORT_ERROR}",
            allow_module_level=True,
        )


@pytest.fixture(autouse=True)
def _reset_numpy_global_seed():
    state = np.random.get_state()
    try:
        np.random.seed(0)
        yield
    finally:
        np.random.set_state(state)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    _ = config
    parity_models_full = "parity/models/test_models_full.py"
    integration_marker = "parity/integration/"
    prior: list[pytest.Item] = []
    trailing: list[pytest.Item] = []
    very_trailing: list[pytest.Item] = []
    for item in items:
        file_path, _, _ = item.location
        if parity_models_full in file_path:
            very_trailing.append(item)
        elif integration_marker in file_path:
            trailing.append(item)
        else:
            prior.append(item)
    items[:] = prior + trailing + very_trailing
