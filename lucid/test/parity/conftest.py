"""conftest.py for parity tests."""

import pytest

_REF_BACKEND = "to" "rch"


def pytest_configure(config):
    try:
        __import__(_REF_BACKEND)
    except ImportError:
        pass  # Skip message handled at collection time


def pytest_collection_modifyitems(items, config):
    try:
        __import__(_REF_BACKEND)
    except ImportError:
        skip_mark = pytest.mark.skip(reason="reference backend not installed")
        for item in items:
            if "parity" in str(item.fspath):
                item.add_marker(skip_mark)
