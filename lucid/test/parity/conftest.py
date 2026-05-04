"""
conftest.py for parity tests: skip entire module if PyTorch is not installed.
"""

import pytest


def pytest_configure(config):
    try:
        import torch  # noqa: F401
    except ImportError:
        pass  # Skip message handled at collection time


def pytest_collection_modifyitems(items, config):
    try:
        import torch  # noqa: F401
    except ImportError:
        skip_mark = pytest.mark.skip(reason="PyTorch not installed")
        for item in items:
            if "parity" in str(item.fspath):
                item.add_marker(skip_mark)
