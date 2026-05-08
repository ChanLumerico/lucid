"""Parity-tier conftest.

Auto-skip every test under ``lucid/test/parity/`` when the reference
framework isn't installed.  Doing this at collection time means the
parity tier never even attempts to import the reference module from
inside individual test modules.
"""

import pytest

from lucid.test._fixtures.ref_framework import ref_module


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if ref_module() is not None:
        return
    skip_marker = pytest.mark.skip(
        reason="reference framework not installed — parity tier auto-skipped"
    )
    for item in items:
        if "lucid/test/parity/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip_marker)
