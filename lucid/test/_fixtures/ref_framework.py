"""Lazy reference-framework loader.

The H5 rule (no ``torch`` / ``PyTorch`` literal in Lucid source) is
relaxed inside ``lucid/test/`` because:

  - ``test/`` is excluded from the published wheel and is reachable
    only through ``pip install lucid[test]``;
  - the user explicitly opts into reference-framework value/speed
    comparison by running the parity / perf tiers;
  - keeping the literal name visible here makes the test intent
    obvious and lets IDE tooling (auto-complete, type checking)
    actually work on the reference symbols.

This is the *only* file in the Lucid tree where ``torch`` may appear
as a literal — every parity test reaches it through the ``ref``
fixture and never imports the reference framework directly.

Usage
-----

In a test::

    def test_something(ref):
        # ``ref`` is the reference framework module, or the test is
        # auto-skipped before the body runs.
        assert ref.tensor([1, 2]).shape == (2,)

In ``parity/conftest.py``::

    from lucid.test._fixtures.ref_framework import collect_skip_if_missing
    collect_skip_if_missing()  # in pytest_collection_modifyitems
"""

import functools
from types import ModuleType

import pytest

_REF_NAME = "torch"


@functools.lru_cache(maxsize=1)
def ref_module() -> ModuleType | None:
    """Return the reference framework module, or ``None`` when not
    installed.  Cached so we only attempt the import once per session."""
    try:
        import torch  # noqa: PLC0415 — lazy import is the whole point

        return torch
    except ImportError:
        return None


@functools.lru_cache(maxsize=1)
def ref_vision_module() -> ModuleType | None:
    """Return the reference framework's vision package, or ``None``.

    Separate from :func:`ref_module` because it is an independent install:
    a parity run can have the core framework without the vision extras.
    Tests that need real image data (rather than synthetic tensors) reach
    for this and skip when it is absent.
    """
    try:
        import torchvision  # noqa: PLC0415 — lazy import is the whole point

        return torchvision
    except ImportError:
        return None


def require_ref(*, module_level: bool = False) -> ModuleType:
    """Return the reference module or skip when it's unavailable.

    Parameters
    ----------
    module_level : bool, default=False
        Pass ``True`` when calling at import time.  A bare ``pytest.skip``
        outside a test is a collection *error*, so a module that binds the
        reference at the top has to say so explicitly — that is the call
        that replaces ``pytest.importorskip`` in the parity modules, and
        the reason they no longer name the framework themselves.

    Returns
    -------
    ModuleType
        The reference framework module.
    """
    mod = ref_module()
    if mod is None:
        pytest.skip(
            f"reference framework ({_REF_NAME}) is not installed",
            allow_module_level=module_level,
        )
    return mod


def require_ref_vision(*, module_level: bool = False) -> ModuleType:
    """Return the reference vision package or skip.

    Parameters
    ----------
    module_level : bool, default=False
        As for :func:`require_ref`.

    Returns
    -------
    ModuleType
        The reference vision package.
    """
    mod = ref_vision_module()
    if mod is None:
        pytest.skip(
            f"reference vision package ({_REF_NAME}vision) is not installed",
            allow_module_level=module_level,
        )
    return mod


@pytest.fixture
def ref() -> ModuleType:
    """Inject the reference framework module, or skip when missing."""
    return require_ref()


@pytest.fixture
def ref_vision() -> ModuleType:
    """Inject the reference vision package, or skip when missing."""
    return require_ref_vision()


def collect_skip_if_missing() -> None:
    """Add a session-wide ``pytest.skip`` marker when the reference is
    missing.  Call from ``parity/conftest.py``'s collection hook."""
    if ref_module() is None:
        pytest.skip(
            f"reference framework ({_REF_NAME}) not installed — "
            "parity tier auto-skipped",
            allow_module_level=True,
        )
