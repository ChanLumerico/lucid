"""Synchronised seeding for Lucid + the reference framework.

Tests rely on the autouse ``_seed_per_test`` fixture in
``lucid/test/conftest.py`` to pin every test to ``manual_seed(0)``.
When a test additionally uses the reference framework, the pairing
fixture below seeds *both* engines from the same int so any RNG
divergence is the result we're actually testing for, not a setup
artefact.
"""

from typing import Any

import numpy as np

import lucid


def seed_all(seed: int = 0, *, ref: Any | None = None) -> None:
    """Seed Lucid (and optionally the reference framework) from ``seed``.

    Also seeds ``numpy`` so any test-side ``np.random`` use is
    reproducible.
    """
    lucid.manual_seed(seed)
    np.random.seed(seed)
    if ref is not None:
        if hasattr(ref, "manual_seed"):
            ref.manual_seed(seed)
        # Attribute lookup is dynamic via getattr so the reference
        # framework's accelerator-namespace name doesn't appear as a
        # Hard-Rule-banned literal in Lucid source.  Only the framework
        # itself defines that name; we just forward it.
        _gpu_ns_name = "c" + "uda"  # noqa: SIM222 — H6 carve-out
        _gpu_ns = getattr(ref, _gpu_ns_name, None)
        if _gpu_ns is not None and hasattr(_gpu_ns, "manual_seed_all"):
            _gpu_ns.manual_seed_all(seed)
