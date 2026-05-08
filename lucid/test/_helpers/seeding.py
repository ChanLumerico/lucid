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
        if hasattr(ref, "cuda") and hasattr(ref.cuda, "manual_seed_all"):
            ref.cuda.manual_seed_all(seed)
