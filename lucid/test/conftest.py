"""Lucid test suite — root pytest configuration.

This file wires the per-area fixtures into pytest's discovery, sets
the autouse determinism hook, and registers a single
``configure_markers`` step that mirrors the marker matrix declared in
``pyproject.toml``.

What you get for free in any test:

* ``device`` — parametrize over CPU and (when present) Metal.
* ``device_cpu_only`` / ``device_gpu_only`` / ``cross_device_pair``.
* ``float_dtype`` — parametrize over float32 + float64.
* ``int_dtype`` — parametrize over int8/16/32/64.
* ``ref`` — lazy reference framework module (skips test if missing).
* ``tensor_factory`` — device-aware ``make_tensor`` shorthand.
* ``bench`` — ``pytest-benchmark``-compatible benchmark callable
  (degrades gracefully when the dep is missing).
* Autouse ``manual_seed(0)`` per test.
"""

import pytest

import lucid

# Re-export every fixture from the per-area modules so test files just
# need to declare the fixture name in their argument list.
from lucid.test._fixtures.devices import (  # noqa: F401
    device,
    device_cpu_only,
    device_gpu_only,
    cross_device_pair,
    skip_if_unsupported,
)
from lucid.test._fixtures.dtypes import (  # noqa: F401
    float_dtype,
    float_dtype_extended,
    int_dtype,
)
from lucid.test._fixtures.ref_framework import ref  # noqa: F401
from lucid.test._fixtures.tensors import tensor_factory  # noqa: F401
from lucid.test._fixtures.perf import bench  # noqa: F401


@pytest.fixture(autouse=True)
def _seed_per_test() -> None:
    """Seed Lucid's default RNG to 0 before every test for
    bit-reproducibility of any sampling op.

    Reference-framework seeding (when applicable) is handled inside
    parity tests via ``lucid.test._helpers.seeding.seed_all``.
    """
    lucid.manual_seed(0)
