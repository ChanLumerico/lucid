"""Lucid test suite.

Public symbols re-exported here are part of the test toolkit (helpers,
fixtures, comparison utilities) — they're imported by Lucid itself
when the user runs `pytest lucid/test/`.

The suite is structured as:
  unit/         — focused per-op behavior, parametrized over CPU + GPU
  parity/       — value comparison vs the reference framework
                  (auto-skip when the reference isn't installed)
  numerical/    — invariants, stability, precision sweeps
  integration/  — multi-component flows (training loops, checkpoint round-trips)
  perf/         — pytest-benchmark powered timing tests
  stubs/        — type-stub regression checks

Bridge invariant: all reference-framework imports happen lazily inside
``_fixtures/ref_framework.py``.  Lucid users who never run tests never
import the reference framework.
"""

from lucid.test._helpers.compare import assert_close, assert_equal_int

__all__ = ["assert_close", "assert_equal_int"]
