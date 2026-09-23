"""Compiled training (``make_step``) against eager backward, case by case.

The cases are the float32 cases of :mod:`._op_matrix`, each wrapped as
``fn(x * w)`` so the gradient reaches the parameter ``w`` through the op —
see :mod:`._grad_matrix`.  A fallback to eager is allowed only where
:data:`EXPECTED_EAGER_GRAD` says why.

Before this test, a manual-VJP gap handed the whole backward to MPSGraph's
automatic differentiation, which aborted the process for 28 of these cases
(``where``, ``leaky_relu``, ``elu``, ``max``, ``cumsum``, ``bce``, …) and
returned a wrong gradient for 3 (``arcsin``, ``triu`` / ``tril`` off the
diagonal).  That fallback is now taken only for traces made of ops measured
to differentiate correctly.
"""

import pytest

import lucid

from lucid.test.unit.compile._grad_matrix import (
    EXPECTED_EAGER_GRAD,
    GRAD_CASES,
    run_grad,
)
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.mark.parametrize("case", GRAD_CASES, ids=lambda c: c.name)
def test_compiled_gradient_matches_eager(case: object) -> None:
    outcome = run_grad(case)  # type: ignore[arg-type]
    if outcome.status == "skip":
        pytest.skip(outcome.detail)
    if outcome.status == "eager":
        if case.name in EXPECTED_EAGER_GRAD:  # type: ignore[attr-defined]
            return
        pytest.fail(f"fell back to eager, not listed: {outcome.detail}")
    assert outcome.status == "ok", f"{outcome.status}: {outcome.detail}"
