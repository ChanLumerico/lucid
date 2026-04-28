"""AMP (autocast) verification.

Current state: the AMP infrastructure (`AutocastGuard`, `amp_is_active`,
`amp_active_dtype`) is plumbed through the engine binding surface and ops
declare `AmpPolicy` in their schemas — but **no compute kernel currently
consults `amp::is_active()` to actually downcast inputs**. AMP is metadata-
only at the moment.

These tests:
  1. Verify the metadata API works (RAII activation, nested guards, query
     functions) — those parts ARE implemented.
  2. Pin the current behavior: matmul under `AutocastGuard(F16)` still
     returns F32 because no kernel downcasts. When AMP is implemented at
     the kernel level later, these "expected F32" assertions will turn
     into `xfail` markers and we'll add new positive tests.

The schemas already carry the policy declarations (Promote / KeepInput /
ForceFP32), so once a kernel-side implementation lands, ops will know
how to react without further schema work.
"""

from __future__ import annotations

import numpy as np
import pytest

from lucid._C import engine as E


# --------------------------------------------------------------------------- #
# Metadata API (implemented — these MUST pass).
# --------------------------------------------------------------------------- #

def test_amp_inactive_by_default():
    assert E.amp_is_active() is False
    assert E.amp_active_dtype() is None


def test_autocast_guard_activates():
    assert E.amp_is_active() is False
    g = E.AutocastGuard(E.Dtype.F16)
    try:
        assert E.amp_is_active() is True
        assert E.amp_active_dtype() == E.Dtype.F16
    finally:
        del g
    assert E.amp_is_active() is False


def test_autocast_guard_nested():
    g_outer = E.AutocastGuard(E.Dtype.F16)
    try:
        assert E.amp_active_dtype() == E.Dtype.F16
        g_inner = E.AutocastGuard(E.Dtype.F32)
        try:
            assert E.amp_active_dtype() == E.Dtype.F32
        finally:
            del g_inner
        assert E.amp_active_dtype() == E.Dtype.F16
    finally:
        del g_outer
    assert E.amp_is_active() is False


def test_amp_policy_names():
    assert E.amp_policy_name(E.AmpPolicy.Promote) == "Promote"
    assert E.amp_policy_name(E.AmpPolicy.KeepInput) == "KeepInput"
    assert E.amp_policy_name(E.AmpPolicy.ForceFP32) == "ForceFP32"


# --------------------------------------------------------------------------- #
# Pin current (unimplemented) behavior. Each of these should turn into a
# proper AMP test (output dtype = autocast target) when kernel-side AMP
# lands. Convert to xfail at that time.
# --------------------------------------------------------------------------- #

def _matmul_dtype_under_autocast(target_dtype):
    A = np.random.randn(4, 5).astype("float32")
    B = np.random.randn(5, 6).astype("float32")
    ta = E.TensorImpl(A, E.Device.CPU, False)
    tb = E.TensorImpl(B, E.Device.CPU, False)
    g = E.AutocastGuard(target_dtype)
    try:
        out = E.matmul(ta, tb)
    finally:
        del g
    return out.dtype


def test_matmul_no_kernel_amp_yet():
    # Promote policy on matmul — when AMP is wired, this should equal F16.
    # Currently: no kernel downcast → output stays F32.
    assert _matmul_dtype_under_autocast(E.Dtype.F16) == E.Dtype.F32


def test_no_op_consults_amp_state():
    """Honest check: scan a representative slice of ops for AMP awareness.
    If/when any op starts consulting amp::is_active(), this test should be
    deleted (it'll become a positive test instead).
    """
    A = np.random.randn(4, 5).astype("float32")
    M = np.random.randn(5, 5).astype("float32")
    ta = E.TensorImpl(A, E.Device.CPU, False)
    tm = E.TensorImpl(M, E.Device.CPU, False)
    g = E.AutocastGuard(E.Dtype.F16)
    try:
        # All of these should output F32 today (no AMP enforcement).
        outputs = [
            E.relu(ta),
            E.exp(ta),
            E.sum(ta, [], False),
            E.softmax(ta, -1),
            E.add(ta, ta),
            E.matmul(tm, tm),  # square 5×5
        ]
    finally:
        del g
    assert all(o.dtype == E.Dtype.F32 for o in outputs), \
        "An op now consults AMP state — replace this test with a positive AMP check"
