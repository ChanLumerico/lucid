"""AMP (autocast) verification — Phase 5 kernel-side enforcement.

SchemaGuard is now wired into UnaryKernel, BinaryKernel, and ReduceKernel
forward(). Tests verify:

  1. Metadata API (RAII activation, nesting, query functions).
  2. Positive policy enforcement on CPU:
       - KeepInput  → output dtype == input dtype regardless of autocast.
       - ForceFP32  → output dtype == F32 regardless of autocast.
       - Promote    → on CPU with F16 autocast, falls back to F32
                      (vDSP/vForce have no F16 ops); with F64 autocast,
                      promotes to F64.
  3. No-autocast baseline: output dtype always == input dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

from lucid._C import engine as E


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _f32(shape):
    return E.TensorImpl(np.random.randn(*shape).astype("float32"), E.Device.CPU, False)

def _f64(shape):
    return E.TensorImpl(np.random.randn(*shape).astype("float64"), E.Device.CPU, False)


# --------------------------------------------------------------------------- #
# 1. Metadata API
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
# 2. No-autocast baseline: dtype unchanged
# --------------------------------------------------------------------------- #

def test_no_autocast_dtype_preserved():
    t = _f32([4, 5])
    assert E.relu(t).dtype == E.Dtype.F32
    assert E.exp(t).dtype == E.Dtype.F32
    assert E.add(t, t).dtype == E.Dtype.F32
    assert E.sum(t, [], False).dtype == E.Dtype.F32


# --------------------------------------------------------------------------- #
# 3. KeepInput policy — dtype always follows input, ignores autocast
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("autocast_dt", [E.Dtype.F16, E.Dtype.F64])
def test_keep_input_policy_ignores_autocast(autocast_dt):
    """relu, round, floor, ceil — KeepInput: output dtype == input dtype."""
    t = _f32([4, 5])
    g = E.AutocastGuard(autocast_dt)
    try:
        # relu: KeepInput
        assert E.relu(t).dtype == E.Dtype.F32, \
            f"relu under {autocast_dt} should stay F32 (KeepInput)"
        # round: KeepInput
        assert E.round(t).dtype == E.Dtype.F32, \
            f"round under {autocast_dt} should stay F32 (KeepInput)"
    finally:
        del g


# --------------------------------------------------------------------------- #
# 4. ForceFP32 policy — always F32, even under F64 autocast
# --------------------------------------------------------------------------- #

def test_force_fp32_under_f64_autocast():
    """exp, log — ForceFP32: always F32 regardless of autocast dtype."""
    t = _f32([4, 5])
    g = E.AutocastGuard(E.Dtype.F64)
    try:
        # exp: ForceFP32
        assert E.exp(t).dtype == E.Dtype.F32, \
            "exp under F64 autocast should stay F32 (ForceFP32)"
        # log: ForceFP32
        assert E.log(t).dtype == E.Dtype.F32, \
            "log under F64 autocast should stay F32 (ForceFP32)"
    finally:
        del g


def test_force_fp32_under_f16_autocast():
    """ForceFP32 also resists F16 downcast."""
    t = _f32([4, 5])
    g = E.AutocastGuard(E.Dtype.F16)
    try:
        assert E.exp(t).dtype == E.Dtype.F32
        assert E.log(t).dtype == E.Dtype.F32
    finally:
        del g


def test_sum_promote_under_f64_autocast():
    """sum has Promote policy — under F64 autocast on CPU, upcasts to F64."""
    t = _f32([4, 5])
    g = E.AutocastGuard(E.Dtype.F64)
    try:
        assert E.sum(t, [], False).dtype == E.Dtype.F64, \
            "sum (Promote) under F64 autocast should upcast to F64"
    finally:
        del g


# --------------------------------------------------------------------------- #
# 5. Promote policy — CPU behavior
# --------------------------------------------------------------------------- #

def test_promote_cpu_f16_fallback_to_f32():
    """CPU + Promote + F16 autocast → F32 (vDSP has no F16 ops)."""
    t = _f32([4, 5])
    g = E.AutocastGuard(E.Dtype.F16)
    try:
        # add: Promote → should NOT downcast to F16 on CPU
        assert E.add(t, t).dtype == E.Dtype.F32, \
            "add (Promote) on CPU under F16 autocast should stay F32"
        # sub: Promote
        assert E.sub(t, t).dtype == E.Dtype.F32
        # mul: Promote
        assert E.mul(t, t).dtype == E.Dtype.F32
    finally:
        del g


def test_promote_cpu_f64_upcasts():
    """CPU + Promote + F64 autocast → F64 (legal upcast)."""
    t = _f32([4, 5])
    g = E.AutocastGuard(E.Dtype.F64)
    try:
        out = E.add(t, t)
        assert out.dtype == E.Dtype.F64, \
            f"add (Promote) on CPU under F64 autocast should upcast to F64, got {out.dtype}"
    finally:
        del g


def test_promote_f64_input_f32_autocast_keeps_f64():
    """F64 input + F32 autocast with Promote → F32 (autocast target)."""
    t = _f64([4, 5])
    g = E.AutocastGuard(E.Dtype.F32)
    try:
        out = E.add(t, t)
        assert out.dtype == E.Dtype.F32, \
            f"add (Promote) on F64 input under F32 autocast should give F32, got {out.dtype}"
    finally:
        del g


# --------------------------------------------------------------------------- #
# 6. Matmul — Promote policy, but goes through its own forward (not BinaryKernel)
#    so AMP is not yet applied; dtype == input dtype.
# --------------------------------------------------------------------------- #

def test_matmul_dtype_unchanged_under_autocast():
    """matmul has its own forward() — SchemaGuard not yet wired there."""
    A = np.random.randn(4, 5).astype("float32")
    B = np.random.randn(5, 6).astype("float32")
    ta = E.TensorImpl(A, E.Device.CPU, False)
    tb = E.TensorImpl(B, E.Device.CPU, False)
    g = E.AutocastGuard(E.Dtype.F16)
    try:
        out = E.matmul(ta, tb)
    finally:
        del g
    assert out.dtype == E.Dtype.F32, \
        "matmul (own forward) should still output F32 under F16 autocast"
