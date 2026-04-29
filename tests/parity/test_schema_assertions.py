"""Phase 5.6 — Schema-vs-behavior assertion harness.

Auto-verifies that every op's runtime behavior matches its OpSchema declaration:

  1. AmpPolicy.ForceFP32  → under any autocast, output dtype is always F32.
  2. AmpPolicy.KeepInput  → under any autocast, output dtype == input dtype.
  3. AmpPolicy.Promote    → under F64 autocast on CPU, output dtype is F64.
                           Under F16 autocast on CPU, stays F32 (no F16 ops).
  4. deterministic=true   → same input twice produces bit-exact output.
  5. deterministic=false  → under set_deterministic(True), throws without seed.

Coverage: all ops reachable with a single F32 tensor of shape (4, 5).
Ops requiring special inputs (positive values, axes, scalar params) are handled
by a per-op input factory registered in SPECIAL_INPUTS below.
"""

from __future__ import annotations

import numpy as np
import pytest

from lucid._C import engine as E


# --------------------------------------------------------------------------- #
# Input factories — ops that need non-default inputs
# --------------------------------------------------------------------------- #

RNG = np.random.default_rng(42)

def _pos_f32(shape=(4, 5)):
    """Positive F32 input (log, log2, sqrt, reciprocal)."""
    return np.abs(RNG.standard_normal(shape).astype("float32")) + 0.1

def _unit_f32(shape=(4, 5)):
    """F32 input in (-1, 1) for arcsin/arccos."""
    return np.clip(RNG.standard_normal(shape).astype("float32"), -0.99, 0.99)

def _std_f32(shape=(4, 5)):
    return RNG.standard_normal(shape).astype("float32")

# op_name → (input_array, call_fn)
# call_fn(tensor) → output tensor
SPECIAL_INPUTS: dict[str, tuple] = {
    "arccos":     (_unit_f32(), lambda t: E.arccos(t)),
    "arcsin":     (_unit_f32(), lambda t: E.arcsin(t)),
    "arctan":     (_std_f32(),  lambda t: E.arctan(t)),
    "log":        (_pos_f32(),  lambda t: E.log(t)),
    "log2":       (_pos_f32(),  lambda t: E.log2(t)),
    "sqrt":       (_pos_f32(),  lambda t: E.sqrt(t)),
    "reciprocal": (_pos_f32(),  lambda t: E.reciprocal(t)),
    # Ops with extra args
    "sum":        (_std_f32(),  lambda t: E.sum(t, [], False)),
    "mean":       (_std_f32(),  lambda t: E.mean(t, [], False)),
    "max":        (_std_f32(),  lambda t: E.max(t, [], False)),
    "min":        (_std_f32(),  lambda t: E.min(t, [], False)),
    "prod":       (_std_f32(),  lambda t: E.prod(t, [], False)),
    "softmax":    (_std_f32(),  lambda t: E.softmax(t, -1)),
    "elu":        (_std_f32(),  lambda t: E.elu(t, 1.0)),
    "leaky_relu": (_std_f32(),  lambda t: E.leaky_relu(t, 0.01)),
}

# Simple unary ops: call as E.<name>(tensor)
SIMPLE_UNARY = [
    "abs", "ceil", "cos", "cosh", "cube", "exp", "floor", "gelu",
    "hard_sigmoid", "hard_swish", "mish", "neg", "relu", "relu6",
    "round", "selu", "sigmoid", "sign", "silu", "sin", "sinh",
    "softplus", "square", "tan", "tanh",
]

def _make_entry(name: str):
    """Return (input_np, call_fn) for a given op."""
    if name in SPECIAL_INPUTS:
        return SPECIAL_INPUTS[name]
    fn = getattr(E, name)
    return _std_f32(), fn


def _all_testable_ops():
    """All ops that can be tested with a single F32 tensor."""
    names = list(SIMPLE_UNARY) + list(SPECIAL_INPUTS.keys())
    return sorted(set(names))


# --------------------------------------------------------------------------- #
# Collect schema metadata once
# --------------------------------------------------------------------------- #

_SCHEMAS = {s.name: s for s in E.op_registry_all()}


def _schema(name):
    return _SCHEMAS.get(name)


# --------------------------------------------------------------------------- #
# 1. ForceFP32 — always F32 under any autocast
# --------------------------------------------------------------------------- #

FORCE_FP32_OPS = [n for n in _all_testable_ops()
                  if _schema(n) and _schema(n).amp_policy == E.AmpPolicy.ForceFP32]


@pytest.mark.parametrize("op", FORCE_FP32_OPS)
@pytest.mark.parametrize("autocast_dt", [E.Dtype.F16, E.Dtype.F64])
def test_force_fp32_policy(op, autocast_dt):
    arr, fn = _make_entry(op)
    t = E.TensorImpl(arr, E.Device.CPU, False)
    g = E.AutocastGuard(autocast_dt)
    try:
        out = fn(t)
    finally:
        del g
    assert out.dtype == E.Dtype.F32, \
        f"{op} (ForceFP32) under {autocast_dt}: expected F32, got {out.dtype}"


# --------------------------------------------------------------------------- #
# 2. KeepInput — output dtype always == input dtype
# --------------------------------------------------------------------------- #

KEEP_INPUT_OPS = [n for n in _all_testable_ops()
                  if _schema(n) and _schema(n).amp_policy == E.AmpPolicy.KeepInput]


@pytest.mark.parametrize("op", KEEP_INPUT_OPS)
@pytest.mark.parametrize("autocast_dt", [E.Dtype.F16, E.Dtype.F64])
def test_keep_input_policy(op, autocast_dt):
    arr, fn = _make_entry(op)
    t = E.TensorImpl(arr, E.Device.CPU, False)
    in_dt = t.dtype
    g = E.AutocastGuard(autocast_dt)
    try:
        out = fn(t)
    finally:
        del g
    assert out.dtype == in_dt, \
        f"{op} (KeepInput) under {autocast_dt}: expected {in_dt}, got {out.dtype}"


# --------------------------------------------------------------------------- #
# 3. Promote — CPU F16 stays F32, CPU F64 upcasts to F64
# --------------------------------------------------------------------------- #

PROMOTE_OPS = [n for n in _all_testable_ops()
               if _schema(n) and _schema(n).amp_policy == E.AmpPolicy.Promote]


@pytest.mark.parametrize("op", PROMOTE_OPS)
def test_promote_cpu_f16_stays_f32(op):
    """CPU cannot do F16 math — Promote under F16 autocast stays F32."""
    arr, fn = _make_entry(op)
    t = E.TensorImpl(arr, E.Device.CPU, False)
    g = E.AutocastGuard(E.Dtype.F16)
    try:
        out = fn(t)
    finally:
        del g
    assert out.dtype == E.Dtype.F32, \
        f"{op} (Promote) on CPU under F16: expected F32 fallback, got {out.dtype}"


@pytest.mark.parametrize("op", PROMOTE_OPS)
def test_promote_cpu_f64_upcasts(op):
    """F32 input + Promote + F64 autocast → output should be F64."""
    arr, fn = _make_entry(op)
    t = E.TensorImpl(arr, E.Device.CPU, False)
    g = E.AutocastGuard(E.Dtype.F64)
    try:
        out = fn(t)
    finally:
        del g
    assert out.dtype == E.Dtype.F64, \
        f"{op} (Promote) on CPU under F64: expected F64, got {out.dtype}"


# --------------------------------------------------------------------------- #
# 4. deterministic=true — bit-exact across two calls with same input
# --------------------------------------------------------------------------- #

DET_OPS = [n for n in _all_testable_ops()
           if _schema(n) and _schema(n).deterministic]


@pytest.mark.parametrize("op", DET_OPS)
def test_deterministic_ops_bit_exact(op):
    arr, fn = _make_entry(op)
    t = E.TensorImpl(arr, E.Device.CPU, False)
    o1 = fn(t)
    o2 = fn(t)
    b1 = np.asarray(o1.data_as_python()).tobytes()
    b2 = np.asarray(o2.data_as_python()).tobytes()
    assert b1 == b2, f"{op} produced different results on two identical calls"


# --------------------------------------------------------------------------- #
# 5. deterministic=false — throws under set_deterministic(True) without seed
# --------------------------------------------------------------------------- #

NONDETERMINISTIC_OPS = [
    ("dropout",       lambda t: E.nn.dropout(t, 0.3, True, None)),
    ("dropoutnd",     lambda t: E.nn.dropoutnd(t, 0.3, True, None)),
    ("alpha_dropout", lambda t: E.nn.alpha_dropout(t, 0.3, True, None)),
    # drop_block needs 4-D input (N, C, H, W)
    ("drop_block",    lambda _: E.nn.drop_block(
        E.TensorImpl(np.ones((2, 4, 8, 8), dtype="float32"), E.Device.CPU, False),
        2, 0.3, 1e-6, None)),
    # drop_path needs 2-D+ batch input
    ("drop_path",     lambda t: E.nn.drop_path(t, 0.3, True, None)),
]


@pytest.mark.parametrize("op,call_fn", NONDETERMINISTIC_OPS)
def test_nondeterministic_throws_without_seed(op, call_fn):
    E.set_deterministic(True)
    try:
        x = np.ones((4, 8), dtype="float32")
        t = E.TensorImpl(x, E.Device.CPU, False)
        with pytest.raises(Exception, match="non-deterministic"):
            call_fn(t)
    finally:
        E.set_deterministic(False)


# --------------------------------------------------------------------------- #
# 6. OpSchema 5.4 — arity fields accessible from Python
# --------------------------------------------------------------------------- #

def test_schema_has_arity_fields():
    """All registered schemas expose input_arity, output_arity, stable_input_indices."""
    schemas = E.op_registry_all()
    assert len(schemas) > 0
    for s in schemas:
        # Fields exist and have correct types
        assert isinstance(s.input_arity, int)
        assert isinstance(s.output_arity, int)
        assert isinstance(s.stable_input_indices, list)
        # output_arity is always 1 for current ops
        assert s.output_arity == 1, \
            f"{s.name}: unexpected output_arity={s.output_arity}"
