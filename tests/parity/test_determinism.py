"""Determinism CI — every op marked `deterministic=true` in its OpSchema
must produce bit-exact identical results across runs given the same seed.

The harness:
  1. Sets `set_deterministic(True)`.
  2. For each op family, runs the op twice (fresh Generator each time, same seed)
     and asserts the output bytes are identical.
  3. Cross-device: separately checks CPU vs CPU and GPU vs GPU bit-exactness.
     CPU↔GPU determinism is NOT required (different floating-point order).
"""

from __future__ import annotations

import numpy as np
import pytest

from lucid._C import engine as E


@pytest.fixture(autouse=True)
def _det_mode():
    E.set_deterministic(True)
    yield
    E.set_deterministic(False)


def _to_bytes(t) -> bytes:
    return np.asarray(t.data_as_python()).tobytes()


# --------------------------------------------------------------------------- #
# Random ops
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
def test_randn_seed_repeatable(device):
    g1 = E.Generator(42)
    a = E.randn([4, 5], E.Dtype.F32, device, g1)
    g2 = E.Generator(42)
    b = E.randn([4, 5], E.Dtype.F32, device, g2)
    assert _to_bytes(a) == _to_bytes(b), "randn not bit-exact under same seed"


@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
def test_rand_seed_repeatable(device):
    g1 = E.Generator(0)
    a = E.rand([3, 7], E.Dtype.F32, device, g1)
    g2 = E.Generator(0)
    b = E.rand([3, 7], E.Dtype.F32, device, g2)
    assert _to_bytes(a) == _to_bytes(b)


@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
def test_uniform_seed_repeatable(device):
    g1 = E.Generator(7)
    a = E.uniform([4, 4], -1.0, 1.0, E.Dtype.F32, device, g1)
    g2 = E.Generator(7)
    b = E.uniform([4, 4], -1.0, 1.0, E.Dtype.F32, device, g2)
    assert _to_bytes(a) == _to_bytes(b)


# --------------------------------------------------------------------------- #
# Compute ops are deterministic by construction (no RNG); verify reproducibility.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
def test_matmul_repeatable(device):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((64, 32)).astype("float32")
    B = rng.standard_normal((32, 16)).astype("float32")
    ta = E.TensorImpl(A, device, False)
    tb = E.TensorImpl(B, device, False)
    out1 = E.matmul(ta, tb)
    out2 = E.matmul(ta, tb)
    assert _to_bytes(out1) == _to_bytes(out2)


@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
def test_softmax_repeatable(device):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 32)).astype("float32")
    t = E.TensorImpl(x, device, False)
    out1 = E.softmax(t, -1)
    out2 = E.softmax(t, -1)
    assert _to_bytes(out1) == _to_bytes(out2)


def test_dropout_seed_repeatable():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 8, 8)).astype("float32")
    g1 = E.Generator(123)
    g2 = E.Generator(123)
    t = E.TensorImpl(x, E.Device.CPU, False)
    o1 = E.nn.dropout(t, 0.3, True, g1)
    o2 = E.nn.dropout(t, 0.3, True, g2)
    assert _to_bytes(o1) == _to_bytes(o2)


# --------------------------------------------------------------------------- #
# Backward also deterministic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
def test_matmul_backward_repeatable(device):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((8, 6)).astype("float32")
    B = rng.standard_normal((6, 4)).astype("float32")

    def run():
        ta = E.TensorImpl(A, device, True)
        tb = E.TensorImpl(B, device, True)
        out = E.matmul(ta, tb)
        s = E.sum(out, [], False)
        E.engine_backward(s, False)
        return _to_bytes(ta) + _to_bytes(tb), \
            np.asarray(ta.grad_as_python()).tobytes(), \
            np.asarray(tb.grad_as_python()).tobytes()

    r1 = run()
    r2 = run()
    assert r1 == r2, "matmul backward not bit-exact under deterministic mode"


# --------------------------------------------------------------------------- #
# Non-deterministic ops must throw under set_deterministic(True) when no seed
# --------------------------------------------------------------------------- #

def test_dropout_without_seed_throws_under_deterministic():
    """dropout with training=True and gen=None must throw."""
    x = np.ones((4, 8), dtype="float32")
    t = E.TensorImpl(x, E.Device.CPU, False)
    with pytest.raises(Exception, match="non-deterministic"):
        E.nn.dropout(t, 0.3, True, None)


def test_dropout_with_seed_ok_under_deterministic():
    """dropout with an explicit Generator must NOT throw."""
    x = np.ones((4, 8), dtype="float32")
    t = E.TensorImpl(x, E.Device.CPU, False)
    gen = E.Generator(42)
    out = E.nn.dropout(t, 0.3, True, gen)
    assert out is not None


def test_dropout_inference_mode_ok_under_deterministic():
    """dropout in inference mode (training=False) is deterministic — no throw."""
    x = np.ones((4, 8), dtype="float32")
    t = E.TensorImpl(x, E.Device.CPU, False)
    out = E.nn.dropout(t, 0.3, False, None)
    assert out is not None


# --------------------------------------------------------------------------- #
# Deterministic elementwise ops (UnaryKernel) — bit-exact under same inputs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
@pytest.mark.parametrize("op,args", [
    ("relu",  []),
    ("exp",   []),
    ("log",   []),
    ("sqrt",  []),
    ("neg",   []),
    ("abs",   []),
    ("sin",   []),
    ("cos",   []),
    ("tanh",  []),
    ("sigmoid", []),
])
def test_unary_deterministic(device, op, args):
    rng = np.random.default_rng(0)
    x = np.abs(rng.standard_normal((4, 5)).astype("float32")) + 0.1
    t = E.TensorImpl(x, device, False)
    fn = getattr(E, op)
    o1 = fn(t, *args)
    o2 = fn(t, *args)
    assert _to_bytes(o1) == _to_bytes(o2), f"{op} not bit-exact under deterministic mode"


@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_binary_deterministic(device, op):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 5)).astype("float32")
    y = rng.standard_normal((4, 5)).astype("float32") + 0.5
    ta = E.TensorImpl(x, device, False)
    tb = E.TensorImpl(y, device, False)
    fn = getattr(E, op)
    o1 = fn(ta, tb)
    o2 = fn(ta, tb)
    assert _to_bytes(o1) == _to_bytes(o2), f"{op} not bit-exact under deterministic mode"


@pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
def test_reduce_sum_deterministic(device):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 6)).astype("float32")
    t = E.TensorImpl(x, device, False)
    o1 = E.sum(t, [0], True)
    o2 = E.sum(t, [0], True)
    assert _to_bytes(o1) == _to_bytes(o2)
