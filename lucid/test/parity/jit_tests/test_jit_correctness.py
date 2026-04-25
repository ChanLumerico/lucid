import numpy as np
import pytest

import lucid
import lucid.nn as lnn


def _close(a, b, *, rtol=1e-7, atol=1e-9):
    np.testing.assert_allclose(
        np.asarray(a.data if hasattr(a, "data") else a),
        np.asarray(b.data if hasattr(b, "data") else b),
        rtol=rtol,
        atol=atol,
    )


def test_jit_function_simple():
    def fn(a, b):
        return (a + b) * a - b.mean()

    compiled = lucid.compile(fn)
    rng = np.random.default_rng(0)
    a = lucid.tensor(rng.standard_normal((3, 4)).astype(np.float64))
    b = lucid.tensor(rng.standard_normal((3, 4)).astype(np.float64))
    eager = fn(a, b)
    jitted = compiled(a, b)
    _close(eager, jitted, rtol=1e-10)


def test_jit_cache_reuse():
    def fn(a):
        return a.sum() + a.mean()

    compiled = lucid.compile(fn, max_cache_entries=4)
    a = lucid.tensor(np.ones((3, 4), dtype=np.float64))
    out1 = compiled(a)
    out2 = compiled(a)
    _close(out1, out2)


def test_jit_module_forward():
    m = lnn.Linear(5, 3, bias=True)
    compiled = lucid.compile(m)
    x = lucid.tensor(
        np.random.default_rng(1).standard_normal((4, 5)).astype(np.float64)
    )
    eager = m(x)
    jitted = compiled(x)
    _close(eager, jitted, rtol=1e-10)


def test_jit_backward_match():
    def fn(a):
        return ((a - 0.5) ** 2).sum()

    compiled = lucid.compile(fn)
    rng = np.random.default_rng(2)
    base = rng.standard_normal((3, 4)).astype(np.float64)
    a_eager = lucid.tensor(base.copy(), requires_grad=True)
    a_jit = lucid.tensor(base.copy(), requires_grad=True)
    fn(a_eager).backward()
    compiled(a_jit).backward()
    _close(a_eager.grad, a_jit.grad, rtol=1e-10)


def test_jit_shape_dispatch():
    def fn(a):
        return a * 2 + 1

    compiled = lucid.compile(fn)
    a1 = lucid.tensor(np.ones((3,), dtype=np.float64))
    a2 = lucid.tensor(np.ones((4, 5), dtype=np.float64))
    o1 = compiled(a1)
    o2 = compiled(a2)
    assert o1.shape == (3,)
    assert o2.shape == (4, 5)
