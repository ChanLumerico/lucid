#!/usr/bin/env python3
"""Phase 4d-D — numerical gradcheck for the new C++ backward formulas:
   var, trace, cumsum, dot (1D & 2D), outer.

Compares analytical grad_x = engine_backward(forward) against finite-difference
numerical grad. Tolerance: 5e-3 absolute (loose because we use float32).

The backward formulas under test:
  var:    dx = (2/N) * (x - mean) * broadcast(grad)
  trace:  dx = eye * grad   (2-D input only)
  cumsum: dx = reverse(cumsum(reverse(grad, axis), axis), axis)
  dot 1D: da = b * grad,   db = a * grad
  dot 2D: da = grad @ b.T, db = a.T @ grad
  outer:  da = grad @ b,   db = grad.T @ a
"""
import sys
import numpy as np

from lucid._C import engine as _C_engine

PASSED = 0
FAILED = 0


def to_np(t):
    return np.array(t.data_as_python())


def make_leaf(arr, requires_grad=True):
    return _C_engine.TensorImpl(arr.copy(), _C_engine.Device.CPU, requires_grad)


def grad_of(out, leaves):
    """Returns each leaf's gradient as a numpy array after engine_backward."""
    for leaf in leaves:
        leaf.zero_grad()
    _C_engine.engine_backward(out, retain_graph=False)
    return [np.array(leaf.grad_as_python()) for leaf in leaves]


def fd_grad(forward_np, x, eps=1e-3):
    """Symmetric finite-difference gradient of `forward_np(x) -> scalar`."""
    x = x.astype(np.float64)
    g = np.zeros_like(x)
    flat = x.reshape(-1)
    grad_flat = g.reshape(-1)
    base = float(forward_np(x.astype(np.float32)))
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        plus = float(forward_np(x.astype(np.float32)))
        flat[i] = old - eps
        minus = float(forward_np(x.astype(np.float32)))
        flat[i] = old
        grad_flat[i] = (plus - minus) / (2 * eps)
    return g.astype(np.float32)


def check(name, got, expected, tol=5e-3):
    global PASSED, FAILED
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if got.shape != expected.shape:
        FAILED += 1
        print(f"  FAIL  {name}: shape {got.shape} != {expected.shape}")
        return
    d = float(np.abs(got - expected).max())
    if d <= tol:
        PASSED += 1
        print(f"  PASS  {name}: max|err|={d:.2e}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}: max|err|={d:.2e}")


def section(t):
    print(f"\n=== {t} ===")


def main():
    rng = np.random.default_rng(0)

    section("var backward")
    x = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
    leaf = make_leaf(x)
    out = _C_engine.var(leaf, [], False)  # scalar variance
    [g] = grad_of(out, [leaf])
    fd = fd_grad(lambda v: np.var(v), x)
    check("var(all)", g, fd)

    leaf = make_leaf(x)
    out = _C_engine.sum(_C_engine.var(leaf, [1], False), [], False)  # sum-reduce so scalar
    [g] = grad_of(out, [leaf])
    fd = fd_grad(lambda v: np.var(v, axis=1).sum(), x)
    check("var(axis=1)", g, fd)

    section("trace backward")
    M = rng.standard_normal((4, 4)).astype(np.float32) * 0.5
    leaf = make_leaf(M)
    out = _C_engine.trace(leaf)  # scalar
    [g] = grad_of(out, [leaf])
    fd = fd_grad(lambda v: float(np.trace(v)), M)
    check("trace(2-D)", g, fd)

    section("cumsum backward")
    x = rng.standard_normal((3, 5)).astype(np.float32) * 0.5
    leaf = make_leaf(x)
    out = _C_engine.sum(_C_engine.cumsum(leaf, 1), [], False)
    [g] = grad_of(out, [leaf])
    fd = fd_grad(lambda v: np.cumsum(v, axis=1).sum(), x)
    check("cumsum(axis=1)", g, fd)

    leaf = make_leaf(x)
    out = _C_engine.sum(_C_engine.cumsum(leaf, 0), [], False)
    [g] = grad_of(out, [leaf])
    fd = fd_grad(lambda v: np.cumsum(v, axis=0).sum(), x)
    check("cumsum(axis=0)", g, fd)

    section("dot 1-D backward")
    a = rng.standard_normal(5).astype(np.float32) * 0.5
    b = rng.standard_normal(5).astype(np.float32) * 0.5
    la = make_leaf(a)
    lb = make_leaf(b)
    out = _C_engine.dot(la, lb)
    ga, gb = grad_of(out, [la, lb])
    fd_a = fd_grad(lambda v: float(np.dot(v, b)), a)
    fd_b = fd_grad(lambda v: float(np.dot(a, v)), b)
    check("dot(1-D) da", ga, fd_a)
    check("dot(1-D) db", gb, fd_b)

    section("dot 2-D backward")
    A = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
    B = rng.standard_normal((4, 2)).astype(np.float32) * 0.5
    lA = make_leaf(A)
    lB = make_leaf(B)
    out = _C_engine.sum(_C_engine.dot(lA, lB), [], False)
    gA, gB = grad_of(out, [lA, lB])
    fd_A = fd_grad(lambda v: float(np.dot(v, B).sum()), A)
    fd_B = fd_grad(lambda v: float(np.dot(A, v).sum()), B)
    check("dot(2-D) dA", gA, fd_A)
    check("dot(2-D) dB", gB, fd_B)

    section("outer backward")
    a = rng.standard_normal(4).astype(np.float32) * 0.5
    b = rng.standard_normal(3).astype(np.float32) * 0.5
    la = make_leaf(a)
    lb = make_leaf(b)
    out = _C_engine.sum(_C_engine.outer(la, lb), [], False)
    ga, gb = grad_of(out, [la, lb])
    fd_a = fd_grad(lambda v: float(np.outer(v, b).sum()), a)
    fd_b = fd_grad(lambda v: float(np.outer(a, v).sum()), b)
    check("outer da", ga, fd_a)
    check("outer db", gb, fd_b)

    print(f"\n--- TOTAL: {PASSED} passed, {FAILED} failed ---")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
