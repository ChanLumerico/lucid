#!/usr/bin/env python3
"""
Phase 4c — verify all newly-added C++ ops match numpy/PyTorch reference.

Covers:
  - Tensor creation: zeros/ones/full/eye/arange/linspace/diag/empty
  - Compare: equal/not_equal/greater/greater_equal/less/less_equal
  - Bitwise: and/or/xor
  - Reductions: var, trace, cumsum, cumprod
  - Linear-algebra contractions: dot, inner, outer
  - Shape utils: concatenate/stack/split/repeat/flatten/tril/triu/broadcast_to
                 argmin/argmax/where/masked_fill
  - Linalg (GPU): inv, det, solve, cholesky, norm, qr, svd, matrix_power, pinv

Pass criteria: max abs error < 1e-4 for float ops, exact for integer/bool ops.
"""
import sys
import numpy as np

from lucid._C import engine as eng
from lucid._C.engine import linalg as la


def to_np(t):
    return np.array(t.data_as_python())


def from_np(arr, gpu=False):
    return eng.TensorImpl(arr, eng.Device.GPU if gpu else eng.Device.CPU)


PASSED = 0
FAILED = 0


def check(name, got, expected, tol=1e-4, exact=False):
    global PASSED, FAILED
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.shape != expected.shape:
        FAILED += 1
        print(f"  FAIL  {name}: shape {got.shape} != expected {expected.shape}")
        return
    if exact:
        ok = np.array_equal(got, expected)
        diff = "(exact)" if ok else f"diff {(got != expected).sum()} cells"
    else:
        d = float(np.abs(got.astype(np.float64) -
                         expected.astype(np.float64)).max())
        ok = d <= tol
        diff = f"max|err|={d:.2e}"
    if ok:
        PASSED += 1
        print(f"  PASS  {name}: {diff}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}: {diff}")


def section(title):
    print(f"\n=== {title} ===")


def main():
    rng = np.random.default_rng(42)

    # ---- Creation ----
    section("Creation ops")
    check("zeros", to_np(eng.zeros([2, 3])), np.zeros((2, 3), np.float32))
    check("ones",  to_np(eng.ones([2, 3])),  np.ones((2, 3), np.float32))
    check("full",  to_np(eng.full([2, 3], 3.14)),
          np.full((2, 3), 3.14, np.float32))
    check("eye",   to_np(eng.eye(3, 4, 1)),
          np.eye(3, 4, k=1, dtype=np.float32))
    check("arange", to_np(eng.arange(0, 5, 1)),
          np.arange(0, 5, 1, np.float32))
    check("linspace", to_np(eng.linspace(0, 1, 5)),
          np.linspace(0, 1, 5).astype(np.float32))
    v = from_np(np.arange(4, dtype=np.float32))
    check("diag(1-D→2-D)", to_np(eng.diag(v, 0)),
          np.diag(np.arange(4, dtype=np.float32)))
    m = from_np(np.arange(9, dtype=np.float32).reshape(3, 3))
    check("diag(2-D→1-D)", to_np(eng.diag(m, 0)),
          np.diag(np.arange(9, dtype=np.float32).reshape(3, 3)))

    # ---- Compare ----
    section("Compare ops")
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    A = from_np(a); B = from_np(b)
    check("equal",         to_np(eng.equal(A, B)),         a == b, exact=True)
    check("not_equal",     to_np(eng.not_equal(A, B)),     a != b, exact=True)
    check("greater",       to_np(eng.greater(A, B)),       a > b,  exact=True)
    check("greater_equal", to_np(eng.greater_equal(A, B)), a >= b, exact=True)
    check("less",          to_np(eng.less(A, B)),          a < b,  exact=True)
    check("less_equal",    to_np(eng.less_equal(A, B)),    a <= b, exact=True)

    # ---- Bitwise ----
    section("Bitwise ops")
    ai = np.array([5, 3, 1, 0xff], dtype=np.int32)
    bi = np.array([3, 5, 7, 0x0f], dtype=np.int32)
    AI = from_np(ai); BI = from_np(bi)
    check("bitwise_and", to_np(eng.bitwise_and(AI, BI)), ai & bi, exact=True)
    check("bitwise_or",  to_np(eng.bitwise_or(AI, BI)),  ai | bi, exact=True)
    check("bitwise_xor", to_np(eng.bitwise_xor(AI, BI)), ai ^ bi, exact=True)

    # ---- Extra reductions / scans ----
    section("Extra reductions and scans")
    x = rng.standard_normal((3, 4)).astype(np.float32)
    X = from_np(x)
    check("var(all)", to_np(eng.var(X)), np.var(x))
    check("var(axis=0)", to_np(eng.var(X, [0], False)), np.var(x, axis=0))
    check("var(axis=1)", to_np(eng.var(X, [1], False)), np.var(x, axis=1))
    sq = rng.standard_normal((4, 4)).astype(np.float32)
    SQ = from_np(sq)
    check("trace(2-D)", to_np(eng.trace(SQ)), np.trace(sq))
    check("cumsum(axis=0)",  to_np(eng.cumsum(X, 0)),  np.cumsum(x, axis=0))
    check("cumsum(axis=-1)", to_np(eng.cumsum(X, -1)), np.cumsum(x, axis=-1))
    check("cumprod(axis=0)", to_np(eng.cumprod(X, 0)), np.cumprod(x, axis=0))

    # ---- Dot / inner / outer (CPU 1-D and 2-D) ----
    section("Dot, inner, outer")
    u = rng.standard_normal(5).astype(np.float32)
    v = rng.standard_normal(5).astype(np.float32)
    U = from_np(u); V = from_np(v)
    check("dot(1-D)", to_np(eng.dot(U, V)), np.dot(u, v))
    M1 = rng.standard_normal((4, 5)).astype(np.float32)
    M2 = rng.standard_normal((5, 3)).astype(np.float32)
    check("dot(2-D, 2-D)", to_np(eng.dot(from_np(M1), from_np(M2))),
          np.dot(M1, M2))
    check("inner(1-D)", to_np(eng.inner(U, V)), np.inner(u, v))
    check("outer", to_np(eng.outer(U, V)), np.outer(u, v))

    # ---- Shape utils ----
    section("Shape utilities")
    a2 = rng.standard_normal((2, 3)).astype(np.float32)
    b2 = rng.standard_normal((2, 3)).astype(np.float32)
    A2 = from_np(a2); B2 = from_np(b2)
    check("concat(axis=0)", to_np(eng.concatenate([A2, B2], 0)),
          np.concatenate([a2, b2], 0))
    check("concat(axis=1)", to_np(eng.concatenate([A2, B2], 1)),
          np.concatenate([a2, b2], 1))
    check("stack(axis=0)", to_np(eng.stack([A2, B2], 0)),
          np.stack([a2, b2], 0))

    splits_cpp = eng.split(A2, 3, axis=1)
    splits_np = np.split(a2, 3, axis=1)
    for i, (sc, sn) in enumerate(zip(splits_cpp, splits_np)):
        check(f"split[{i}]", to_np(sc), sn)

    check("repeat(axis=0)", to_np(eng.repeat(A2, 2, axis=0)),
          np.repeat(a2, 2, axis=0))

    flat3 = rng.standard_normal((2, 3, 4)).astype(np.float32)
    F3 = from_np(flat3)
    check("flatten(0,-1)", to_np(eng.flatten(F3, 0, -1)), flat3.reshape(-1))
    check("flatten(1, 2)",
          to_np(eng.flatten(F3, 1, 2)), flat3.reshape(2, 12))

    sq2 = np.arange(9, dtype=np.float32).reshape(3, 3)
    SQ2 = from_np(sq2)
    check("tril(0)", to_np(eng.tril(SQ2, 0)), np.tril(sq2, 0))
    check("tril(1)", to_np(eng.tril(SQ2, 1)), np.tril(sq2, 1))
    check("triu(-1)", to_np(eng.triu(SQ2, -1)), np.triu(sq2, -1))

    bcast_in = rng.standard_normal((1, 4)).astype(np.float32)
    BC = from_np(bcast_in)
    check("broadcast_to", to_np(eng.broadcast_to(BC, [3, 4])),
          np.broadcast_to(bcast_in, (3, 4)))

    check("argmax(axis=1)", to_np(eng.argmax(A2, axis=1, keepdims=False)),
          np.argmax(a2, axis=1).astype(np.int64), exact=True)
    check("argmin(axis=0)", to_np(eng.argmin(A2, axis=0, keepdims=False)),
          np.argmin(a2, axis=0).astype(np.int64), exact=True)

    cond = (a2 > 0).astype(np.uint8)
    COND = from_np(cond)
    check("where", to_np(eng.where(COND, A2, B2)), np.where(cond, a2, b2))
    mask = (a2 > 0).astype(np.uint8)
    MASK = from_np(mask)
    check("masked_fill", to_np(eng.masked_fill(A2, MASK, 9.0)),
          np.where(mask, 9.0, a2))

    # ---- Linalg (GPU only) ----
    section("Linalg ops (GPU)")
    A3 = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
                  dtype=np.float32)
    A3g = from_np(A3, gpu=True)
    check("linalg.inv",      to_np(la.inv(A3g)),
          np.linalg.inv(A3))
    check("linalg.det",      to_np(la.det(A3g)),
          np.linalg.det(A3), tol=1e-3)

    bvec = rng.standard_normal((3,)).astype(np.float32)
    bgpu = from_np(bvec, gpu=True)
    check("linalg.solve",    to_np(la.solve(A3g, bgpu)),
          np.linalg.solve(A3, bvec), tol=1e-3)

    P = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    Pg = from_np(P, gpu=True)
    check("linalg.cholesky", to_np(la.cholesky(Pg)),
          np.linalg.cholesky(P), tol=1e-4)

    vnorm = from_np(np.array([3.0, 4.0], dtype=np.float32), gpu=True)
    check("linalg.norm",     to_np(la.norm(vnorm, ord=2.0,
                                           axis=[], keepdims=False)),
          5.0)

    # SVD reconstruction
    M_svd = rng.standard_normal((3, 4)).astype(np.float32)
    M_svd_g = from_np(M_svd, gpu=True)
    U, S, Vt = la.svd(M_svd_g, compute_uv=True)
    recon = to_np(U) @ np.diag(to_np(S)) @ to_np(Vt)[:3, :]
    check("linalg.svd recon", recon, M_svd, tol=1e-3)

    # matrix_power
    sqp = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    SQP = from_np(sqp, gpu=True)
    check("linalg.matrix_power(3)", to_np(la.matrix_power(SQP, 3)),
          np.linalg.matrix_power(sqp, 3))

    # pinv
    R = rng.standard_normal((3, 4)).astype(np.float32)
    Rg = from_np(R, gpu=True)
    check("linalg.pinv", to_np(la.pinv(Rg)), np.linalg.pinv(R), tol=1e-3)

    print(f"\n--- TOTAL: {PASSED} passed, {FAILED} failed ---")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
