#!/usr/bin/env python3
"""Phase 4f — verify newly-added ops:
    floordiv, _like family, expand_dims, ravel,
    nonzero, unique, histogram*,
    in-place variants (~30).
"""
import sys
import numpy as np

from lucid._C import engine as eng

PASSED = 0
FAILED = 0


def to_np(t):
    return np.array(t.data_as_python())


def from_np(arr, requires_grad=False):
    return eng.TensorImpl(arr.copy(), eng.Device.CPU, requires_grad)


def check(name, got, expected, tol=1e-4, exact=False):
    global PASSED, FAILED
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.shape != expected.shape:
        FAILED += 1
        print(f"  FAIL  {name}: shape {got.shape} != {expected.shape}")
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


def section(t):
    print(f"\n=== {t} ===")


def main():
    rng = np.random.default_rng(7)

    section("floordiv")
    a = np.array([7.0, 8.0, 9.0, -3.0], dtype=np.float32)
    b = np.array([2.0, 3.0, 4.0, 2.0],  dtype=np.float32)
    check("floordiv f32",
          to_np(eng.floordiv(from_np(a), from_np(b))),
          np.floor(a / b).astype(np.int64), exact=True)

    section("_like family")
    x = rng.standard_normal((2, 3)).astype(np.float32)
    X = from_np(x)
    z = eng.zeros_like(X)
    o = eng.ones_like(X)
    e = eng.empty_like(X)
    f = eng.full_like(X, 7.0)
    check("zeros_like shape", to_np(z), np.zeros_like(x))
    check("ones_like sum", float(to_np(o).sum()), float(np.ones_like(x).sum()))
    check("empty_like shape", np.array(e.shape), np.array(x.shape), exact=True)
    check("full_like value", to_np(f), np.full_like(x, 7.0))

    section("expand_dims, ravel")
    x = rng.standard_normal((3, 4)).astype(np.float32)
    X = from_np(x)
    check("expand_dims(0)", to_np(eng.expand_dims(X, 0)), x[None, ...])
    check("expand_dims(-1)", to_np(eng.expand_dims(X, -1)), x[..., None])
    check("ravel", to_np(eng.ravel(X)), x.reshape(-1))

    section("nonzero")
    nz = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32)
    NZ = from_np(nz)
    out = to_np(eng.nonzero(NZ))
    expected = np.argwhere(nz).astype(np.int64)
    check("nonzero", out, expected, exact=True)

    section("unique")
    u = np.array([3, 1, 2, 1, 4, 3, 5, 2], dtype=np.int32)
    out = to_np(eng.unique(from_np(u)))
    check("unique", out, np.array([1, 2, 3, 4, 5], dtype=np.int32), exact=True)

    section("histogram")
    x = rng.standard_normal(1000).astype(np.float32)
    counts_eng, edges_eng = eng.histogram(from_np(x), 10, -3.0, 3.0, False)
    counts_np, edges_np = np.histogram(x, bins=10, range=(-3.0, 3.0))
    check("histogram counts",
          to_np(counts_eng).astype(np.int64),
          counts_np.astype(np.int64), exact=True)
    check("histogram edges", to_np(edges_eng), edges_np.astype(np.float64),
          tol=1e-6)

    counts_eng, edges_eng = eng.histogram2d(
        from_np(x[:500]), from_np(x[500:1000]), 5, 5,
        -3.0, 3.0, -3.0, 3.0, False)
    counts_np, ex_np, ey_np = np.histogram2d(
        x[:500], x[500:1000], bins=[5, 5],
        range=[[-3.0, 3.0], [-3.0, 3.0]])
    check("histogram2d counts",
          to_np(counts_eng).astype(np.int64),
          counts_np.astype(np.int64), exact=True)

    # histogramdd
    multi = rng.standard_normal((500, 2)).astype(np.float32)
    counts_eng, _ = eng.histogramdd(
        from_np(multi), [4, 4], [(-3.0, 3.0), (-3.0, 3.0)], False)
    counts_np, _ = np.histogramdd(multi, bins=[4, 4],
                                   range=[[-3.0, 3.0], [-3.0, 3.0]])
    check("histogramdd counts",
          to_np(counts_eng).astype(np.int64),
          counts_np.astype(np.int64), exact=True)

    section("in-place binary (add_, sub_, mul_, div_, max_, min_)")
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    A = from_np(a); B = from_np(b)
    eng.add_(A, B)
    check("add_ result", to_np(A), a + b)
    A = from_np(a)
    eng.sub_(A, B)
    check("sub_ result", to_np(A), a - b)
    A = from_np(a)
    eng.mul_(A, B)
    check("mul_ result", to_np(A), a * b)

    section("in-place unary (neg_, exp_, square_, sqrt_, etc.)")
    x = rng.standard_normal((2, 3)).astype(np.float32)
    X = from_np(x)
    eng.neg_(X)
    check("neg_", to_np(X), -x)
    X = from_np(np.abs(x) + 0.5)
    eng.sqrt_(X)
    check("sqrt_", to_np(X), np.sqrt(np.abs(x) + 0.5))
    X = from_np(x)
    eng.square_(X)
    check("square_", to_np(X), x * x)
    X = from_np(x)
    eng.exp_(X)
    check("exp_", to_np(X), np.exp(x))
    X = from_np(x)
    eng.tanh_(X)
    check("tanh_", to_np(X), np.tanh(x))

    section("in-place version bumping")
    A = from_np(rng.standard_normal((3,)).astype(np.float32), requires_grad=False)
    v0 = A.version
    eng.neg_(A)
    v1 = A.version
    eng.exp_(A)
    v2 = A.version
    check("version bumps", np.array([v0, v1, v2]),
          np.array([0, 1, 2], dtype=np.int64), exact=True)

    print(f"\n--- TOTAL: {PASSED} passed, {FAILED} failed ---")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
