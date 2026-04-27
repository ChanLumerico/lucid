#!/usr/bin/env python3
"""Phase 4d-A — verify the CPU paths just added for shape/indexing utilities."""
import sys
import numpy as np

from lucid._C import engine as _C_engine

PASSED = 0
FAILED = 0


def to_np(t):
    return np.array(t.data_as_python())


def from_np(arr):
    return _C_engine.TensorImpl(arr)


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


def section(t):
    print(f"\n=== {t} ===")


def main():
    rng = np.random.default_rng(0)

    section("split_at (CPU)")
    a = np.arange(20, dtype=np.float32).reshape(4, 5)
    parts = _C_engine.split_at(from_np(a), [2], axis=1)
    np_parts = np.split(a, [2], axis=1)
    for i, (p, np_p) in enumerate(zip(parts, np_parts)):
        check(f"split_at[{i}]", to_np(p), np_p)

    section("tile (CPU)")
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    check("tile([2,3])", to_np(_C_engine.tile(from_np(x), [2, 3])),
          np.tile(x, (2, 3)))

    section("pad (CPU)")
    p = rng.standard_normal((2, 3)).astype(np.float32)
    check("pad constant=0", to_np(_C_engine.pad(from_np(p), [(1, 1), (2, 0)], 0.0)),
          np.pad(p, [(1, 1), (2, 0)], constant_values=0.0))
    check("pad constant=7", to_np(_C_engine.pad(from_np(p), [(0, 1), (1, 2)], 7.0)),
          np.pad(p, [(0, 1), (1, 2)], constant_values=7.0))

    section("roll (CPU)")
    r = np.arange(12, dtype=np.float32).reshape(3, 4)
    check("roll(shift=2,axis=1)",
          to_np(_C_engine.roll(from_np(r), [2], [1])), np.roll(r, 2, axis=1))
    check("roll(multi-axis)",
          to_np(_C_engine.roll(from_np(r), [1, 2], [0, 1])),
          np.roll(r, (1, 2), axis=(0, 1)))

    section("gather (CPU)")
    g = rng.standard_normal((3, 5)).astype(np.float32)
    idx = np.array([[2, 0, 1, 4, 3], [1, 1, 0, 2, 4], [4, 3, 2, 1, 0]],
                   dtype=np.int32)
    check("gather(axis=1)",
          to_np(_C_engine.gather(from_np(g), from_np(idx), 1)),
          np.take_along_axis(g, idx.astype(np.int64), axis=1))

    section("diagonal (CPU)")
    d3 = rng.standard_normal((4, 5)).astype(np.float32)
    check("diagonal(offset=0)",
          to_np(_C_engine.diagonal(from_np(d3), 0, -2, -1)),
          np.diagonal(d3, 0, -2, -1))
    check("diagonal(offset=1)",
          to_np(_C_engine.diagonal(from_np(d3), 1, -2, -1)),
          np.diagonal(d3, 1, -2, -1))
    d4 = rng.standard_normal((2, 4, 5)).astype(np.float32)
    check("diagonal(batched)",
          to_np(_C_engine.diagonal(from_np(d4), 0, -2, -1)),
          np.diagonal(d4, 0, -2, -1))

    section("sort / argsort (CPU)")
    s = rng.standard_normal((3, 5)).astype(np.float32)
    check("sort(axis=1)",
          to_np(_C_engine.sort(from_np(s), 1)), np.sort(s, axis=1))
    check("sort(axis=0)",
          to_np(_C_engine.sort(from_np(s), 0)), np.sort(s, axis=0))
    check("argsort(axis=1)",
          to_np(_C_engine.argsort(from_np(s), 1)).astype(np.int64),
          np.argsort(s, axis=1).astype(np.int64), exact=True)

    section("topk (CPU)")
    tk = np.array([[1.0, 5.0, 2.0, 8.0, 3.0]], dtype=np.float32)
    out = to_np(_C_engine.topk(from_np(tk), 3, 1))
    expected = np.sort(tk, axis=1)[:, ::-1][:, :3]
    check("topk(k=3)", out, expected)

    section("meshgrid (CPU)")
    xv = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    yv = np.array([10.0, 20.0], dtype=np.float32)
    a_eng, b_eng = _C_engine.meshgrid([from_np(xv), from_np(yv)], indexing_xy=False)
    a_np, b_np = np.meshgrid(xv, yv, indexing="ij")
    check("meshgrid ij A", to_np(a_eng), a_np)
    check("meshgrid ij B", to_np(b_eng), b_np)
    a2_eng, b2_eng = _C_engine.meshgrid([from_np(xv), from_np(yv)], indexing_xy=True)
    a2_np, b2_np = np.meshgrid(xv, yv, indexing="xy")
    check("meshgrid xy A", to_np(a2_eng), a2_np)
    check("meshgrid xy B", to_np(b2_eng), b2_np)

    section("tensordot (CPU)")
    A = rng.standard_normal((3, 4, 5)).astype(np.float32)
    B = rng.standard_normal((5, 4, 6)).astype(np.float32)
    check("tensordot axes_a=[2],axes_b=[0]",
          to_np(_C_engine.tensordot(from_np(A), from_np(B), [2], [0])),
          np.tensordot(A, B, axes=([2], [0])), tol=1e-3)
    A2 = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    B2 = rng.standard_normal((4, 5, 6)).astype(np.float32)
    check("tensordot axes_a=[2,3],axes_b=[0,1]",
          to_np(_C_engine.tensordot(from_np(A2), from_np(B2), [2, 3], [0, 1])),
          np.tensordot(A2, B2, axes=([2, 3], [0, 1])), tol=1e-3)

    print(f"\n--- TOTAL: {PASSED} passed, {FAILED} failed ---")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
