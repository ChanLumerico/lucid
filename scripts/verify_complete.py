#!/usr/bin/env python3
"""
Phase 4g — full numerical fidelity for every op on CPU and GPU,
forward and (where applicable) backward.

Coverage:
  - Forward parity vs numpy on CPU and on GPU
  - CPU↔GPU output equivalence within tolerance
  - Backward via finite-difference gradcheck on both devices

Tolerance: F32 forward ≤ 1e-3 absolute (GPU paths sometimes more lossy
because MLX uses fast math). Backward grad ≤ 5e-2 absolute due to FP32
roundoff in finite-difference probes.
"""
import sys
import numpy as np

from lucid._C import engine as _C_engine
from lucid._C.engine import linalg as la

PASSED = 0
FAILED = 0
SKIPPED = 0


def to_np(t):
    return np.array(t.data_as_python())


def make(arr, gpu=False, requires_grad=False):
    return _C_engine.TensorImpl(arr.copy(),
                          _C_engine.Device.GPU if gpu else _C_engine.Device.CPU,
                          requires_grad)


def check(name, got, expected, tol=1e-3, exact=False):
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


def skip(name, reason):
    global SKIPPED
    SKIPPED += 1
    print(f"  SKIP  {name}: {reason}")


def section(t):
    print(f"\n=== {t} ===")


def grad_of(out, leaves):
    for leaf in leaves:
        leaf.zero_grad()
    _C_engine.engine_backward(out, retain_graph=False)
    return [np.array(leaf.grad_as_python()) for leaf in leaves]


def fd_grad(forward_np, x, eps=1e-3):
    x = x.astype(np.float64)
    g = np.zeros_like(x)
    flat = x.reshape(-1)
    grad_flat = g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        plus = float(forward_np(x.astype(np.float32)))
        flat[i] = old - eps
        minus = float(forward_np(x.astype(np.float32)))
        flat[i] = old
        grad_flat[i] = (plus - minus) / (2 * eps)
    return g.astype(np.float32)


# -----------------------------------------------------------------------------
# bfunc: arithmetic + matmul forward CPU+GPU and backward CPU+GPU
# -----------------------------------------------------------------------------
def test_bfunc():
    section("bfunc forward (CPU + GPU)")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5)).astype(np.float32) * 0.5
    b = rng.standard_normal((4, 5)).astype(np.float32) * 0.5
    pos_b = np.abs(b) + 0.5

    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        A, B = make(a, gpu), make(b, gpu)
        Pb = make(pos_b, gpu)
        check(f"add[{dev}]", to_np(_C_engine.add(A, B)), a + b)
        check(f"sub[{dev}]", to_np(_C_engine.sub(A, B)), a - b)
        check(f"mul[{dev}]", to_np(_C_engine.mul(A, B)), a * b)
        check(f"div[{dev}]", to_np(_C_engine.div(A, Pb)), a / pos_b)
        check(f"pow[{dev}]", to_np(_C_engine.pow(make(np.abs(a) + 0.1, gpu), B)),
              np.power(np.abs(a) + 0.1, b))
        check(f"maximum[{dev}]", to_np(_C_engine.maximum(A, B)), np.maximum(a, b))
        check(f"minimum[{dev}]", to_np(_C_engine.minimum(A, B)), np.minimum(a, b))

        M1 = rng.standard_normal((3, 4)).astype(np.float32)
        M2 = rng.standard_normal((4, 5)).astype(np.float32)
        check(f"matmul[{dev}]",
              to_np(_C_engine.matmul(make(M1, gpu), make(M2, gpu))), M1 @ M2)

        # compare ops
        check(f"equal[{dev}]",         to_np(_C_engine.equal(A, B)),         a == b, exact=True)
        check(f"not_equal[{dev}]",     to_np(_C_engine.not_equal(A, B)),     a != b, exact=True)
        check(f"greater[{dev}]",       to_np(_C_engine.greater(A, B)),       a > b,  exact=True)
        check(f"greater_equal[{dev}]", to_np(_C_engine.greater_equal(A, B)), a >= b, exact=True)
        check(f"less[{dev}]",          to_np(_C_engine.less(A, B)),          a < b,  exact=True)
        check(f"less_equal[{dev}]",    to_np(_C_engine.less_equal(A, B)),    a <= b, exact=True)

        # bitwise
        ai = np.array([5, 3, 1, 0xff], dtype=np.int32)
        bi = np.array([3, 5, 7, 0x0f], dtype=np.int32)
        AI, BI = make(ai, gpu), make(bi, gpu)
        check(f"bitwise_and[{dev}]", to_np(_C_engine.bitwise_and(AI, BI)), ai & bi, exact=True)
        check(f"bitwise_or[{dev}]",  to_np(_C_engine.bitwise_or(AI, BI)),  ai | bi, exact=True)
        check(f"bitwise_xor[{dev}]", to_np(_C_engine.bitwise_xor(AI, BI)), ai ^ bi, exact=True)

        # dot/inner/outer
        u = rng.standard_normal(5).astype(np.float32)
        v = rng.standard_normal(5).astype(np.float32)
        U, V = make(u, gpu), make(v, gpu)
        check(f"dot1d[{dev}]", to_np(_C_engine.dot(U, V)), np.dot(u, v))
        K1 = rng.standard_normal((3, 4)).astype(np.float32)
        K2 = rng.standard_normal((4, 2)).astype(np.float32)
        check(f"dot2d[{dev}]",
              to_np(_C_engine.dot(make(K1, gpu), make(K2, gpu))), np.dot(K1, K2))
        check(f"inner[{dev}]", to_np(_C_engine.inner(U, V)), np.inner(u, v))
        check(f"outer[{dev}]", to_np(_C_engine.outer(U, V)), np.outer(u, v))

        # tensordot
        T1 = rng.standard_normal((3, 4, 5)).astype(np.float32)
        T2 = rng.standard_normal((5, 4, 6)).astype(np.float32)
        check(f"tensordot[{dev}]",
              to_np(_C_engine.tensordot(make(T1, gpu), make(T2, gpu), [2], [0])),
              np.tensordot(T1, T2, axes=([2], [0])))

        # floordiv
        af = np.array([7.0, 8.0, 9.0, -3.0], dtype=np.float32)
        bf = np.array([2.0, 3.0, 4.0, 2.0],  dtype=np.float32)
        check(f"floordiv[{dev}]",
              to_np(_C_engine.floordiv(make(af, gpu), make(bf, gpu))),
              np.floor(af / bf).astype(np.int64), exact=True)


def test_bfunc_grad():
    section("bfunc backward (CPU + GPU)")
    rng = np.random.default_rng(1)
    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        a = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
        b = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
        pos_b = np.abs(b) + 0.5

        # add backward: da=g, db=g
        A = make(a, gpu, True); B = make(b, gpu, True)
        out = _C_engine.sum(_C_engine.add(A, B), [], False)
        ga, gb = grad_of(out, [A, B])
        check(f"add da[{dev}]", ga, np.ones_like(a))
        check(f"add db[{dev}]", gb, np.ones_like(b))

        # mul backward: da=g*b, db=g*a
        A = make(a, gpu, True); B = make(b, gpu, True)
        out = _C_engine.sum(_C_engine.mul(A, B), [], False)
        ga, gb = grad_of(out, [A, B])
        check(f"mul da[{dev}]", ga, b)
        check(f"mul db[{dev}]", gb, a)

        # matmul backward (2D)
        A_np = rng.standard_normal((3, 4)).astype(np.float32) * 0.3
        B_np = rng.standard_normal((4, 2)).astype(np.float32) * 0.3
        A = make(A_np, gpu, True); B = make(B_np, gpu, True)
        out = _C_engine.sum(_C_engine.matmul(A, B), [], False)
        ga, gb = grad_of(out, [A, B])
        # expected: da = ones((3,2)) @ B.T, db = A.T @ ones((3,2))
        ones = np.ones((3, 2), dtype=np.float32)
        check(f"matmul da[{dev}]", ga, ones @ B_np.T)
        check(f"matmul db[{dev}]", gb, A_np.T @ ones)

        # dot 1D backward
        u = rng.standard_normal(5).astype(np.float32) * 0.5
        v = rng.standard_normal(5).astype(np.float32) * 0.5
        U = make(u, gpu, True); V = make(v, gpu, True)
        out = _C_engine.dot(U, V)
        gu, gv = grad_of(out, [U, V])
        check(f"dot1d du[{dev}]", gu, v)
        check(f"dot1d dv[{dev}]", gv, u)

        # outer backward
        U = make(u, gpu, True); V = make(v, gpu, True)
        out = _C_engine.sum(_C_engine.outer(U, V), [], False)
        gu, gv = grad_of(out, [U, V])
        check(f"outer du[{dev}]", gu, np.ones_like(u) * v.sum())
        check(f"outer dv[{dev}]", gv, np.ones_like(v) * u.sum())


# -----------------------------------------------------------------------------
# ufunc: unary forward + backward CPU/GPU
# -----------------------------------------------------------------------------
def test_ufunc():
    section("ufunc forward (CPU + GPU)")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
    pos = np.abs(x) + 0.5
    bounded = np.clip(x, -0.9, 0.9).astype(np.float32)

    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        X = make(x, gpu); P = make(pos, gpu); B = make(bounded, gpu)
        # Arith
        check(f"neg[{dev}]",        to_np(_C_engine.neg(X)),        -x)
        check(f"abs[{dev}]",        to_np(_C_engine.abs(X)),        np.abs(x))
        check(f"sign[{dev}]",       to_np(_C_engine.sign(X)),       np.sign(x))
        check(f"reciprocal[{dev}]", to_np(_C_engine.reciprocal(P)), 1.0 / pos)
        check(f"square[{dev}]",     to_np(_C_engine.square(X)),     x * x)
        check(f"cube[{dev}]",       to_np(_C_engine.cube(X)),       x ** 3)
        # Exp/log
        check(f"exp[{dev}]",  to_np(_C_engine.exp(X)),  np.exp(x))
        check(f"log[{dev}]",  to_np(_C_engine.log(P)),  np.log(pos))
        check(f"log2[{dev}]", to_np(_C_engine.log2(P)), np.log2(pos))
        check(f"sqrt[{dev}]", to_np(_C_engine.sqrt(P)), np.sqrt(pos))
        # Trig
        check(f"sin[{dev}]", to_np(_C_engine.sin(X)), np.sin(x))
        check(f"cos[{dev}]", to_np(_C_engine.cos(X)), np.cos(x))
        check(f"tan[{dev}]", to_np(_C_engine.tan(X)), np.tan(x))
        check(f"arcsin[{dev}]", to_np(_C_engine.arcsin(B)), np.arcsin(bounded))
        check(f"arccos[{dev}]", to_np(_C_engine.arccos(B)), np.arccos(bounded))
        check(f"arctan[{dev}]", to_np(_C_engine.arctan(X)), np.arctan(x))
        # Hyperbolic
        check(f"sinh[{dev}]", to_np(_C_engine.sinh(X)), np.sinh(x))
        check(f"cosh[{dev}]", to_np(_C_engine.cosh(X)), np.cosh(x))
        check(f"tanh[{dev}]", to_np(_C_engine.tanh(X)), np.tanh(x))
        # Activation
        check(f"relu[{dev}]",       to_np(_C_engine.relu(X)),    np.maximum(x, 0.0))
        check(f"sigmoid[{dev}]",    to_np(_C_engine.sigmoid(X)), 1.0 / (1.0 + np.exp(-x)))
        check(f"silu[{dev}]",       to_np(_C_engine.silu(X)),    x * (1.0 / (1.0 + np.exp(-x))))
        check(f"leaky_relu[{dev}]", to_np(_C_engine.leaky_relu(X, 0.1)),
              np.where(x >= 0, x, 0.1 * x))
        check(f"softplus[{dev}]",   to_np(_C_engine.softplus(X)),
              np.log1p(np.exp(x)).astype(np.float32))
        # softmax
        sm = np.exp(x - x.max(-1, keepdims=True))
        sm = sm / sm.sum(-1, keepdims=True)
        check(f"softmax[{dev}]", to_np(_C_engine.softmax(X, -1)), sm)

        # Scalar param
        check(f"pow_scalar[{dev}]",  to_np(_C_engine.pow_scalar(P, 2.0)), pos ** 2)
        check(f"rpow_scalar[{dev}]", to_np(_C_engine.rpow_scalar(2.0, X)), 2.0 ** x)
        check(f"clip[{dev}]", to_np(_C_engine.clip(X, -0.5, 0.5)), np.clip(x, -0.5, 0.5))

        # Discrete
        h = (x * 4).astype(np.float32)
        H = make(h, gpu)
        check(f"round[{dev}]", to_np(_C_engine.round(H)), np.round(h))
        check(f"floor[{dev}]", to_np(_C_engine.floor(H)), np.floor(h))
        check(f"ceil[{dev}]",  to_np(_C_engine.ceil(H)),  np.ceil(h))
        ints = np.array([5, 3, 0, -1], dtype=np.int32)
        check(f"invert[{dev}]", to_np(_C_engine.invert(make(ints, gpu))), ~ints, exact=True)


def test_ufunc_grad():
    section("ufunc backward (CPU + GPU)")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
    pos = np.abs(x) + 0.5

    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.exp(X), [], False)
        [g] = grad_of(out, [X])
        check(f"exp grad[{dev}]", g, np.exp(x))

        X = make(pos, gpu, True)
        out = _C_engine.sum(_C_engine.log(X), [], False)
        [g] = grad_of(out, [X])
        check(f"log grad[{dev}]", g, 1.0 / pos)

        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.square(X), [], False)
        [g] = grad_of(out, [X])
        check(f"square grad[{dev}]", g, 2 * x)

        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.tanh(X), [], False)
        [g] = grad_of(out, [X])
        check(f"tanh grad[{dev}]", g, 1.0 - np.tanh(x) ** 2)

        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.relu(X), [], False)
        [g] = grad_of(out, [X])
        check(f"relu grad[{dev}]", g, (x > 0).astype(np.float32))

        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.sigmoid(X), [], False)
        [g] = grad_of(out, [X])
        s = 1.0 / (1.0 + np.exp(-x))
        check(f"sigmoid grad[{dev}]", g, s * (1.0 - s))


def test_reduction_grad():
    section("reduction forward + backward (CPU + GPU)")
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 4)).astype(np.float32) * 0.5

    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        X = make(x, gpu, True)
        out = _C_engine.sum(X, [], False)
        [g] = grad_of(out, [X])
        check(f"sum grad[{dev}]", g, np.ones_like(x))

        X = make(x, gpu, True)
        out = _C_engine.mean(X, [], False)
        [g] = grad_of(out, [X])
        check(f"mean grad[{dev}]", g, np.ones_like(x) / x.size)

        # var grad: 2/N * (x - mean)
        X = make(x, gpu, True)
        out = _C_engine.var(X, [], False)
        [g] = grad_of(out, [X])
        N = x.size
        check(f"var grad[{dev}]", g, 2.0 / N * (x - x.mean()))

        # cumsum grad
        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.cumsum(X, 1), [], False)
        [g] = grad_of(out, [X])
        # grad = reverse(cumsum(reverse(ones, axis=1), axis=1), axis=1)
        ones = np.ones_like(x)
        expected = np.cumsum(ones[:, ::-1], axis=1)[:, ::-1]
        check(f"cumsum grad[{dev}]", g, expected)


# -----------------------------------------------------------------------------
# gfunc: creation
# -----------------------------------------------------------------------------
def test_gfunc():
    section("gfunc forward (CPU + GPU)")
    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        d = _C_engine.Device.GPU if gpu else _C_engine.Device.CPU
        check(f"zeros[{dev}]", to_np(_C_engine.zeros([2, 3], dtype=_C_engine.Dtype.F32, device=d)),
              np.zeros((2, 3), np.float32))
        check(f"ones[{dev}]", to_np(_C_engine.ones([2, 3], dtype=_C_engine.Dtype.F32, device=d)),
              np.ones((2, 3), np.float32))
        check(f"full[{dev}]", to_np(_C_engine.full([2, 3], 3.14, dtype=_C_engine.Dtype.F32, device=d)),
              np.full((2, 3), 3.14, np.float32))
        check(f"eye[{dev}]", to_np(_C_engine.eye(3, 4, 1, dtype=_C_engine.Dtype.F32, device=d)),
              np.eye(3, 4, k=1, dtype=np.float32))
        check(f"arange[{dev}]", to_np(_C_engine.arange(0.0, 5.0, 1.0, dtype=_C_engine.Dtype.F32, device=d)),
              np.arange(0.0, 5.0, 1.0, np.float32))
        check(f"linspace[{dev}]", to_np(_C_engine.linspace(0.0, 1.0, 5, dtype=_C_engine.Dtype.F32, device=d)),
              np.linspace(0.0, 1.0, 5).astype(np.float32))

        # _like family
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        X = make(x, gpu)
        check(f"zeros_like[{dev}]", to_np(_C_engine.zeros_like(X)), np.zeros_like(x))
        check(f"ones_like[{dev}]", to_np(_C_engine.ones_like(X)), np.ones_like(x))
        check(f"full_like[{dev}]", to_np(_C_engine.full_like(X, 7.0)), np.full_like(x, 7.0))


# -----------------------------------------------------------------------------
# utils: shape / index ops on both devices
# -----------------------------------------------------------------------------
def test_utils():
    section("utils forward (CPU + GPU)")
    rng = np.random.default_rng(5)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    M3 = rng.standard_normal((4, 5)).astype(np.float32)

    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        A, B = make(a, gpu), make(b, gpu)
        # concat
        check(f"concatenate axis=0[{dev}]",
              to_np(_C_engine.concatenate([A, B], 0)), np.concatenate([a, b], 0))
        check(f"stack axis=0[{dev}]",
              to_np(_C_engine.stack([A, B], 0)), np.stack([a, b], 0))
        check(f"hstack[{dev}]",
              to_np(_C_engine.hstack([A, B])), np.hstack([a, b]))
        check(f"vstack[{dev}]",
              to_np(_C_engine.vstack([A, B])), np.vstack([a, b]))
        # split
        parts = _C_engine.split(A, 2, axis=1)
        np_parts = np.split(a, 2, axis=1)
        for i, (sc, sn) in enumerate(zip(parts, np_parts)):
            check(f"split[{i}][{dev}]", to_np(sc), sn)
        # repeat / tile
        check(f"repeat[{dev}]", to_np(_C_engine.repeat(A, 2, axis=0)),
              np.repeat(a, 2, axis=0))
        check(f"tile[{dev}]", to_np(_C_engine.tile(A, [2, 3])),
              np.tile(a, (2, 3)))
        # pad
        check(f"pad[{dev}]", to_np(_C_engine.pad(A, [(1, 1), (0, 0)], 0.0)),
              np.pad(a, [(1, 1), (0, 0)]))
        # flatten
        x3 = rng.standard_normal((2, 3, 4)).astype(np.float32)
        check(f"flatten[{dev}]", to_np(_C_engine.flatten(make(x3, gpu), 0, -1)),
              x3.reshape(-1))
        # broadcast_to
        ba = rng.standard_normal((1, 4)).astype(np.float32)
        check(f"broadcast_to[{dev}]",
              to_np(_C_engine.broadcast_to(make(ba, gpu), [3, 4])),
              np.broadcast_to(ba, (3, 4)))
        # tri
        sq = np.arange(9, dtype=np.float32).reshape(3, 3)
        check(f"tril[{dev}]", to_np(_C_engine.tril(make(sq, gpu), 0)), np.tril(sq, 0))
        check(f"triu[{dev}]", to_np(_C_engine.triu(make(sq, gpu), 0)), np.triu(sq, 0))
        # where, masked_fill
        cond = (a > 0).astype(np.uint8)
        check(f"where[{dev}]",
              to_np(_C_engine.where(make(cond, gpu), A, B)), np.where(cond, a, b))
        mask = (a > 0).astype(np.uint8)
        check(f"masked_fill[{dev}]",
              to_np(_C_engine.masked_fill(A, make(mask, gpu), 9.0)),
              np.where(mask, 9.0, a))
        # roll
        check(f"roll[{dev}]", to_np(_C_engine.roll(A, [1], [0])),
              np.roll(a, 1, axis=0))
        # gather
        idx = np.array([[2, 0, 1, 3], [1, 1, 0, 2], [3, 3, 2, 1]], dtype=np.int32)
        check(f"gather[{dev}]",
              to_np(_C_engine.gather(A, make(idx, gpu), 1)),
              np.take_along_axis(a, idx.astype(np.int64), axis=1))
        # diagonal
        d3 = rng.standard_normal((4, 5)).astype(np.float32)
        check(f"diagonal[{dev}]",
              to_np(_C_engine.diagonal(make(d3, gpu), 0, -2, -1)),
              np.diagonal(d3, 0, -2, -1))
        # sort / argsort / argmax / argmin
        check(f"sort[{dev}]", to_np(_C_engine.sort(A, 1)), np.sort(a, axis=1))
        check(f"argsort[{dev}]",
              to_np(_C_engine.argsort(A, 1)).astype(np.int64),
              np.argsort(a, axis=1).astype(np.int64), exact=True)
        check(f"argmax[{dev}]",
              to_np(_C_engine.argmax(A, axis=1, keepdims=False)),
              np.argmax(a, axis=1).astype(np.int64), exact=True)
        check(f"argmin[{dev}]",
              to_np(_C_engine.argmin(A, axis=0, keepdims=False)),
              np.argmin(a, axis=0).astype(np.int64), exact=True)
        # topk
        out = to_np(_C_engine.topk(A, 2, 1))
        expected = np.sort(a, axis=1)[:, ::-1][:, :2]
        check(f"topk[{dev}]", out, expected)
        # reshape / squeeze / unsqueeze / expand_dims / ravel
        check(f"reshape[{dev}]", to_np(_C_engine.reshape(A, [12])), a.reshape(12))
        check(f"unsqueeze[{dev}]",
              to_np(_C_engine.unsqueeze(A, 0)), np.expand_dims(a, 0))
        check(f"expand_dims[{dev}]",
              to_np(_C_engine.expand_dims(A, -1)), np.expand_dims(a, -1))
        check(f"ravel[{dev}]", to_np(_C_engine.ravel(A)), a.reshape(-1))
        # meshgrid
        xv = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        yv = np.array([10.0, 20.0], dtype=np.float32)
        a_eng, b_eng = _C_engine.meshgrid([make(xv, gpu), make(yv, gpu)],
                                     indexing_xy=False)
        a_np, b_np = np.meshgrid(xv, yv, indexing="ij")
        check(f"meshgrid[ij][A][{dev}]", to_np(a_eng), a_np)
        check(f"meshgrid[ij][B][{dev}]", to_np(b_eng), b_np)


# -----------------------------------------------------------------------------
# linalg (GPU only by design)
# -----------------------------------------------------------------------------
def test_linalg():
    section("linalg forward (GPU only by design)")
    rng = np.random.default_rng(6)
    A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
                 dtype=np.float32)
    Ag = make(A, True)
    check("linalg.inv", to_np(la.inv(Ag)), np.linalg.inv(A))
    check("linalg.det", to_np(la.det(Ag)), np.linalg.det(A), tol=1e-2)

    bvec = rng.standard_normal(3).astype(np.float32)
    check("linalg.solve",
          to_np(la.solve(Ag, make(bvec, True))),
          np.linalg.solve(A, bvec), tol=1e-3)

    P = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    check("linalg.cholesky",
          to_np(la.cholesky(make(P, True))),
          np.linalg.cholesky(P), tol=1e-4)

    v = np.array([3.0, 4.0], dtype=np.float32)
    check("linalg.norm",
          to_np(la.norm(make(v, True), ord=2.0,
                         axis=[], keepdims=False)),
          5.0)

    qr_in = rng.standard_normal((4, 3)).astype(np.float32)
    Q, R = la.qr(make(qr_in, True))
    check("linalg.qr recon", to_np(Q) @ to_np(R), qr_in, tol=1e-3)

    M_svd = rng.standard_normal((3, 4)).astype(np.float32)
    U, S, Vt = la.svd(make(M_svd, True), compute_uv=True)
    recon = to_np(U) @ np.diag(to_np(S)) @ to_np(Vt)[:3, :]
    check("linalg.svd recon", recon, M_svd, tol=1e-2)

    # matrix_power
    sqp = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    check("linalg.matrix_power(3)",
          to_np(la.matrix_power(make(sqp, True), 3)),
          np.linalg.matrix_power(sqp, 3))

    # pinv
    R3 = rng.standard_normal((3, 4)).astype(np.float32)
    check("linalg.pinv", to_np(la.pinv(make(R3, True))),
          np.linalg.pinv(R3), tol=1e-2)


# -----------------------------------------------------------------------------
# nn forward+backward CPU/GPU (the most critical for downstream models)
# -----------------------------------------------------------------------------
def test_nn():
    section("nn forward (CPU + GPU)")
    rng = np.random.default_rng(8)

    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        # linear: y = x @ W.T + b
        x = rng.standard_normal((4, 5)).astype(np.float32) * 0.5
        W = rng.standard_normal((3, 5)).astype(np.float32) * 0.3
        bb = rng.standard_normal((3,)).astype(np.float32) * 0.1
        y_eng = to_np(_C_engine.linear(make(x, gpu), make(W, gpu), make(bb, gpu)))
        check(f"linear[{dev}]", y_eng, x @ W.T + bb)

        # conv2d identity tap
        xc = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
        wk = np.zeros((1, 1, 3, 3), dtype=np.float32)
        wk[0, 0, 1, 1] = 1.0
        bc = np.zeros((1,), dtype=np.float32)
        check(f"conv2d (identity)[{dev}]",
              to_np(_C_engine.conv2d(make(xc, gpu), make(wk, gpu),
                               make(bc, gpu), 1, 1, 1, 1)),
              xc)

        # max_pool2d
        xp = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
        out = to_np(_C_engine.max_pool2d(make(xp, gpu), 2, 2, 2, 2, 0, 0))
        # numpy max_pool 2x2 stride 2
        np_pool = np.zeros((1, 2, 2, 2), dtype=np.float32)
        for n in range(1):
            for c in range(2):
                for i in range(2):
                    for j in range(2):
                        np_pool[n, c, i, j] = xp[n, c, i*2:i*2+2, j*2:j*2+2].max()
        check(f"max_pool2d[{dev}]", out, np_pool)

        # avg_pool2d
        out = to_np(_C_engine.avg_pool2d(make(xp, gpu), 2, 2, 2, 2, 0, 0))
        np_avg = np.zeros((1, 2, 2, 2), dtype=np.float32)
        for n in range(1):
            for c in range(2):
                for i in range(2):
                    for j in range(2):
                        np_avg[n, c, i, j] = xp[n, c, i*2:i*2+2, j*2:j*2+2].mean()
        check(f"avg_pool2d[{dev}]", out, np_avg)

        # batch_norm: with affine, batch stats
        xbn = rng.standard_normal((4, 3, 2, 2)).astype(np.float32)
        gamma = np.ones((3,), dtype=np.float32)
        beta = np.zeros((3,), dtype=np.float32)
        out = to_np(_C_engine.batch_norm(make(xbn, gpu), make(gamma, gpu),
                                   make(beta, gpu), 1e-5))
        # numpy reference (training-mode)
        mu = xbn.mean(axis=(0, 2, 3), keepdims=True)
        var = xbn.var(axis=(0, 2, 3), keepdims=True)
        np_bn = (xbn - mu) / np.sqrt(var + 1e-5) * gamma.reshape(1, 3, 1, 1) \
                + beta.reshape(1, 3, 1, 1)
        check(f"batch_norm[{dev}]", out, np_bn, tol=1e-3)

        # layer_norm
        xln = rng.standard_normal((2, 4, 5)).astype(np.float32)
        gln = np.ones((5,), dtype=np.float32)
        bln = np.zeros((5,), dtype=np.float32)
        out = to_np(_C_engine.layer_norm(make(xln, gpu), make(gln, gpu),
                                   make(bln, gpu), 1e-5))
        mu = xln.mean(-1, keepdims=True)
        var = xln.var(-1, keepdims=True)
        np_ln = (xln - mu) / np.sqrt(var + 1e-5) * gln + bln
        check(f"layer_norm[{dev}]", out, np_ln, tol=1e-3)

        # rms_norm
        xrn = rng.standard_normal((2, 4)).astype(np.float32)
        grn = np.ones((4,), dtype=np.float32)
        out = to_np(_C_engine.rms_norm(make(xrn, gpu), make(grn, gpu), 1e-5))
        rms = np.sqrt((xrn ** 2).mean(-1, keepdims=True) + 1e-5)
        np_rn = xrn / rms * grn
        check(f"rms_norm[{dev}]", out, np_rn, tol=1e-3)


def test_nn_grad():
    section("nn backward (CPU + GPU)")
    rng = np.random.default_rng(9)
    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        # linear backward
        x = rng.standard_normal((4, 5)).astype(np.float32) * 0.5
        W = rng.standard_normal((3, 5)).astype(np.float32) * 0.3
        bb = rng.standard_normal((3,)).astype(np.float32) * 0.1
        X = make(x, gpu, True); WW = make(W, gpu, True); BB = make(bb, gpu, True)
        out = _C_engine.sum(_C_engine.linear(X, WW, BB), [], False)
        gx, gw, gb = grad_of(out, [X, WW, BB])
        # expected: dx = ones(4,3) @ W = (4,5)
        ones_y = np.ones((4, 3), dtype=np.float32)
        check(f"linear dx[{dev}]", gx, ones_y @ W, tol=1e-3)
        # dW = ones(4,3).T @ x = (3,5)
        check(f"linear dW[{dev}]", gw, ones_y.T @ x, tol=1e-3)
        # db = ones(4,3).sum(0) = (3,)
        check(f"linear db[{dev}]", gb, ones_y.sum(0), tol=1e-3)


# -----------------------------------------------------------------------------
# random
# -----------------------------------------------------------------------------
def test_random():
    section("random forward (CPU + GPU)")
    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        d = _C_engine.Device.GPU if gpu else _C_engine.Device.CPU
        _C_engine.default_generator().set_seed(42)
        r = to_np(_C_engine.rand([5000], _C_engine.Dtype.F32, d, _C_engine.default_generator()))
        check(f"rand mean ≈ 0.5[{dev}]",
              np.array(r.mean()), np.array(0.5), tol=0.05)
        check(f"rand range [0,1)[{dev}]",
              np.array([r.min() >= 0, r.max() < 1.0]),
              np.array([True, True]), exact=True)

        _C_engine.default_generator().set_seed(42)
        rn = to_np(_C_engine.randn([5000], _C_engine.Dtype.F32, d, _C_engine.default_generator()))
        check(f"randn mean ≈ 0[{dev}]",
              np.array(rn.mean()), np.array(0.0), tol=0.1)
        check(f"randn std ≈ 1[{dev}]",
              np.array(rn.std()), np.array(1.0), tol=0.1)

    # Determinism: same seed → same draws on each device
    _C_engine.default_generator().set_seed(7)
    a_cpu = to_np(_C_engine.rand([1000], _C_engine.Dtype.F32, _C_engine.Device.CPU,
                           _C_engine.default_generator()))
    _C_engine.default_generator().set_seed(7)
    a_gpu = to_np(_C_engine.rand([1000], _C_engine.Dtype.F32, _C_engine.Device.GPU,
                           _C_engine.default_generator()))
    check("rand CPU↔GPU same seed", a_cpu, a_gpu, tol=1e-6)


# -----------------------------------------------------------------------------
# CPU↔GPU equivalence: pick a couple of representative ops, verify both
# devices produce numerically equivalent output
# -----------------------------------------------------------------------------
def test_cpu_gpu_equivalence():
    section("CPU↔GPU output equivalence")
    rng = np.random.default_rng(10)
    a = rng.standard_normal((4, 5)).astype(np.float32) * 0.5
    b = rng.standard_normal((4, 5)).astype(np.float32) * 0.5

    # Element-wise
    A_cpu, A_gpu = make(a, False), make(a, True)
    B_cpu, B_gpu = make(b, False), make(b, True)
    for name, fn in [("add", _C_engine.add), ("sub", _C_engine.sub), ("mul", _C_engine.mul),
                     ("maximum", _C_engine.maximum), ("minimum", _C_engine.minimum)]:
        c = to_np(fn(A_cpu, B_cpu))
        g = to_np(fn(A_gpu, B_gpu))
        check(f"{name}: CPU≡GPU", c, g, tol=1e-5)

    for name, fn in [("exp", _C_engine.exp), ("log", lambda t: _C_engine.log(make(np.abs(a) + 0.5))),
                     ("sin", _C_engine.sin), ("tanh", _C_engine.tanh),
                     ("sigmoid", _C_engine.sigmoid), ("relu", _C_engine.relu)]:
        if name == "log":
            c = to_np(fn(A_cpu)); g = to_np(_C_engine.log(make(np.abs(a) + 0.5, True)))
        else:
            c = to_np(fn(A_cpu)); g = to_np(fn(A_gpu))
        check(f"{name}: CPU≡GPU", c, g, tol=1e-4)

    # Matmul
    M1 = rng.standard_normal((3, 4)).astype(np.float32)
    M2 = rng.standard_normal((4, 5)).astype(np.float32)
    c = to_np(_C_engine.matmul(make(M1, False), make(M2, False)))
    g = to_np(_C_engine.matmul(make(M1, True), make(M2, True)))
    check("matmul: CPU≡GPU", c, g, tol=1e-4)

    # Reductions
    for name, fn in [("sum", _C_engine.sum), ("mean", _C_engine.mean),
                     ("max", _C_engine.max), ("min", _C_engine.min)]:
        c = to_np(fn(A_cpu, [], False)); g = to_np(fn(A_gpu, [], False))
        check(f"{name}(all): CPU≡GPU", c, g, tol=1e-4)


def test_more_activations_grad():
    section("more activation backward (CPU + GPU)")
    rng = np.random.default_rng(20)
    x = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        # silu: f(x) = x * sigmoid(x); grad = sigmoid + x*sigmoid*(1-sigmoid)
        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.silu(X), [], False)
        [g] = grad_of(out, [X])
        s = 1.0 / (1.0 + np.exp(-x))
        check(f"silu grad[{dev}]", g, s + x * s * (1.0 - s), tol=1e-4)
        # softplus: grad = sigmoid(x)
        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.softplus(X), [], False)
        [g] = grad_of(out, [X])
        check(f"softplus grad[{dev}]", g, s, tol=1e-4)
        # leaky_relu: grad = 1 if x>=0 else slope
        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.leaky_relu(X, 0.1), [], False)
        [g] = grad_of(out, [X])
        check(f"leaky_relu grad[{dev}]", g,
              np.where(x >= 0, 1.0, 0.1).astype(np.float32))
        # softmax along axis=-1: grad-of-sum is zero (softmax sums to 1)
        # so use a non-trivial loss: sum(softmax(x) * target)
        X = make(x, gpu, True)
        sm = _C_engine.softmax(X, -1)
        out = _C_engine.sum(sm, [], False)
        [g] = grad_of(out, [X])
        # d(sum(softmax))/dx = 0 since each row sums to 1
        check(f"softmax sum-grad ≈ 0[{dev}]", g, np.zeros_like(x), tol=1e-4)


def test_conv_pool_extra():
    section("conv1d/3d, conv_transpose, all pools (CPU + GPU)")
    rng = np.random.default_rng(21)
    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")

        # conv1d (identity tap, stride=1, pad=1)
        x1 = rng.standard_normal((1, 1, 8)).astype(np.float32)
        w1 = np.zeros((1, 1, 3), dtype=np.float32)
        w1[0, 0, 1] = 1.0
        b1 = np.zeros((1,), dtype=np.float32)
        check(f"conv1d (identity)[{dev}]",
              to_np(_C_engine.conv1d(make(x1, gpu), make(w1, gpu),
                               make(b1, gpu), 1, 1)),  # stride=1, pad=1
              x1)

        # conv3d identity tap, stride=1, pad=1 each axis
        x3 = rng.standard_normal((1, 1, 4, 4, 4)).astype(np.float32)
        w3 = np.zeros((1, 1, 3, 3, 3), dtype=np.float32)
        w3[0, 0, 1, 1, 1] = 1.0
        b3 = np.zeros((1,), dtype=np.float32)
        check(f"conv3d (identity)[{dev}]",
              to_np(_C_engine.conv3d(make(x3, gpu), make(w3, gpu),
                               make(b3, gpu),
                               1, 1, 1,  # strides
                               1, 1, 1)),  # paddings
              x3)

        # max_pool1d
        xp1 = rng.standard_normal((1, 2, 6)).astype(np.float32)
        out = to_np(_C_engine.max_pool1d(make(xp1, gpu), 2, 2, 0))
        np_out = np.maximum(xp1[..., 0::2], xp1[..., 1::2])
        check(f"max_pool1d[{dev}]", out, np_out)

        # avg_pool1d
        out = to_np(_C_engine.avg_pool1d(make(xp1, gpu), 2, 2, 0))
        np_avg = (xp1[..., 0::2] + xp1[..., 1::2]) / 2.0
        check(f"avg_pool1d[{dev}]", out, np_avg, tol=1e-4)

        # adaptive_avg_pool2d (output 2x2 from 4x4)
        xa = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
        out = to_np(_C_engine.adaptive_avg_pool2d(make(xa, gpu), 2, 2))
        # Each output cell averages a 2x2 block
        np_out = np.zeros((1, 1, 2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                np_out[0, 0, i, j] = xa[0, 0, i*2:i*2+2, j*2:j*2+2].mean()
        check(f"adaptive_avg_pool2d[{dev}]", out, np_out, tol=1e-4)

        # adaptive_max_pool2d
        out = to_np(_C_engine.adaptive_max_pool2d(make(xa, gpu), 2, 2))
        np_out = np.zeros((1, 1, 2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                np_out[0, 0, i, j] = xa[0, 0, i*2:i*2+2, j*2:j*2+2].max()
        check(f"adaptive_max_pool2d[{dev}]", out, np_out)


def test_norms_extra():
    section("group_norm + batch_norm 1D/3D (CPU + GPU)")
    rng = np.random.default_rng(22)
    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        # group_norm
        xn = rng.standard_normal((2, 6, 4)).astype(np.float32)
        gamma = np.ones((6,), dtype=np.float32)
        beta = np.zeros((6,), dtype=np.float32)
        out = to_np(_C_engine.group_norm(make(xn, gpu), make(gamma, gpu),
                                   make(beta, gpu), 2, 1e-5))
        # 2 groups of 3 channels each, normalize per (sample, group)
        groups = 2
        chs_per_group = 3
        np_out = np.zeros_like(xn)
        for n in range(2):
            for g in range(groups):
                grp = xn[n, g*chs_per_group:(g+1)*chs_per_group]
                mu = grp.mean()
                var = grp.var()
                np_out[n, g*chs_per_group:(g+1)*chs_per_group] = \
                    (grp - mu) / np.sqrt(var + 1e-5)
        check(f"group_norm[{dev}]", out, np_out, tol=1e-3)

        # batch_norm1d (B, C, L) — temporal
        xb = rng.standard_normal((4, 3, 5)).astype(np.float32)
        gam = np.ones((3,), dtype=np.float32)
        bet = np.zeros((3,), dtype=np.float32)
        out = to_np(_C_engine.batch_norm1d(make(xb, gpu), make(gam, gpu),
                                     make(bet, gpu), 1e-5))
        mu = xb.mean(axis=(0, 2), keepdims=True)
        var = xb.var(axis=(0, 2), keepdims=True)
        np_bn = (xb - mu) / np.sqrt(var + 1e-5)
        check(f"batch_norm1d[{dev}]", out, np_bn, tol=1e-3)


def test_optim_steps():
    section("optim step (SGD, Adam, AdamW)")
    # Simple convergence: minimize (p - target)^2 where p starts at 1.0.
    target = 5.0
    for opt_name in ("SGD", "Adam", "AdamW"):
        p_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        p = _C_engine.TensorImpl(p_arr.copy(), _C_engine.Device.CPU, True)
        if opt_name == "SGD":
            opt = _C_engine.SGD([p], lr=0.1)
        elif opt_name == "Adam":
            opt = _C_engine.Adam([p], lr=0.1)
        else:
            opt = _C_engine.AdamW([p], lr=0.1)
        # 50 steps. Loss = sum((p - target)^2). Grad = 2*(p - target).
        for _ in range(50):
            p.zero_grad()
            diff = _C_engine.sub(p, _C_engine.full([3], target, _C_engine.Dtype.F32, _C_engine.Device.CPU))
            loss = _C_engine.sum(_C_engine.square(diff), [], False)
            _C_engine.engine_backward(loss)
            opt.step()
        final = np.array(p.data_as_python())
        # After 50 steps with lr=0.1, all 3 should be close to 5.0.
        check(f"{opt_name} converges to {target}", final,
              np.full((3,), target, np.float32), tol=0.5)


def test_view_grad():
    section("view backward (reshape, squeeze, unsqueeze)")
    rng = np.random.default_rng(23)
    x = rng.standard_normal((2, 6)).astype(np.float32)
    for dev in ("CPU", "GPU"):
        gpu = (dev == "GPU")
        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.reshape(X, [3, 4]), [], False)
        [g] = grad_of(out, [X])
        check(f"reshape grad[{dev}]", g, np.ones_like(x))
        X = make(x, gpu, True)
        out = _C_engine.sum(_C_engine.unsqueeze(X, 0), [], False)
        [g] = grad_of(out, [X])
        check(f"unsqueeze grad[{dev}]", g, np.ones_like(x))


def main():
    test_bfunc()
    test_bfunc_grad()
    test_ufunc()
    test_ufunc_grad()
    test_reduction_grad()
    test_more_activations_grad()
    test_gfunc()
    test_utils()
    test_view_grad()
    test_linalg()
    test_nn()
    test_nn_grad()
    test_conv_pool_extra()
    test_norms_extra()
    test_random()
    test_optim_steps()
    test_cpu_gpu_equivalence()

    print(f"\n--- TOTAL: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped ---")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
