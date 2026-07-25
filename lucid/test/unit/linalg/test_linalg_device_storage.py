"""Regression tests for linalg ops whose Metal output desynced from its backing.

Four bugs found on 2026-07-22, all one invariant violation: a ``TensorImpl``'s
metadata disagreeing with the storage behind it.

* ``solve_triangular`` / ``lstsq`` / ``lu_factor`` — ``GpuBackend`` computed on
  the CPU (no MLX equivalent) and returned ``CpuStorage``, while the op wrapper
  tagged the result with the *input* device (Metal).  The device tag then
  disagreed with the storage variant, so the next engine op's
  ``std::get<GpuStorage>`` threw ``bad_variant_access``.  This is also what made
  ``cholesky`` / ``qr`` *backward* fail on Metal — both call
  ``solve_triangular``.
* ``svd`` — MLX returns the FULL decomposition (U is m×m) but the op wrapper
  tags the REDUCED shape (U is m×k), so ``U.shape`` read ``(5, 4)`` while the
  MLX array was ``(5, 5)`` and any consuming matmul got the wrong extents.

Separately, ``qr``'s Q-gradient used ``sym(M) = (M + Mᵀ)/2`` where the QR
backward requires ``copyltu(M)``; those agree only for symmetric ``M``, so the
gradient was wrong on *both* devices (rel err ~1.2 vs finite differences).
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as LA

DEVICES = ["cpu", "metal"]


def _assert_storage_usable(t, label):
    """A desynced device tag / shape only explodes on the *next* engine op."""
    _ = t + 1.0
    if t.ndim >= 2:
        _ = t.mT
        _ = t @ lucid.eye(int(t.shape[-1]), dtype=t.dtype, device=t.device)
    else:
        _ = t.sum()


def _spd(n, device, seed=0):
    rng = np.random.default_rng(seed)
    a = lucid.tensor(rng.standard_normal((n, n)).astype(np.float32), device=device)
    return a @ a.mT + lucid.eye(n, device=device) * 4.0


def _upper(n, device, seed=0):
    rng = np.random.default_rng(seed)
    m = np.triu(rng.standard_normal((n, n))).astype(np.float32)
    m += np.eye(n, dtype=np.float32) * 3.0
    return lucid.tensor(m, device=device)


# ── storage / shape integrity ────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_solve_triangular_output_is_usable(device):
    rng = np.random.default_rng(0)
    b = lucid.tensor(rng.standard_normal((4, 3)).astype(np.float32), device=device)
    out = LA.solve_triangular(_upper(4, device), b, upper=True)
    assert out.device == b.device
    _assert_storage_usable(out, "solve_triangular")


@pytest.mark.parametrize("device", DEVICES)
def test_lstsq_output_is_usable(device):
    rng = np.random.default_rng(1)
    a = lucid.tensor(rng.standard_normal((5, 4)).astype(np.float32), device=device)
    b = lucid.tensor(rng.standard_normal((5, 2)).astype(np.float32), device=device)
    sol = LA.lstsq(a, b)[0]
    assert sol.device == a.device
    _assert_storage_usable(sol, "lstsq")


@pytest.mark.parametrize("device", DEVICES)
def test_lu_factor_outputs_are_usable(device):
    lu, pivots = LA.lu_factor(_spd(4, device))
    assert lu.device == pivots.device
    _assert_storage_usable(lu, "lu_factor.LU")
    _assert_storage_usable(pivots, "lu_factor.pivots")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(5, 4), (4, 5), (4, 4), (6, 3)])
def test_svd_outputs_are_usable_and_reduced(device, shape):
    rng = np.random.default_rng(2)
    a = lucid.tensor(rng.standard_normal(shape).astype(np.float32), device=device)
    u, s, vh = LA.svd(a)
    k = min(shape)
    assert u.shape == (shape[0], k)
    assert s.shape == (k,)
    assert vh.shape == (k, shape[1])
    for name, t in (("U", u), ("S", s), ("Vh", vh)):
        _assert_storage_usable(t, f"svd.{name}")


# ── numerical correctness across devices ─────────────────────────────────────


def test_solve_triangular_matches_across_devices():
    rng = np.random.default_rng(3)
    u = np.triu(rng.standard_normal((4, 4))).astype(np.float32)
    u += np.eye(4, dtype=np.float32) * 3.0
    b = rng.standard_normal((4, 3)).astype(np.float32)
    outs = {
        dev: LA.solve_triangular(
            lucid.tensor(u, device=dev), lucid.tensor(b, device=dev), upper=True
        ).numpy()
        for dev in DEVICES
    }
    assert np.abs(outs["cpu"] - outs["metal"]).max() < 1e-5
    # A @ x == b  (absolute check, not just parity)
    assert np.abs(u @ outs["cpu"] - b).max() < 1e-4


def test_lu_factor_matches_across_devices():
    rng = np.random.default_rng(4)
    m = (rng.standard_normal((4, 4)) + np.eye(4) * 4).astype(np.float32)
    got = {dev: LA.lu_factor(lucid.tensor(m, device=dev)) for dev in DEVICES}
    assert np.abs(got["cpu"][0].numpy() - got["metal"][0].numpy()).max() < 1e-5
    assert np.abs(got["cpu"][1].numpy() - got["metal"][1].numpy()).max() == 0


@pytest.mark.parametrize("shape", [(5, 4), (4, 5), (6, 3), (4, 4)])
@pytest.mark.parametrize("device", DEVICES)
def test_svd_reconstructs_input(shape, device):
    rng = np.random.default_rng(5)
    a = rng.standard_normal(shape).astype(np.float32)
    u, s, vh = LA.svd(lucid.tensor(a, device=device))
    recon = (u @ lucid.diag_embed(s) @ vh).numpy()
    assert np.abs(recon - a).max() < 1e-4


# ── backward paths that depended on the above ────────────────────────────────


def _grad(base, dev, fn, w):
    dt = lucid.float64 if dev == "cpu" else lucid.float32
    data = base if dev == "cpu" else base.astype(np.float32)
    weight = w if dev == "cpu" else w.astype(np.float32)
    x = lucid.tensor(data, device=dev, dtype=dt)
    x.requires_grad = True
    (fn(x) * lucid.tensor(weight, device=dev, dtype=dt)).sum().backward()
    assert x.grad is not None
    return np.asarray(x.grad.numpy(), dtype=np.float64)


def _finite_diff(base, fn, w, eps=1e-6):
    def value(b):
        x = lucid.tensor(b, dtype=lucid.float64)
        return float((fn(x) * lucid.tensor(w, dtype=lucid.float64)).sum().item())

    out = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        plus = base.copy()
        plus[idx] += eps
        minus = base.copy()
        minus[idx] -= eps
        out[idx] = (value(plus) - value(minus)) / (2 * eps)
        it.iternext()
    return out


def _check_grad(shape, fn, dev, seed=6, tol_cpu=1e-7, tol_metal=1e-4):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape).astype(np.float64)
    w = rng.standard_normal(fn(lucid.tensor(base, dtype=lucid.float64)).shape)
    w = w.astype(np.float64)
    got = _grad(base, dev, fn, w)
    ref = _finite_diff(base, fn, w)
    scale = max(float(np.abs(ref).max()), 1e-12)
    tol = tol_cpu if dev == "cpu" else tol_metal
    assert np.abs(got - ref).max() / scale < tol


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(4, 4), (5, 4), (6, 3)])
def test_qr_q_backward_matches_finite_differences(device, shape):
    """The Q-gradient used sym() instead of copyltu() — wrong on both devices."""
    _check_grad(shape, lambda t: LA.qr(t)[0], device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(4, 4), (5, 4)])
def test_qr_r_backward_matches_finite_differences(device, shape):
    _check_grad(shape, lambda t: LA.qr(t)[1], device)


@pytest.mark.parametrize("device", DEVICES)
def test_qr_combined_backward(device):
    """Q and R contributions are summed by separate nodes — check the sum."""
    _check_grad((5, 4), lambda t: LA.qr(t)[0].sum() + LA.qr(t)[1].sum(), device)


@pytest.mark.parametrize("device", DEVICES)
def test_cholesky_backward_runs_and_is_correct(device):
    def fn(t):
        spd = t @ t.mT + lucid.eye(4, dtype=t.dtype, device=t.device) * 4.0
        return LA.cholesky(spd)

    _check_grad((4, 4), fn, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(5, 4), (4, 5), (4, 4), (6, 3)])
def test_svd_singular_value_backward(device, shape):
    _check_grad(shape, lambda t: LA.svd(t)[1], device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("out_idx", [0, 1, 2])
@pytest.mark.parametrize("shape", [(5, 4), (4, 5), (4, 4)])
def test_svd_all_outputs_backward_runs(device, out_idx, shape):
    """U / Vh have sign ambiguity so only assert the backward completes."""
    rng = np.random.default_rng(7)
    x = lucid.tensor(rng.standard_normal(shape).astype(np.float32), device=device)
    x.requires_grad = True
    out = LA.svd(x)[out_idx]
    w = lucid.tensor(rng.standard_normal(out.shape).astype(np.float32), device=device)
    (out * w).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
