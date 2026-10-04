"""``lucid.linalg`` correctness against the reference — one section per issue.

Each section names the defect it pins and was written to fail first:

* **CHA-141** — ``eigh`` / ``eigvalsh`` ignored ``UPLO`` (the CPU always read
  the lower triangle, Metal the upper), and the ``*_ex`` family failed the
  whole batch when one matrix failed, with an invented ``info = 1``.
* **CHA-142** — ``ldl_solve`` applied the Bunch-Kaufman interchanges as one
  up-front permutation (wrong as soon as a later step swaps rows an earlier
  column of ``L`` reaches), ``ldl_factor`` left the input's upper triangle
  in ``LD``, and both pivot paths read their pivots through numpy (H4).

Values are compared with the reference framework through the ``ref``
fixture on every device; gradients in float64 on the CPU, where the
reference and a finite difference are both exact enough to disagree with.
"""

import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.linalg as LA


def _np(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.to("cpu").numpy())


def _rtol(device: str) -> float:
    return 1e-3 if device == "metal" else 1e-5


# ── CHA-141: eigh / eigvalsh read the triangle UPLO names ────────────────────


def _lopsided(seed: int = 0) -> np.ndarray:
    """An SPD matrix with +10 added to its upper triangle only.

    The two triangles describe different symmetric matrices with different
    spectra, so reading the wrong one cannot pass by accident.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((4, 4))
    spd = m @ m.T + 4 * np.eye(4)
    return (spd + np.triu(np.full((4, 4), 10.0), 1)).astype(np.float32)


@pytest.mark.parity
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_eigh_reads_the_triangle_uplo_names(uplo: str, device: str, ref: Any) -> None:
    a = _lopsided()
    w, v = LA.eigh(lucid.tensor(a, device=device), UPLO=uplo)
    rw, rv = ref.linalg.eigh(ref.tensor(a), UPLO=uplo)
    np.testing.assert_allclose(_np(w), rw.numpy(), rtol=_rtol(device), atol=1e-3)
    # Eigenvectors are defined up to sign; their squares are not.
    np.testing.assert_allclose(_np(v) ** 2, rv.numpy() ** 2, atol=1e-3)


@pytest.mark.parity
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_eigvalsh_reads_the_triangle_uplo_names(
    uplo: str, requires_grad: bool, device: str, ref: Any
) -> None:
    a = _lopsided(1)
    got = LA.eigvalsh(
        lucid.tensor(a, device=device, requires_grad=requires_grad), UPLO=uplo
    )
    want = ref.linalg.eigvalsh(ref.tensor(a), UPLO=uplo)
    np.testing.assert_allclose(
        _np(got.detach()), want.numpy(), rtol=_rtol(device), atol=1e-3
    )


@pytest.mark.parity
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_eigh_gradient_matches_the_reference(uplo: str, ref: Any) -> None:
    """The reference's gradient is the symmetric one, whichever triangle
    was read — not a gradient confined to that triangle."""
    a = _lopsided(2).astype(np.float64)
    rng = np.random.default_rng(3)
    ww, wv = rng.standard_normal(4), rng.standard_normal((4, 4))

    x = lucid.tensor(a, requires_grad=True)
    w, v = LA.eigh(x, UPLO=uplo)
    ((w * lucid.tensor(ww)).sum() + (v * v * lucid.tensor(wv)).sum()).backward()

    t = ref.tensor(a, requires_grad=True)
    rw, rv = ref.linalg.eigh(t, UPLO=uplo)
    ((rw * ref.tensor(ww)).sum() + (rv * rv * ref.tensor(wv)).sum()).backward()
    np.testing.assert_allclose(_np(x.grad), t.grad.numpy(), atol=1e-8)


def test_eigh_refuses_an_unknown_uplo() -> None:
    with pytest.raises(ValueError, match="UPLO"):
        LA.eigh(lucid.eye(2), UPLO="X")
    with pytest.raises(ValueError, match="UPLO"):
        LA.eigvalsh(lucid.eye(2), UPLO="lower")


# ── CHA-141: *_ex report each matrix of a batch on its own ───────────────────

_SINGULAR = [[1.0, 2.0], [2.0, 4.0]]  # getrf: U[2, 2] == 0  → info 2
_NOT_PD = [[1.0, 2.0], [2.0, 1.0]]  # potrf: order-2 minor < 0 → info 2


def _batch(*mats: Any) -> np.ndarray:
    return np.stack([np.asarray(m, dtype=np.float32) for m in mats])


@pytest.mark.parity
def test_inv_ex_keeps_the_matrices_that_succeeded(device: str, ref: Any) -> None:
    a = _batch(np.eye(2), _SINGULAR, [[4.0, 7.0], [2.0, 6.0]])
    out, info = LA.inv_ex(lucid.tensor(a, device=device))
    rout, rinfo = ref.linalg.inv_ex(ref.tensor(a))
    assert _np(info).tolist() == rinfo.numpy().tolist() == [0, 2, 0]
    assert info.dtype == lucid.int32
    ok = [0, 2]
    np.testing.assert_allclose(
        _np(out)[ok], rout.numpy()[ok], rtol=_rtol(device), atol=1e-5
    )
    np.testing.assert_array_equal(_np(out)[1], np.zeros((2, 2)))


@pytest.mark.parity
@pytest.mark.parametrize("rhs", ["matrix", "vector"])
def test_solve_ex_keeps_the_systems_that_succeeded(
    rhs: str, device: str, ref: Any
) -> None:
    if rhs == "vector" and device == "metal":
        pytest.skip("Metal solve refuses a batched vector right-hand side (engine)")
    a = _batch(np.eye(2), _SINGULAR)
    b = np.ones((2, 2, 1) if rhs == "matrix" else (2, 2), dtype=np.float32)
    out, info = LA.solve_ex(
        lucid.tensor(a, device=device), lucid.tensor(b, device=device)
    )
    rout, rinfo = ref.linalg.solve_ex(ref.tensor(a), ref.tensor(b))
    assert _np(info).tolist() == rinfo.numpy().tolist() == [0, 2]
    assert tuple(out.shape) == tuple(rout.shape)
    np.testing.assert_allclose(_np(out)[0], rout.numpy()[0], rtol=_rtol(device))
    np.testing.assert_array_equal(_np(out)[1], np.zeros_like(rout.numpy()[1]))


@pytest.mark.parity
def test_cholesky_ex_keeps_the_factors_that_succeeded(device: str, ref: Any) -> None:
    a = _batch([[4.0, 2.0], [2.0, 3.0]], _NOT_PD, [[9.0, 3.0], [3.0, 5.0]])
    out, info = LA.cholesky_ex(lucid.tensor(a, device=device))
    rout, rinfo = ref.linalg.cholesky_ex(ref.tensor(a))
    assert _np(info).tolist() == rinfo.numpy().tolist() == [0, 2, 0]
    ok = [0, 2]
    np.testing.assert_allclose(
        _np(out)[ok], rout.numpy()[ok], rtol=_rtol(device), atol=1e-6
    )
    np.testing.assert_array_equal(_np(out)[1], np.zeros((2, 2)))


@pytest.mark.parity
@pytest.mark.parametrize(
    "op,matrix",
    [
        ("inv_ex", _SINGULAR),
        ("inv_ex", [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ("inv_ex", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        ("cholesky_ex", _NOT_PD),
        ("cholesky_ex", np.diag([1.0, 1.0, -1.0])),
        ("cholesky_ex", np.diag([1.0, 0.0, 1.0])),
        ("cholesky_ex", [[float("nan"), 0.0], [0.0, 4.0]]),
    ],
)
def test_ex_info_is_the_lapack_code(
    op: str, matrix: Any, device: str, ref: Any
) -> None:
    """Not a bare non-zero flag: the pivot / leading-minor order."""
    a = np.asarray(matrix, dtype=np.float32)
    _, info = getattr(LA, op)(lucid.tensor(a, device=device))
    _, rinfo = getattr(ref.linalg, op)(ref.tensor(a))
    assert info.shape == ()
    assert int(info.item()) == int(rinfo.item()) != 0


@pytest.mark.parametrize(
    "op,matrix,message",
    [
        ("inv_ex", _SINGULAR, "diagonal element 2 is zero"),
        ("cholesky_ex", _NOT_PD, "leading minor of order 2"),
    ],
)
def test_check_errors_names_the_failed_batch_element(
    op: str, matrix: Any, message: str, device: str
) -> None:
    a = _batch(np.eye(2), matrix)
    with pytest.raises(RuntimeError, match=rf"Batch element 1\): .*{message}"):
        getattr(LA, op)(lucid.tensor(a, device=device), check_errors=True)


def test_solve_ex_check_errors_raises(device: str) -> None:
    a = _batch(np.eye(2), _SINGULAR)
    with pytest.raises(RuntimeError, match="Batch element 1"):
        LA.solve_ex(
            lucid.tensor(a, device=device),
            lucid.ones(2, 2, 1, device=device),
            check_errors=True,
        )


@pytest.mark.parametrize("op", ["inv_ex", "cholesky_ex"])
def test_ex_gradient_reaches_only_the_matrices_that_succeeded(
    op: str, device: str
) -> None:
    good = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    bad = np.asarray(_SINGULAR if op == "inv_ex" else _NOT_PD, dtype=np.float32)
    x = lucid.tensor(_batch(good, bad), device=device, requires_grad=True)
    out, _ = getattr(LA, op)(x)
    out.sum().backward()

    alone = lucid.tensor(good, device=device, requires_grad=True)
    (LA.inv(alone) if op == "inv_ex" else LA.cholesky(alone)).sum().backward()
    np.testing.assert_allclose(
        _np(x.grad)[0], _np(alone.grad), rtol=_rtol(device), atol=1e-6
    )
    np.testing.assert_array_equal(_np(x.grad)[1], np.zeros((2, 2)))


@pytest.mark.parametrize("op", ["inv_ex", "cholesky_ex", "solve_ex"])
@pytest.mark.parametrize("shape", [(2, 3), (3,)])
def test_ex_refuses_a_non_square_input(op: str, shape: tuple[int, ...]) -> None:
    """A shape error is not a singular matrix: it was reported as
    ``info = 1`` with a zero result, because the kernel's shape refusal is
    the same exception type as a failed factorisation."""
    args = (lucid.ones(*shape),) + ((lucid.ones(2, 1),) if op == "solve_ex" else ())
    with pytest.raises(ValueError, match="square"):
        getattr(LA, op)(*args)


# ── CHA-142: ldl_solve replays Bunch-Kaufman interchanges in order ───────────


def _symmetric(seed: int, n: int, dtype: Any = np.float64) -> np.ndarray:
    a = np.random.default_rng(seed).standard_normal((n, n))
    return (a + a.T).astype(dtype)


def _dtype(device: str) -> Any:
    return np.float32 if device == "metal" else np.float64


@pytest.mark.parity
def test_ldl_factor_zeroes_the_upper_triangle(device: str, ref: Any) -> None:
    """``sytrf`` leaves the input's upper triangle in place; the reference
    clears it, and so does Lucid now."""
    s = _symmetric(0, 4, _dtype(device))
    ld, piv = LA.ldl_factor(lucid.tensor(s, device=device))
    rld, rpiv = ref.linalg.ldl_factor(ref.tensor(s))
    assert _np(piv).tolist() == rpiv.numpy().tolist()
    np.testing.assert_array_equal(np.triu(_np(ld), 1), np.zeros((4, 4)))
    np.testing.assert_allclose(_np(ld), rld.numpy(), rtol=_rtol(device), atol=1e-6)


@pytest.mark.parity
def test_ldl_solve_with_row_interchanges(device: str, ref: Any) -> None:
    """The issue's case: pivots ``[4, 4, 3, 4]`` — two 1×1 steps that both
    swap with row 4, which a single up-front permutation of ``B`` cannot
    express."""
    a = np.random.default_rng(0).standard_normal((4, 4)).astype(_dtype(device))
    s = a + a.T
    b = np.random.default_rng(1).standard_normal((4, 2)).astype(_dtype(device))
    ld, piv = LA.ldl_factor(lucid.tensor(s, device=device))
    assert _np(piv).tolist() == [4, 4, 3, 4]
    x = LA.ldl_solve(ld, piv, lucid.tensor(b, device=device))
    rld, rpiv = ref.linalg.ldl_factor(ref.tensor(s))
    want = ref.linalg.ldl_solve(rld, rpiv, ref.tensor(b)).numpy()
    np.testing.assert_allclose(_np(x), want, rtol=_rtol(device), atol=1e-4)
    np.testing.assert_allclose(
        _np(x), np.linalg.solve(s.astype(np.float64), b), rtol=1e-3, atol=1e-4
    )


def test_ldl_solve_over_every_pivot_pattern(device: str) -> None:
    """Random symmetric indefinite systems: 1×1 interchanges, 2×2 blocks
    and none, checked against a dense solve — and the sweep asserts it
    actually met each pattern."""
    dtype = _dtype(device)
    seen = {"interchange": 0, "2x2": 0, "none": 0}
    worst = 0.0
    for seed in range(60):
        n = 3 + seed % 5
        s = _symmetric(seed, n, dtype)
        b = np.random.default_rng(seed + 1000).standard_normal((n, 3)).astype(dtype)
        ld, piv = LA.ldl_factor(lucid.tensor(s, device=device))
        p = _np(piv).tolist()
        seen["2x2"] += any(v < 0 for v in p)
        seen["interchange"] += any(v > 0 and v != i + 1 for i, v in enumerate(p))
        seen["none"] += all(v == i + 1 for i, v in enumerate(p))
        x = _np(LA.ldl_solve(ld, piv, lucid.tensor(b, device=device)))
        want = np.linalg.solve(s.astype(np.float64), b.astype(np.float64))
        worst = max(worst, float(np.max(np.abs(x - want)) / np.max(np.abs(want))))
    assert all(seen.values()), seen
    assert worst < (1e-3 if device == "metal" else 1e-10), worst


def test_ldl_solve_batched_broadcast_and_vector() -> None:
    s = np.stack([_symmetric(seed, 5) for seed in (3, 4, 5)])
    b = np.random.default_rng(9).standard_normal((3, 5, 2))
    ld, piv = LA.ldl_factor(lucid.tensor(s))
    x = LA.ldl_solve(ld, piv, lucid.tensor(b))
    np.testing.assert_allclose(_np(x), np.linalg.solve(s, b), atol=1e-10)
    # One right-hand side shared by the batch.
    x = LA.ldl_solve(ld, piv, lucid.tensor(b[0]))
    np.testing.assert_allclose(
        _np(x), np.linalg.solve(s, np.broadcast_to(b[0], (3, 5, 2))), atol=1e-10
    )
    # A single vector.
    x = LA.ldl_solve(ld[1], piv[1], lucid.tensor(b[1, :, 0]))
    assert tuple(x.shape) == (5,)
    np.testing.assert_allclose(_np(x), np.linalg.solve(s[1], b[1, :, 0]), atol=1e-10)


def test_ldl_solve_is_differentiable_in_b() -> None:
    s = _symmetric(0, 4)
    ld, piv = LA.ldl_factor(lucid.tensor(s))
    b = lucid.tensor(np.ones((4, 1)), requires_grad=True)
    LA.ldl_solve(ld, piv, b).sum().backward()
    # d(1ᵀ A⁻¹ b)/db = A⁻ᵀ 1 = A⁻¹ 1 for symmetric A.
    np.testing.assert_allclose(
        _np(b.grad), np.linalg.solve(s, np.ones((4, 1))), atol=1e-10
    )


@pytest.mark.parametrize(
    "pivots",
    [
        [1, 2, 0],  # zero is not a LAPACK pivot
        [1, 2, 7],  # out of range
        [-2, 2, 3],  # an unpaired negative pivot
        [1, 2, -3],  # a 2x2 block running past the end
    ],
)
def test_ldl_solve_refuses_malformed_pivots(pivots: list[int]) -> None:
    ld, _ = LA.ldl_factor(lucid.tensor(_symmetric(0, 3)))
    with pytest.raises(ValueError, match="pivot"):
        LA.ldl_solve(ld, lucid.tensor(pivots, dtype=lucid.int32), lucid.ones(3, 1))


def test_ldl_solve_refuses_float_pivots() -> None:
    ld, piv = LA.ldl_factor(lucid.tensor(_symmetric(0, 3)))
    with pytest.raises(TypeError, match="integer"):
        LA.ldl_solve(ld, piv.to(lucid.float32), lucid.ones(3, 1))


def test_lu_and_lu_factor_gradient_take_a_batch(device: str) -> None:
    """The pivot-to-permutation conversion handled one matrix only, so
    ``lu`` and the gradient of ``lu_factor`` refused a batch."""
    a = np.random.default_rng(4).standard_normal((3, 4, 4)).astype(np.float32)
    P, L, U = LA.lu(lucid.tensor(a, device=device))
    np.testing.assert_allclose(_np(P @ L @ U), a, atol=1e-5)

    x = lucid.tensor(a, device=device, requires_grad=True)
    LA.lu_factor(x)[0].sum().backward()
    for i in range(3):
        one = lucid.tensor(a[i], device=device, requires_grad=True)
        LA.lu_factor(one)[0].sum().backward()
        np.testing.assert_allclose(
            _np(x.grad)[i], _np(one.grad), rtol=_rtol(device), atol=1e-5
        )


def test_pivot_paths_run_without_numpy() -> None:
    """``ldl_solve`` and the LU permutation read their pivots through
    ``Tensor.numpy`` — numpy on the compute path, outside every H4 bridge.
    ``check_numpy_h4`` only sees ``import`` statements, and the import
    happened inside the sanctioned ``Tensor.numpy``, so it never noticed.
    """
    script = textwrap.dedent("""
        import sys

        for mod in list(sys.modules):
            if mod == "numpy" or mod.startswith("numpy."):
                del sys.modules[mod]

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "numpy" or name.startswith("numpy."):
                    raise ImportError("numpy blocked for test")
                return None

        sys.meta_path.insert(0, _Blocker())

        import lucid
        import lucid.linalg as LA

        S = lucid.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 1.0]])
        LD, piv = LA.ldl_factor(S)
        x = LA.ldl_solve(LD, piv, lucid.ones(3, 1))
        assert float((S @ x - 1.0).abs().max().item()) < 1e-4

        A = lucid.tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 1.0], [1.0, 3.0]]])
        P, L, U = LA.lu(A)
        assert float((P @ L @ U - A).abs().max().item()) < 1e-5
        A.requires_grad_(True)
        LA.lu_factor(A)[0].sum().backward()
        assert A.grad is not None
        assert "numpy" not in sys.modules
        print("ok")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
