"""``lucid.linalg`` correctness against the reference — one section per issue.

Each section names the defect it pins and was written to fail first:

* **CHA-141** — ``eigh`` / ``eigvalsh`` ignored ``UPLO`` (the CPU always read
  the lower triangle, Metal the upper), and the ``*_ex`` family failed the
  whole batch when one matrix failed, with an invented ``info = 1``.

Values are compared with the reference framework through the ``ref``
fixture on every device; gradients in float64 on the CPU, where the
reference and a finite difference are both exact enough to disagree with.
"""

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
