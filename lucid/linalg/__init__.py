"""
lucid.linalg: linear algebra operations.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_la = _C_engine.linalg


def inv(x: Tensor) -> Tensor:
    """Matrix inverse."""
    return _wrap(_la.inv(_unwrap(x)))


def det(x: Tensor) -> Tensor:
    """Matrix determinant."""
    return _wrap(_la.det(_unwrap(x)))


def solve(A: Tensor, b: Tensor) -> Tensor:
    """Solve linear system Ax = b."""
    return _wrap(_la.solve(_unwrap(A), _unwrap(b)))


def cholesky(x: Tensor, *, upper: bool = False) -> Tensor:
    """Cholesky decomposition."""
    return _wrap(_la.cholesky(_unwrap(x), upper))


def norm(
    x: Tensor,
    ord: int | float | str | None = None,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Matrix or vector norm."""
    return _wrap(_la.norm(_unwrap(x)))


def qr(x: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    """QR decomposition."""
    q, r = _la.qr(_unwrap(x))
    return _wrap(q), _wrap(r)


def svd(
    x: Tensor, full_matrices: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    """Singular value decomposition. Returns (U, S, Vh)."""
    u, s, v = _la.svd(_unwrap(x))
    return _wrap(u), _wrap(s), _wrap(v)


def svdvals(x: Tensor) -> Tensor:
    """Singular values only (no U/Vh)."""
    result = _la.svd(_unwrap(x), False)
    # C++ returns {S} when compute_uv=False
    if isinstance(result, (list, tuple)):
        return _wrap(result[0])
    return _wrap(result)


def matrix_power(x: Tensor, n: int) -> Tensor:
    """Raise a matrix to an integer power."""
    return _wrap(_la.matrix_power(_unwrap(x), n))


def pinv(x: Tensor) -> Tensor:
    """Moore-Penrose pseudo-inverse."""
    return _wrap(_la.pinv(_unwrap(x)))


def eig(x: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalue decomposition."""
    vals, vecs = _la.eig(_unwrap(x))
    return _wrap(vals), _wrap(vecs)


def eigvals(x: Tensor) -> Tensor:
    """Eigenvalues only (no eigenvectors)."""
    vals, _ = _la.eig(_unwrap(x))
    return _wrap(vals)


def eigh(x: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Eigenvalue decomposition of a symmetric/Hermitian matrix."""
    vals, vecs = _la.eigh(_unwrap(x))
    return _wrap(vals), _wrap(vecs)


def eigvalsh(x: Tensor, UPLO: str = "L") -> Tensor:
    """Eigenvalues of a symmetric/Hermitian matrix (no eigenvectors)."""
    vals, _ = _la.eigh(_unwrap(x))
    return _wrap(vals)


# ── Pure-Python compositions ───────────────────────────────────────────────────


def slogdet(A: Tensor) -> tuple[Tensor, Tensor]:
    """Sign and log-absolute-determinant of a square matrix.

    Returns ``(sign, logabsdet)`` such that ``det(A) == sign * exp(logabsdet)``.
    """
    d = det(A)
    sign = _wrap(_C_engine.sign(_unwrap(d)))
    logabsdet = _wrap(_C_engine.log(_C_engine.abs(_unwrap(d))))
    return sign, logabsdet


def matrix_rank(
    A: Tensor,
    tol: float | None = None,
    hermitian: bool = False,
) -> Tensor:
    """Numerical matrix rank via SVD.

    Counts singular values strictly greater than *tol*.  When *tol* is
    ``None`` uses ``max(m, n) * eps * max_sv`` (PyTorch default).
    """
    _, S, _ = svd(A)
    # Convert to numpy for threshold computation (rank is not differentiable)
    S_np = np.asarray(S._impl.data_as_python(), dtype=np.float64).ravel()
    m, n = int(A.shape[-2]), int(A.shape[-1])
    if tol is None:
        eps = float(np.finfo(S_np.dtype).eps)
        tol_val = max(m, n) * float(S_np.max()) * eps
    else:
        tol_val = float(tol)
    rank = int(np.sum(S_np > tol_val))
    return _wrap(_C_engine.full([], float(rank), _C_engine.I64, _C_engine.CPU))


def cond(A: Tensor, p: int | float | str | None = None) -> Tensor:
    """Matrix condition number.

    For *p* = 2 (spectral norm, the default), returns ``max(sv) / min(sv)``.
    For other *p*, returns ``norm(A, p) * norm(inv(A), p)``.
    """
    if p is None or p == 2:
        _, S, _ = svd(A)
        S_impl = _unwrap(S)
        smax = _C_engine.max(S_impl, [-1], False)
        smin = _C_engine.min(S_impl, [-1], False)
        return _wrap(_C_engine.div(smax, smin))
    if p == -2:
        _, S, _ = svd(A)
        S_impl = _unwrap(S)
        smax = _C_engine.max(S_impl, [-1], False)
        smin = _C_engine.min(S_impl, [-1], False)
        return _wrap(_C_engine.div(smin, smax))
    # General p-norm condition number
    return _wrap(_C_engine.mul(_unwrap(norm(A, ord=p)), _unwrap(norm(inv(A), ord=p))))


def multi_dot(tensors: list[Tensor]) -> Tensor:
    """Efficiently multiply a sequence of matrices (left-to-right chain)."""
    if len(tensors) == 0:
        raise ValueError("multi_dot requires at least one tensor")
    if len(tensors) == 1:
        return tensors[0]
    result = _wrap(_C_engine.matmul(_unwrap(tensors[0]), _unwrap(tensors[1])))
    for t in tensors[2:]:
        result = _wrap(_C_engine.matmul(_unwrap(result), _unwrap(t)))
    return result


def lu_factor(A: Tensor) -> tuple[Tensor, Tensor]:
    """LU factorisation with partial pivoting.

    Returns ``(LU, pivots)`` where ``LU`` is the packed n×n matrix
    (LAPACK ``dgetrf_`` format: U on and above the diagonal, L below with
    implicit unit diagonal) and ``pivots`` is an int32 tensor of 1-based
    pivot indices.  Matches ``torch.linalg.lu_factor``.
    """
    lu, pivots = _la.lu_factor(_unwrap(A))
    return _wrap(lu), _wrap(pivots)


def solve_triangular(
    A: Tensor,
    B: Tensor,
    *,
    upper: bool = True,
    left: bool = True,
    unitriangular: bool = False,
) -> Tensor:
    """Solve the triangular linear system A X = B for X.

    Parameters
    ----------
    A : Tensor
        Triangular coefficient matrix.
    B : Tensor
        Right-hand side.
    upper : bool
        ``True`` if *A* is upper triangular (default); ``False`` for lower.
    left : bool
        ``True`` to solve ``A X = B`` (default).  ``left=False`` (``X A = B``)
        is not yet implemented.
    unitriangular : bool
        If ``True``, the diagonal entries of *A* are treated as 1.
    """
    if not left:
        # X A = B  ⟺  Aᵀ Xᵀ = Bᵀ  — solve the transposed system, then transpose result.
        AT = _wrap(_C_engine.mT(_unwrap(A)))
        BT = _wrap(_C_engine.mT(_unwrap(B)))
        XT = _wrap(_la.solve_triangular(_unwrap(AT), _unwrap(BT), not upper, unitriangular))
        return _wrap(_C_engine.mT(_unwrap(XT)))
    return _wrap(_la.solve_triangular(_unwrap(A), _unwrap(B), upper, unitriangular))


def vander(x: Tensor, N: int | None = None, increasing: bool = False) -> Tensor:
    """Vandermonde matrix.

    Given a 1-D vector *x* of length *n*, returns an *n* × *N* matrix
    where column *j* is ``x ** j`` when *increasing* is ``True``, or
    ``x ** (N-1-j)`` when *increasing* is ``False`` (the default).
    """
    n = int(x.shape[0])
    if N is None:
        N = n
    x_impl = _unwrap(x)
    # Build exponent vector
    if increasing:
        exp_impl = _C_engine.arange(0.0, float(N), 1.0, _C_engine.F32, _C_engine.CPU)
    else:
        exp_impl = _C_engine.arange(float(N - 1), -1.0, -1.0, _C_engine.F32, _C_engine.CPU)
    # x → (n, 1), exponents → (1, N)
    x_col = _C_engine.reshape(x_impl, [n, 1])
    exp_row = _C_engine.reshape(exp_impl, [1, N])
    return _wrap(_C_engine.pow(x_col, exp_row))


__all__ = [
    "inv",
    "det",
    "solve",
    "cholesky",
    "norm",
    "qr",
    "svd",
    "svdvals",
    "matrix_power",
    "pinv",
    "eig",
    "eigvals",
    "eigh",
    "eigvalsh",
    "slogdet",
    "matrix_rank",
    "cond",
    "multi_dot",
    "vander",
    "lu_factor",
    "solve_triangular",
]
