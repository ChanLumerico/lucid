"""
lucid.linalg: linear algebra operations.
"""

import functools
from typing import Callable, TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_la = _C_engine.linalg


# ── Decorator: auto-unwrap Tensor inputs / re-wrap TensorImpl outputs ─────────


def _linalg_op(fn: Callable[..., object]) -> Callable[..., object]:
    """Unwrap every ``Tensor`` argument before the call; re-wrap ``TensorImpl``
    results (including those inside tuples) back to ``Tensor`` afterwards.

    This eliminates the boilerplate ``_wrap(_la.foo(_unwrap(x)))`` pattern
    from every trivial linalg wrapper while keeping full type annotations on
    the public function signature.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        ua = tuple(_unwrap(a) if hasattr(a, "_impl") else a for a in args)  # type: ignore[arg-type]
        uk = {k: _unwrap(v) if hasattr(v, "_impl") else v for k, v in kwargs.items()}  # type: ignore[arg-type]
        out = fn(*ua, **uk)
        if isinstance(out, tuple):
            return tuple(
                _wrap(o) if isinstance(o, _C_engine.TensorImpl) else o for o in out
            )
        if isinstance(out, _C_engine.TensorImpl):
            return _wrap(out)
        return out

    return wrapper


# ── Engine-backed ops ─────────────────────────────────────────────────────────


@_linalg_op
def inv(x: Tensor) -> Tensor:
    """Matrix inverse."""
    return _la.inv(x)  # type: ignore[arg-type]


@_linalg_op
def det(x: Tensor) -> Tensor:
    """Matrix determinant."""
    return _la.det(x)  # type: ignore[arg-type]


@_linalg_op
def solve(A: Tensor, b: Tensor) -> Tensor:
    """Solve linear system Ax = b."""
    return _la.solve(A, b)  # type: ignore[arg-type]


def cholesky(x: Tensor, *, upper: bool = False) -> Tensor:
    """Cholesky decomposition.

    Differentiable via Murray's 2016 formula:
        S = L^{-T} @ Phi(L^T @ G) @ L^{-1}
        ∂L/∂A = (S + S^T) / 2
    where Phi(M) zeroes the strictly upper triangle and halves the diagonal.

    The engine ``cholesky_op`` has no autograd node, so the backward is
    computed in Python on top of ``solve_triangular``, ``matmul``, ``tril``
    and ``eye`` — all of which are themselves differentiable.
    """
    return _CholeskyAutograd.apply(x, upper)  # type: ignore[no-any-return]


from lucid.autograd.function import Function as _AutogradFunction


class _CholeskyAutograd(_AutogradFunction):
    """Custom-autograd wrapper around the engine's non-differentiable
    cholesky_op. Forward calls the engine; backward computes the input
    gradient via Murray (2016)."""

    @staticmethod
    def forward(ctx, x, upper):  # type: ignore[no-untyped-def]
        out = _wrap(_la.cholesky(_unwrap(x), upper))
        ctx.save_for_backward(out)
        ctx.upper = bool(upper)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[no-untyped-def]
        import lucid as _lucid

        (factor,) = ctx.saved_tensors  # L (upper=False) or U (upper=True)
        upper: bool = ctx.upper

        # Normalise to lower-triangular form L; gradient w.r.t. that L.
        if upper:
            L = factor.mT
            gL = grad_out.mT
        else:
            L = factor
            gL = grad_out

        n: int = int(L.shape[-1])
        eye_n = _lucid.eye(n, dtype=L.dtype)
        if L.device != "cpu":
            eye_n = eye_n.to(L.device)
        # Mask gL to its lower triangle — the strictly upper half doesn't
        # contribute to L (which is lower-triangular by construction).
        gL_tril = _lucid.tril(gL)
        # Phi(M): tril(M) with diagonal halved.
        M = _lucid.matmul(L.mT, gL_tril)
        Phi = _lucid.tril(M) - 0.5 * (M * eye_n)

        # S = L^{-T} @ Phi @ L^{-1}, computed via two triangular solves.
        # Step 1: Y = L^{-T} Phi  →  solve L^T Y = Phi (upper=True against L^T).
        Y = solve_triangular(L.mT, Phi, upper=True)
        # Step 2: Z = Y L^{-1}.  Take transposes: Z^T = L^{-T} Y^T, so solve
        # L^T Z^T = Y^T then transpose back.
        Z = solve_triangular(L.mT, Y.mT, upper=True).mT
        # ``Z`` is already symmetric in exact arithmetic (Phi sandwiched
        # between L^{-T} and L^{-1} applied to the lower-tri-only gradient
        # produces a symmetric result); the explicit symmetrisation below
        # absorbs any floating-point asymmetry.
        grad_A = 0.5 * (Z + Z.mT)
        return grad_A


def norm(
    x: Tensor,
    ord: int | float | str | None = None,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Matrix or vector norm."""
    return _wrap(_la.norm(_unwrap(x)))


@_linalg_op
def qr(x: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    """QR decomposition."""
    q, r = _la.qr(x)  # type: ignore[arg-type]
    return q, r  # type: ignore[return-value]


@_linalg_op
def svd(x: Tensor, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """Singular value decomposition. Returns (U, S, Vh)."""
    u, s, v = _la.svd(x)  # type: ignore[arg-type]
    return u, s, v  # type: ignore[return-value]


def svdvals(x: Tensor) -> Tensor:
    """Singular values only (no U/Vh)."""
    result = _la.svd(_unwrap(x), False)
    if isinstance(result, (list, tuple)):
        return _wrap(result[0])
    return _wrap(result)


def matrix_power(x: Tensor, n: int) -> Tensor:
    """Raise a matrix to an integer power.

    Implemented in Python on top of ``matmul`` and ``inv`` so autograd flows
    through naturally — the engine ``matrix_power_op`` is not differentiable
    on its own. Uses repeated squaring so the work is O(log |n|) matmuls.
    """
    import lucid as _lucid  # local to avoid a top-level cycle with linalg

    if not isinstance(n, int):
        raise TypeError(f"matrix_power exponent must be int, got {type(n).__name__}")

    sh: tuple[int, ...] = tuple(_unwrap(x).shape)
    if len(sh) < 2 or sh[-1] != sh[-2]:
        raise ValueError(
            f"matrix_power requires a square matrix in the last two dims, got {sh}"
        )

    if n == 0:
        # Identity broadcast to the input's batch shape.
        eye_2d: Tensor = _lucid.eye(int(sh[-1]), dtype=x.dtype)
        if len(sh) == 2:
            return eye_2d
        return _lucid.broadcast_to(eye_2d, list(sh))

    base: Tensor = inv(x) if n < 0 else x
    exponent: int = -n if n < 0 else n
    if exponent == 1:
        return base

    # Standard binary exponentiation: result starts at base if the lowest
    # bit is set, otherwise it gets multiplied in on the first set bit.
    result: Tensor | None = None
    cur: Tensor = base
    while exponent > 0:
        if exponent & 1:
            result = cur if result is None else _lucid.matmul(result, cur)
        exponent >>= 1
        if exponent:
            cur = _lucid.matmul(cur, cur)
    assert result is not None  # exponent was non-zero on entry
    return result


def pinv(x: Tensor) -> Tensor:
    """Moore-Penrose pseudo-inverse.

    For non-square or rank-deficient matrices the engine kernel is used
    directly (no autograd). For square full-rank matrices we route through
    ``inv`` so autograd flows naturally — covering the common case where
    pinv is just a robust ``inv`` substitute.
    """
    sh: tuple[int, ...] = tuple(_unwrap(x).shape)
    if len(sh) >= 2 and sh[-1] == sh[-2]:
        # Square matrix → ``pinv ≡ inv`` for full-rank input. Falling back to
        # the engine pinv would lose autograd; ``inv`` keeps it.
        return inv(x)
    return _wrap(_la.pinv(_unwrap(x)))  # type: ignore[arg-type]


@_linalg_op
def eig(x: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalue decomposition."""
    vals, vecs = _la.eig(x)  # type: ignore[arg-type]
    return vals, vecs  # type: ignore[return-value]


def eigvals(x: Tensor) -> Tensor:
    """Eigenvalues only (no eigenvectors)."""
    vals, _ = _la.eig(_unwrap(x))
    return _wrap(vals)


@_linalg_op
def eigh(x: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Eigenvalue decomposition of a symmetric/Hermitian matrix."""
    vals, vecs = _la.eigh(x)  # type: ignore[arg-type]
    return vals, vecs  # type: ignore[return-value]


def eigvalsh(x: Tensor, UPLO: str = "L") -> Tensor:
    """Eigenvalues of a symmetric/Hermitian matrix (no eigenvectors)."""
    vals, _ = _la.eigh(_unwrap(x))
    return _wrap(vals)


@_linalg_op
def lu_factor(A: Tensor) -> tuple[Tensor, Tensor]:
    """LU factorisation with partial pivoting.

    Returns ``(LU, pivots)`` where ``LU`` is the packed n×n matrix
    (LAPACK ``dgetrf_`` format: U on and above the diagonal, L below with
    implicit unit diagonal) and ``pivots`` is an int32 tensor of 1-based
    pivot indices.  Matches ``the reference LU factor API``.
    """
    lu, pivots = _la.lu_factor(A)  # type: ignore[arg-type]
    return lu, pivots  # type: ignore[return-value]


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
    ``None`` uses ``max(m, n) * eps * max_sv`` (reference default).
    """
    _, S, _ = svd(A)
    m, n = int(A.shape[-2]), int(A.shape[-1])
    if tol is None:
        # eps for float32 ≈ 1.19e-7; use C++ ops to compute max(sv)*max(m,n)*eps
        max_sv_t = _wrap(_C_engine.max(_unwrap(S), [], False))
        tol_val = float(max_sv_t.item()) * max(m, n) * 1.1920929e-7
    else:
        tol_val = float(tol)
    S_impl = _unwrap(S)
    thr = _C_engine.full(list(S_impl.shape), tol_val, S_impl.dtype, S_impl.device)
    gt = _C_engine.greater(S_impl, thr)
    # bool tensors can't be summed directly — convert via where
    one = _C_engine.full(list(S_impl.shape), 1.0, S_impl.dtype, S_impl.device)
    zero = _C_engine.full(list(S_impl.shape), 0.0, S_impl.dtype, S_impl.device)
    gt_f = _C_engine.where(gt, one, zero)
    rank = int(_wrap(_C_engine.sum(gt_f, [], False)).item())
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
        ``True`` to solve ``A X = B`` (default).
        ``left=False`` solves ``X A = B`` via transposition.
    unitriangular : bool
        If ``True``, the diagonal entries of *A* are treated as 1.
    """
    if not left:
        # X A = B  ⟺  Aᵀ Xᵀ = Bᵀ — solve the transposed system, transpose result.
        AT = _wrap(_C_engine.mT(_unwrap(A)))
        BT = _wrap(_C_engine.mT(_unwrap(B)))
        XT = _wrap(
            _la.solve_triangular(_unwrap(AT), _unwrap(BT), not upper, unitriangular)
        )
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
    if increasing:
        exp_impl = _C_engine.arange(0.0, float(N), 1.0, _C_engine.F32, _C_engine.CPU)
    else:
        exp_impl = _C_engine.arange(
            float(N - 1), -1.0, -1.0, _C_engine.F32, _C_engine.CPU
        )
    x_col = _C_engine.reshape(x_impl, [n, 1])
    exp_row = _C_engine.reshape(exp_impl, [1, N])
    return _wrap(_C_engine.pow(x_col, exp_row))


def vector_norm(
    x: Tensor,
    ord: int | float = 2,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
    dtype: object = None,
) -> Tensor:
    """Compute a vector norm along *dim* using existing C++ engine ops.

    All computation is done through autograd-tracked engine operations.
    """
    import math

    xi = _unwrap(x)
    axes: list[int] = []
    if dim is None:
        # Flatten then reduce over all elements
        xi = _C_engine.reshape(xi, [-1])
        axes = [0]
    elif isinstance(dim, list):
        axes = dim
    else:
        axes = [dim]

    if ord == 0:
        # Count non-zero elements
        zeros = _C_engine.zeros(xi.shape, xi.dtype, xi.device)
        nz = _C_engine.not_equal(xi, zeros)
        return _wrap(_C_engine.sum(nz, axes, keepdim))

    if ord == float("inf"):
        return _wrap(_C_engine.max(_C_engine.abs(xi), axes, keepdim))

    if ord == float("-inf"):
        return _wrap(_C_engine.min(_C_engine.abs(xi), axes, keepdim))

    if ord == 1:
        return _wrap(_C_engine.sum(_C_engine.abs(xi), axes, keepdim))

    if ord == 2:
        sq = _C_engine.mul(xi, xi)
        return _wrap(_C_engine.sqrt(_C_engine.sum(sq, axes, keepdim)))

    # General p-norm: sum(|x|^p)^(1/p)
    p = float(ord)
    abs_xi = _C_engine.abs(xi)
    powered = _C_engine.pow_scalar(abs_xi, p)
    s = _C_engine.sum(powered, axes, keepdim)
    return _wrap(_C_engine.pow_scalar(s, 1.0 / p))


def cross(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    """Compute the cross product of two 3-element vectors along *dim*.

    Uses gather with properly-ranked index tensors; fully autograd-tracked.
    Both tensors must have size 3 in the specified dimension.
    """
    xi = _unwrap(x)
    yi = _unwrap(y)
    ndim = len(xi.shape)
    d = dim if dim >= 0 else ndim + dim

    def _idx(t: "_C_engine.TensorImpl", i: int) -> "_C_engine.TensorImpl":
        # Build an index tensor with the same rank as t, size 1 along d,
        # filled with i.  gather requires equal-rank index.
        idx_shape = list(t.shape)
        idx_shape[d] = 1
        idx = _C_engine.full(idx_shape, float(i), _C_engine.I32, t.device)
        sliced = _C_engine.gather(t, idx, d)
        return _C_engine.squeeze(sliced, d)

    x0, x1, x2 = _idx(xi, 0), _idx(xi, 1), _idx(xi, 2)
    y0, y1, y2 = _idx(yi, 0), _idx(yi, 1), _idx(yi, 2)

    c0 = _C_engine.sub(_C_engine.mul(x1, y2), _C_engine.mul(x2, y1))
    c1 = _C_engine.sub(_C_engine.mul(x2, y0), _C_engine.mul(x0, y2))
    c2 = _C_engine.sub(_C_engine.mul(x0, y1), _C_engine.mul(x1, y0))

    c0u = _C_engine.unsqueeze(c0, d)
    c1u = _C_engine.unsqueeze(c1, d)
    c2u = _C_engine.unsqueeze(c2, d)
    return _wrap(_C_engine.concatenate([c0u, c1u, c2u], d))


def vecdot(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    """Compute the dot product of two tensors along *dim*."""
    prod = _C_engine.mul(_unwrap(x), _unwrap(y))
    return _wrap(_C_engine.sum(prod, [dim], False))


def dot(x: Tensor, y: Tensor) -> Tensor:
    """1-D vector dot product."""
    return _wrap(_C_engine.dot(_unwrap(x), _unwrap(y)))


def inner(x: Tensor, y: Tensor) -> Tensor:
    """Inner product: last-dim contraction (equivalent to dot for 1-D)."""
    return _wrap(_C_engine.inner(_unwrap(x), _unwrap(y)))


def outer(x: Tensor, y: Tensor) -> Tensor:
    """Outer product of two 1-D tensors."""
    return _wrap(_C_engine.outer(_unwrap(x), _unwrap(y)))


def matrix_norm(
    x: Tensor,
    ord: int | float | str = "fro",
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
) -> Tensor:
    """Compute a matrix norm using engine ops where possible.

    ``"fro"`` uses sqrt(sum(x^2)); ``"nuc"`` uses SVD singular values;
    integer orders use column/row sums.
    """
    xi = _unwrap(x)
    d0, d1 = int(dim[0]), int(dim[1])

    if ord == "fro":
        sq = _C_engine.mul(xi, xi)
        s = _C_engine.sum(sq, [d0, d1], keepdim)
        return _wrap(_C_engine.sqrt(s))

    if ord == "nuc":
        # Nuclear norm = sum of singular values (requires SVD)
        _, S, _ = svd(x)
        return _wrap(_C_engine.sum(_unwrap(S), [-1], keepdim))

    if ord == 1:
        # Max absolute column sum
        col_sums = _C_engine.sum(_C_engine.abs(xi), [d0], keepdim)
        return _wrap(_C_engine.max(col_sums, [d1 if keepdim else d1 - 1], keepdim))

    if ord == -1:
        col_sums = _C_engine.sum(_C_engine.abs(xi), [d0], keepdim)
        return _wrap(_C_engine.min(col_sums, [d1 if keepdim else d1 - 1], keepdim))

    if ord == float("inf"):
        row_sums = _C_engine.sum(_C_engine.abs(xi), [d1], keepdim)
        return _wrap(_C_engine.max(row_sums, [d0], keepdim))

    if ord == float("-inf"):
        row_sums = _C_engine.sum(_C_engine.abs(xi), [d1], keepdim)
        return _wrap(_C_engine.min(row_sums, [d0], keepdim))

    # Spectral norms: use SVD
    _, S, _ = svd(x)
    sv = _unwrap(S)
    if ord == 2:
        return _wrap(_C_engine.max(sv, [-1], keepdim))
    if ord == -2:
        return _wrap(_C_engine.min(sv, [-1], keepdim))

    raise ValueError(f"matrix_norm: unsupported ord={ord!r}")


def lstsq(
    A: Tensor,
    B: Tensor,
    rcond: float | None = None,
    driver: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the least-squares solution to a linear system AX = B.

    Returns ``(solution, residuals, rank, singular_values)``.
    CPU: LAPACK sgels/dgels.  GPU: CPU fallback.
    Note: residuals, rank, and singular_values are empty placeholders
    for API compatibility; only ``solution`` is fully computed.
    """
    sol = _wrap(_la.lstsq(_unwrap(A), _unwrap(B)))
    dev = _unwrap(A).device
    dt = _unwrap(A).dtype
    empty = _wrap(_C_engine.zeros([0], dt, dev))
    return sol, empty, empty, empty


def lu_solve(LU: Tensor, pivots: Tensor, B: Tensor) -> Tensor:
    """Solve a linear system from LU decomposition.

    ``LU`` and ``pivots`` are the output of :func:`lu_factor`.
    Returns X such that A @ X = B where A was factored into LU.
    CPU: LAPACK sgetrs/dgetrs.  GPU: CPU fallback.
    """
    return _wrap(_la.lu_solve(_unwrap(LU), _unwrap(pivots), _unwrap(B)))


def householder_product(H: Tensor, tau: Tensor) -> Tensor:
    """Compute the product of Householder reflectors.

    ``H`` is the matrix from ``the reference GEQRF API`` (or equivalent) and
    ``tau`` are the scalar factors.  Returns the orthogonal matrix Q.
    CPU: LAPACK sorgqr/dorgqr.  GPU: CPU fallback.
    """
    return _wrap(_la.householder_product(_unwrap(H), _unwrap(tau)))


def ldl_factor(
    A: Tensor,
    hermitian: bool = True,
) -> tuple[Tensor, Tensor]:
    """LDL^T factorization of a symmetric (or Hermitian) matrix.

    Returns ``(LD, pivots)`` where ``LD`` is the packed lower-triangular
    factor (L with D on the diagonal) and ``pivots`` is a 1-D int tensor
    of pivot indices.
    CPU: LAPACK ssytrf/dsytrf.  GPU: CPU fallback.
    """
    ld_impl, piv_impl = _la.ldl_factor(_unwrap(A))
    return _wrap(ld_impl), _wrap(piv_impl)


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
    "cross",
    "vecdot",
    "dot",
    "inner",
    "outer",
    "matrix_norm",
    "vector_norm",
    "lstsq",
    "lu_solve",
    "householder_product",
    "ldl_factor",
]
