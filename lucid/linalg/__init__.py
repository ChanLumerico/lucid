"""
lucid.linalg: linear algebra operations.
"""

import functools
from typing import Callable, cast

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
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
        """Decorator-generated wrapper that applies the surrounding behaviour to the wrapped callable."""
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
    r"""Compute the multiplicative inverse of a square matrix.

    Returns the unique matrix :math:`A^{-1}` such that

    .. math::

        A A^{-1} = A^{-1} A = I

    Inversion is performed via LU decomposition with partial pivoting.
    When the goal is to apply :math:`A^{-1}` to a known right-hand side,
    prefer :func:`solve` — explicit inversion is both slower and less
    numerically stable than back-substitution.

    Parameters
    ----------
    x : Tensor
        Square matrix of shape ``(*, n, n)`` (batch dims allowed).  Must
        be non-singular; raises a runtime error on singular input.

    Returns
    -------
    Tensor
        Inverse matrix of shape ``(*, n, n)`` with the same dtype as
        ``x``.

    Notes
    -----
    Internally computes :math:`PA = LU` and then solves ``LU X = P`` for
    ``X`` via two triangular sweeps.  Cost is :math:`O(n^3)` per matrix.
    For ill-conditioned ``A`` consider :func:`pinv` (SVD-based) for a
    more robust pseudo-inverse.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import inv
    >>> A = lucid.tensor([[4.0, 7.0], [2.0, 6.0]])
    >>> inv(A)
    Tensor([[ 0.6000, -0.7000],
            [-0.2000,  0.4000]])
    """
    return _la.inv(x)  # type: ignore[arg-type, return-value]


@_linalg_op
def det(x: Tensor) -> Tensor:
    r"""Compute the determinant of a square matrix.

    For a square matrix :math:`A \in \mathbb{R}^{n \times n}` returns the
    scalar :math:`\det(A)`.  The determinant is the signed volume of the
    parallelepiped spanned by the rows (or columns) of :math:`A`; it is
    non-zero if and only if :math:`A` is invertible.

    Parameters
    ----------
    x : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    Tensor
        Determinant of shape ``(*,)`` (one scalar per batch entry).

    Notes
    -----
    Computed as the product of the diagonal of :math:`U` from
    :math:`PA = LU` together with the sign of the row-permutation
    :math:`P`.  Cost is :math:`O(n^3)`.

    For numerically large matrices :math:`\det(A)` can overflow or
    underflow easily — prefer :func:`slogdet` which returns
    :math:`(\mathrm{sign},\,\log|\det A|)` and is stable across scales.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import det
    >>> det(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
    Tensor(-2.0)
    """
    return _la.det(x)  # type: ignore[arg-type, return-value]


@_linalg_op
def solve(A: Tensor, b: Tensor) -> Tensor:
    r"""Solve a square linear system :math:`AX = B`.

    Returns the unique solution :math:`X` of the system

    .. math::

        A X = B

    where :math:`A` is a non-singular square matrix.  The system is
    solved by LU decomposition with partial pivoting, which is both
    faster and more accurate than forming :math:`A^{-1}` explicitly.

    Parameters
    ----------
    A : Tensor
        Square coefficient matrix of shape ``(*, n, n)``.
    b : Tensor
        Right-hand side of shape ``(*, n, k)`` (multiple RHS columns) or
        ``(*, n)`` (single RHS vector).

    Returns
    -------
    Tensor
        Solution :math:`X` with the same shape as ``b``.

    Notes
    -----
    Algorithm: factor :math:`PA = LU`, then perform forward-substitution
    :math:`L Y = P B` followed by back-substitution :math:`U X = Y`.
    Total cost is :math:`O(n^3 + k n^2)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import solve
    >>> A = lucid.tensor([[3.0, 1.0], [1.0, 2.0]])
    >>> b = lucid.tensor([9.0, 8.0])
    >>> solve(A, b)
    Tensor([2.0000, 3.0000])
    """
    return _la.solve(A, b)  # type: ignore[arg-type, return-value]


def cholesky(x: Tensor, *, upper: bool = False) -> Tensor:
    r"""Cholesky decomposition of a symmetric positive-definite matrix.

    For a real symmetric positive-definite (SPD) matrix :math:`A`
    returns the unique lower-triangular factor :math:`L` with positive
    diagonal such that

    .. math::

        A = L L^\top \qquad \text{(or } A = U^\top U \text{ if } \texttt{upper=True}\text{)}.

    The Cholesky factor is the standard tool for solving SPD linear
    systems, sampling from a multivariate Gaussian, and computing
    matrix square roots.  It is roughly twice as fast as LU and avoids
    any pivoting.

    Parameters
    ----------
    x : Tensor
        Symmetric positive-definite matrix of shape ``(*, n, n)``.
    upper : bool, optional
        If ``True`` return the upper-triangular factor :math:`U` such
        that :math:`A = U^\top U`.  Default ``False`` returns the
        lower-triangular factor :math:`L`.

    Returns
    -------
    Tensor
        Triangular Cholesky factor of shape ``(*, n, n)``.

    Notes
    -----
    Algorithm: LAPACK ``potrf`` via Apple Accelerate on CPU, MLX on GPU.
    Cost is :math:`O(n^3 / 3)`, half the work of LU.

    Backward is implemented in Python via Murray's (2016) formula

    .. math::

        \frac{\partial L}{\partial A} \,=\, \mathrm{sym}\!\big(L^{-\top}\,\Phi(L^\top G)\,L^{-1}\big),

    where :math:`\Phi` zeros the strict upper triangle and halves the
    diagonal.  Two triangular solves implement the inversion implicitly.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cholesky
    >>> A = lucid.tensor([[4.0, 2.0], [2.0, 3.0]])  # SPD
    >>> L = cholesky(A)
    >>> L
    Tensor([[2.0000, 0.0000],
            [1.0000, 1.4142]])
    >>> L @ L.T
    Tensor([[4.0000, 2.0000],
            [2.0000, 3.0000]])
    """
    return cast(Tensor, _CholeskyAutograd.apply(x, upper))


from lucid.autograd.function import Function as _AutogradFunction
from lucid.autograd.function import FunctionCtx


class _CholeskyAutograd(_AutogradFunction):
    """Custom-autograd wrapper around the engine's non-differentiable
    cholesky_op. Forward calls the engine; backward computes the input
    gradient via Murray (2016)."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, upper: bool) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        out = _wrap(_la.cholesky(_unwrap(x), upper))
        ctx.save_for_backward(out)
        ctx.upper = bool(upper)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        (factor,) = ctx.saved_tensors  # L (upper=False) or U (upper=True)
        upper: bool = cast(bool, ctx.upper)

        # Normalise to lower-triangular form L; gradient w.r.t. that L.
        if upper:
            L = factor.mT
            gL = grad_out.mT
        else:
            L = factor
            gL = grad_out

        n: int = int(L.shape[-1])
        eye_n = lucid.eye(n, dtype=L.dtype)
        if L.device != "cpu":
            eye_n = eye_n.to(L.device)
        # Mask gL to its lower triangle — the strictly upper half doesn't
        # contribute to L (which is lower-triangular by construction).
        gL_tril = lucid.tril(gL)
        # Phi(M): tril(M) with diagonal halved.
        M = lucid.matmul(L.mT, gL_tril)
        Phi = lucid.tril(M) - 0.5 * (M * eye_n)

        # S = L^{-T} @ Phi @ L^{-1}, computed via two triangular solves.
        # Step 1: Y = L^{-T} Phi  →  solve L^T Y = Phi (upper=True against L^T).
        Y = solve_triangular(L.mT, Phi, upper=True)
        # Step 2: Z = Y L^{-1}.  Take transposes: Z^T = L^{-T} Y^T, so solve
        # L^T Z^T = Y^T then transpose back.
        Z = solve_triangular(L.mT, Y.mT, upper=True).mT
        # Murray's formula gives the Riemannian gradient for a symmetric
        # input matrix A (it implicitly assumes ∂A[i,j] = ∂A[j,i]).
        # gradcheck treats A as a general matrix and perturbs each element
        # independently.  Cholesky uses only tril(A), so:
        #   · upper triangle of A  → gradient = 0
        #   · diagonal of A        → gradient = (Z+Z^T)[i,i]/2  (unchanged)
        #   · lower off-diagonal   → gradient = (Z+Z^T)[i,j]  (×2 vs Murray)
        # When upper=False, cholesky reads only tril(A):
        #   · upper off-diagonal → gradient 0
        #   · diagonal           → sym[i,i]  (unchanged)
        #   · lower off-diagonal → 2·sym[i,j]
        #   ⟹ grad_A = 2·tril(sym) − diag(sym)
        # When upper=True, cholesky reads only triu(A) (we normalised to L above,
        # so Z is still in lower-triangular space, but the active elements of A
        # are in the upper triangle):
        #   · lower off-diagonal → gradient 0
        #   · diagonal           → sym[i,i]
        #   · upper off-diagonal → 2·sym[i,j]
        #   ⟹ grad_A = 2·triu(sym) − diag(sym)
        sym = 0.5 * (Z + Z.mT)
        diag_vals = sym.diagonal(dim1=-2, dim2=-1)  # (..., n)
        if upper:
            grad_A = 2.0 * lucid.triu(sym) - lucid.diag_embed(diag_vals)
        else:
            grad_A = 2.0 * lucid.tril(sym) - lucid.diag_embed(diag_vals)
        return grad_A


def norm(
    x: Tensor,
    ord: int | float | str | None = None,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Compute a vector or matrix norm.

    Generic norm dispatcher that delegates to :func:`vector_norm` or
    :func:`matrix_norm` based on the input rank and reduction axes.
    The default behaviour computes the Frobenius norm of a matrix input
    and the Euclidean (:math:`\ell_2`) norm of a vector input:

    .. math::

        \|x\|_2 \,=\, \Big(\sum_i |x_i|^2\Big)^{1/2}, \qquad
        \|A\|_F \,=\, \Big(\sum_{i,j} |A_{ij}|^2\Big)^{1/2}.

    Parameters
    ----------
    x : Tensor
        Input tensor.  May be a vector ``(n,)``, a matrix ``(m, n)``, or
        higher-rank with explicit ``dim``.
    ord : int, float, str or None, optional
        Order of the norm.  Vector orders: ``0``, ``1``, ``2`` (default
        when None), ``inf``, ``-inf``, or any positive real.  Matrix
        orders: ``"fro"``, ``"nuc"``, ``1``, ``-1``, ``2``, ``-2``,
        ``inf``, ``-inf``.
    dim : int, list of int or None, optional
        Reduction axis (or pair of axes for matrix norms).  ``None``
        reduces over all elements.
    keepdim : bool, optional
        If ``True``, retains reduced dimensions with size 1.

    Returns
    -------
    Tensor
        Norm value(s); shape depends on ``dim`` / ``keepdim``.

    Notes
    -----
    Many norm orders (spectral, nuclear) require an SVD and therefore
    cost :math:`O(\min(m,n) \cdot mn)`; entry-wise norms reduce in a
    single pass.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import norm
    >>> norm(lucid.tensor([3.0, 4.0]))
    Tensor(5.0)
    """
    return _wrap(_la.norm(_unwrap(x)))


# ── SVD with backward ─────────────────────────────────────────────────────────
#
# The SVD returns three tensors (U, S, Vh).  Lucid's Python Function
# mechanism supports only single-output backward registration, so we use one
# Function per output.  Gradients from each path accumulate correctly
# (dA from U-path + dA from S-path + dA from Vh-path = total dA).
#
# KEY DESIGN: pre-computed TensorImpl objects (not Tensor) are passed as
# extra args so the engine never sees them as differentiable inputs.
# Only `A` (the first arg) is a Tensor with requires_grad=True; the others
# are TensorImpl and are skipped by `isinstance(a, Tensor)` in _make_apply.
# This avoids spurious graph edges between the three Function wrappers.
#
# Backward formula (Giles 2008, extended to rectangular A(m×n), k=min(m,n)):
#   F[i,j] = s_i / (s_i² - s_j²)  for i≠j,  F[i,i] = 0
#   dA from S: U diag(G_S) Vh
#   dA from U: U (F ⊙ U^T G_U) Vh + (I_m - U U^T) G_U Σ^{-1} Vh    [if m>k]
#   dA from Vh: U (F ⊙ -(Vh G_V)^T) Vh + U Σ^{-1} G_Vh (I_n - Vh^T Vh) [if n>k]


def _svd_loewner(S: Tensor) -> Tensor:
    """Build the Loewner matrix F[i,j] = s_i/(s_i²-s_j²) for i≠j."""
    k = int(S.shape[-1])
    Si = S.unsqueeze(-1)  # (..., k, 1)
    Sj = S.unsqueeze(-2)  # (..., 1, k)
    denom = Si * Si - Sj * Sj  # (..., k, k)
    eye_k = lucid.eye(k, dtype=S.dtype, device=S.device)
    safe_denom = denom + eye_k  # avoid div-by-zero on diagonal
    F = Si / safe_denom * (1.0 - eye_k)  # zero diagonal
    return F


class _SVDSGrad(_AutogradFunction):
    """Backward: singular-value contribution  dA = U diag(G_S) Vh."""

    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        ctx: FunctionCtx,
        A: Tensor,
        u_impl: _C_engine.TensorImpl,
        s_impl: _C_engine.TensorImpl,
        vh_impl: _C_engine.TensorImpl,
    ) -> Tensor:
        # Save TensorImpl references (not Tensor) to avoid autograd entanglement.
        ctx.u_impl = u_impl
        ctx.vh_impl = vh_impl
        return _wrap(s_impl)

    @staticmethod
    def backward(ctx: FunctionCtx, G_S: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        U = _wrap(ctx.u_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        Vh = _wrap(ctx.vh_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        return U @ lucid.diag_embed(G_S) @ Vh


class _SVDUGrad(_AutogradFunction):
    """Backward: left-singular-vector contribution to dA."""

    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        ctx: FunctionCtx,
        A: Tensor,
        u_impl: _C_engine.TensorImpl,
        s_impl: _C_engine.TensorImpl,
        vh_impl: _C_engine.TensorImpl,
    ) -> Tensor:
        ctx.u_impl = u_impl
        ctx.s_impl = s_impl
        ctx.vh_impl = vh_impl
        return _wrap(u_impl)

    @staticmethod
    def backward(ctx: FunctionCtx, G_U: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        U = _wrap(ctx.u_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        S = _wrap(ctx.s_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        Vh = _wrap(ctx.vh_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        k = int(S.shape[-1])
        m = int(U.shape[-2])
        F = _svd_loewner(S)
        UtgU = U.mT @ G_U  # (..., k, k)
        K = F * UtgU
        dA = U @ K @ Vh
        if m > k:
            S_inv = lucid.diag_embed(1.0 / S)
            proj_gU = G_U - U @ UtgU
            dA = dA + proj_gU @ S_inv @ Vh
        return dA


class _SVDVhGrad(_AutogradFunction):
    """Backward: right-singular-vector (Vh) contribution to dA."""

    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        ctx: FunctionCtx,
        A: Tensor,
        u_impl: _C_engine.TensorImpl,
        s_impl: _C_engine.TensorImpl,
        vh_impl: _C_engine.TensorImpl,
    ) -> Tensor:
        ctx.u_impl = u_impl
        ctx.s_impl = s_impl
        ctx.vh_impl = vh_impl
        return _wrap(vh_impl)

    @staticmethod
    def backward(ctx: FunctionCtx, G_Vh: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        U = _wrap(ctx.u_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        S = _wrap(ctx.s_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        Vh = _wrap(ctx.vh_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        k = int(S.shape[-1])
        n = int(Vh.shape[-1])
        F = _svd_loewner(S)
        gV = G_Vh.mT  # (..., n, k)
        VtgV = Vh @ gV  # (..., k, k)
        K = F * (-VtgV.mT)
        dA = U @ K @ Vh
        if n > k:
            S_inv = lucid.diag_embed(1.0 / S)
            proj_gVh = G_Vh - (G_Vh @ Vh.mT) @ Vh
            dA = dA + U @ S_inv @ proj_gVh
        return dA


def svd(x: Tensor, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    r"""Singular value decomposition of a matrix.

    Factorizes any (possibly rectangular) matrix :math:`A \in
    \mathbb{R}^{m \times n}` into

    .. math::

        A \,=\, U \,\Sigma\, V^\top,

    where :math:`U` and :math:`V` are orthogonal and
    :math:`\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_k)` with
    :math:`\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_k \ge 0` and
    :math:`k = \min(m, n)`.  The SVD is the foundation of low-rank
    approximation, pseudo-inverse, matrix rank, condition number, and
    PCA.

    Parameters
    ----------
    x : Tensor
        Input matrix of shape ``(*, m, n)`` (batch dims allowed).
    full_matrices : bool, optional
        If ``True`` (default), :math:`U` is ``(m, m)`` and
        :math:`V^\top` is ``(n, n)`` — the full orthogonal factors.
        If ``False``, returns the reduced SVD with :math:`U` shaped
        ``(m, k)`` and :math:`V^\top` shaped ``(k, n)``.

    Returns
    -------
    U : Tensor
        Left singular vectors, shape ``(*, m, m)`` or ``(*, m, k)``.
    S : Tensor
        Singular values in descending order, shape ``(*, k)``.
    Vh : Tensor
        Right singular vectors (conjugate-transposed), shape
        ``(*, n, n)`` or ``(*, k, n)``.

    Notes
    -----
    Backward is implemented via three separate ``Function`` wrappers —
    one per output — so gradients from :math:`U`, :math:`S`, and
    :math:`V^\top` accumulate into the input via Giles' (2008) formula
    using the Loewner matrix :math:`F_{ij} = \sigma_i / (\sigma_i^2 -
    \sigma_j^2)`.  Gradients blow up when singular values are nearly
    repeated — degenerate or rank-deficient inputs are not
    differentiation-friendly.

    Cost is :math:`O(\min(m,n)^2 \cdot \max(m,n))`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import svd
    >>> A = lucid.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    >>> U, S, Vh = svd(A, full_matrices=False)
    >>> S
    Tensor([2.0000, 1.0000])
    """
    _svd_result = _la.svd(_unwrap(x))
    u_impl: _C_engine.TensorImpl
    s_impl: _C_engine.TensorImpl
    vh_impl: _C_engine.TensorImpl
    u_impl, s_impl, vh_impl = _svd_result
    if not _C_engine.grad_enabled() or not x.requires_grad:
        return _wrap(u_impl), _wrap(s_impl), _wrap(vh_impl)
    # Pass TensorImpl (not Tensor) so _make_apply ignores them in the
    # differentiable-input scan — no spurious cross-edges in the graph.
    U = _SVDUGrad.apply(x, u_impl, s_impl, vh_impl)
    S = _SVDSGrad.apply(x, u_impl, s_impl, vh_impl)
    Vh = _SVDVhGrad.apply(x, u_impl, s_impl, vh_impl)
    return U, S, Vh  # type: ignore[return-value]


def svdvals(x: Tensor) -> Tensor:
    r"""Compute only the singular values of a matrix.

    Returns the singular values :math:`\sigma_1 \ge \cdots \ge
    \sigma_k \ge 0` (with :math:`k = \min(m, n)`) of an input matrix
    :math:`A` without forming :math:`U` or :math:`V^\top`.  Equivalent
    to ``svd(A)[1]`` but avoids the work of constructing the singular
    vectors.

    Parameters
    ----------
    x : Tensor
        Input matrix of shape ``(*, m, n)``.

    Returns
    -------
    Tensor
        Singular values in descending order, shape ``(*, k)``.

    Notes
    -----
    When gradients are required this routes through the full
    :func:`svd` so backward still works.  Without ``requires_grad``,
    the engine kernel skips assembly of the singular vectors and is
    roughly :math:`2\times` faster.  Useful for computing
    :func:`matrix_rank`, :func:`cond`, or the nuclear / spectral norms.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import svdvals
    >>> svdvals(lucid.tensor([[3.0, 0.0], [0.0, 4.0]]))
    Tensor([4.0000, 3.0000])
    """
    if _C_engine.grad_enabled() and x.requires_grad:
        _, S, _ = svd(x)
        return S
    result = _la.svd(_unwrap(x), False)
    if isinstance(result, (list, tuple)):
        return _wrap(result[0])
    return _wrap(result)


# ── QR with backward ─────────────────────────────────────────────────────────
#
# Strategy: express R via the Cholesky decomposition of A^T A.
#
# For A = Q R (m≥n, R upper triangular):
#   A^T A = R^T R  (since Q^T Q = I)
#
# The lower Cholesky factor L = chol(A^T A) satisfies L L^T = A^T A,
# with positive diagonal (by definition).  The relationship between L and R
# from LAPACK's dgeqrf is: R = D L^T  where D = diag(sign(diag(R))).
#
# This lets us route the R backward through the existing (correct) Cholesky
# backward without implementing the Gu-Eisenstat formula, which requires
# positive-diagonal R and fails for LAPACK's sign convention.
#
# Q backward: from A = Q R, Q = A R^{-1} for square A.  For thin QR (m>n),
# Q doesn't have a simple closed-form inverse through A.  We route Q through
# the same Cholesky path or via projection.
#
# For simplicity, the combined QR backward is implemented in a single
# Function wrapper that handles both Q and R gradients together.


class _QRCombinedGrad(_AutogradFunction):
    """Joint backward for QR: accumulates G_Q and G_R and computes dA.

    Both Q and R gradients flow to the same A via the Cholesky-based formula:
      L = chol(A^T A),  R = D L^T,  Q = A R^{-1}  (conceptually)

    In practice, because LAPACK's QR can have negative diagonal, we use the
    correct chain-rule path:
      G_B = chol_backward(L, G_L)   where L = chol(A^T A)
      dA  = 2 A G_B                 where G_B is symmetric
    combined with the Q-path:
      dA += G_Q R^{-T} + Q tril(G_Q^T Q - 0) R^{-T}  (Stiefel projection)

    This is split into two Function wrappers that each handle one output;
    their contributions accumulate in A.grad.
    """


class _QRRGrad(_AutogradFunction):
    """Backward: R contribution.  Uses Cholesky route for correctness with
    LAPACK's sign convention (negative diagonal R elements are allowed)."""

    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        ctx: FunctionCtx,
        A: Tensor,
        q_impl: _C_engine.TensorImpl,
        r_impl: _C_engine.TensorImpl,
    ) -> Tensor:
        ctx.r_impl = r_impl
        return _wrap(r_impl)

    @staticmethod
    def backward(ctx: FunctionCtx, G_R: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        # R backward via Cholesky of A^T A:
        #   R = D L^T  where L = chol(A^T A), D = diag(sign(diag(R)))
        #   G_L = G_R.mT @ D  (chain rule through R = D L^T)
        #   G_B = chol_backward(L, G_L)
        #   dA  = 2 A G_B  (chain rule through B = A^T A)
        #
        # But we don't have A here — use the functional Cholesky backward
        # via Murray's formula directly on R.
        #
        # Simpler equivalent: use the fact that R^T R = A^T A.
        # dA from G_R:
        #   ∂f/∂A = A G_B + A G_B^T = 2 A G_B  (G_B symmetric)
        # where G_B is the gradient of f w.r.t. B = R^T R:
        #   ∂f/∂B[i,j] = Σ_{k,l} G_R[k,l] ∂R[k,l]/∂B[i,j]
        #
        # Using Murray's Cholesky backward directly:
        # G_L[a,b] = G_R[b,a] * D[b,b]  (= G_R.mT @ D)
        # G_B = chol_backward(L, G_L) via the two triangular solves
        R = _wrap(ctx.r_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        n = int(R.shape[-1])
        diag_R = R.diagonal(dim1=-2, dim2=-1)
        # sign of diagonal: +1 or -1
        D = lucid.diag_embed(diag_R / diag_R.abs().clamp(min=1e-12))  # sign matrix
        # L = R^T @ D  (lower triangular, positive diagonal)
        # Derivation: R = D1 R_pos (D1 = sign matrix), R_pos = L^T
        # → L = R_pos^T = (D1 R)^T = R^T D1 = R.mT @ D
        L = (R.mT @ D).detach()
        # G_L = G_R.mT @ D
        G_L = G_R.mT @ D
        # Murray's Cholesky backward: G_B = sym(L^{-T} Phi(L^T G_L) L^{-1})
        eye_n = lucid.eye(n, dtype=L.dtype, device=L.device)
        M = L.mT @ G_L
        Phi = lucid.tril(M) - 0.5 * (M * eye_n)
        Y = solve_triangular(L.mT, Phi, upper=True)
        Z = solve_triangular(L.mT, Y.mT, upper=True).mT
        G_B = (Z + Z.mT) * 0.5  # symmetric
        # dA = 2 * ctx_A * G_B — but we don't have A here.
        # Recover A: A = Q R but we don't have Q stored in ctx.
        # Use A = L^T D R ??? No: L = D R^T → L^T = R D^T = R D (D is diagonal & symmetric)
        # So A = Q R and R = D L^T → A = Q D L^T.
        # G_B is the gradient w.r.t. A^T A, so dA = 2 A G_B.
        # But we need A. Store it in forward.
        raise RuntimeError(
            "Internal error: _QRRGrad.backward called without A stored in ctx. "
            "Use _QRRGradWithA which stores A."
        )


class _QRRGradWithA(_AutogradFunction):
    """Backward: R contribution to dA via Cholesky of A^T A."""

    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        ctx: FunctionCtx,
        A: Tensor,
        r_impl: _C_engine.TensorImpl,
    ) -> Tensor:
        ctx.A_impl = _unwrap(A)  # store A as TensorImpl
        ctx.r_impl = r_impl
        return _wrap(r_impl)

    @staticmethod
    def backward(ctx: FunctionCtx, G_R: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        A = _wrap(ctx.A_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        R = _wrap(ctx.r_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        n = int(R.shape[-1])
        # D = sign matrix of R diagonal
        diag_R = R.diagonal(dim1=-2, dim2=-1)
        abs_d = diag_R.abs()
        safe_abs_d = lucid.where(abs_d < 1e-12, lucid.ones_like(abs_d), abs_d)
        sign_d = diag_R / safe_abs_d
        D = lucid.diag_embed(sign_d)
        # L = R.mT @ D  (lower triangular, positive diagonal)
        # Derivation: R_pos = D @ R (sign-normalised), L = R_pos^T = R^T D = R.mT @ D
        L = (R.mT @ D).detach()
        # G_L = G_R_pos^T = (D @ G_R)^T = G_R^T @ D  (since D is symmetric)
        G_L = G_R.mT @ D
        # Murray's Cholesky backward → G_B = sym(L^{-T} Phi(L^T G_L) L^{-1})
        eye_n = lucid.eye(n, dtype=L.dtype, device=L.device)
        M = L.mT @ G_L
        Phi = lucid.tril(M) - 0.5 * (M * eye_n)
        Y = solve_triangular(L.mT, Phi, upper=True)
        Z = solve_triangular(L.mT, Y.mT, upper=True).mT
        G_B = (Z + Z.mT) * 0.5
        # dA = 2 A G_B  (from B = A^T A)
        return (2.0 * A) @ G_B


class _QRQGrad(_AutogradFunction):
    """Backward: Q contribution to dA.

    From A = Q R with Q^T Q = I (Stiefel manifold constraint):
      dA from G_Q = G_Q R^{-T} - Q skew(Q^T G_Q) R^{-T}
                  = (G_Q - Q sym(Q^T G_Q)) R^{-T}
    where sym(M) = (M + M^T)/2.

    For m > n: add the off-range projection (I - QQ^T) G_Q R^{-T}.
    """

    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        ctx: FunctionCtx,
        A: Tensor,
        q_impl: _C_engine.TensorImpl,
        r_impl: _C_engine.TensorImpl,
    ) -> Tensor:
        ctx.q_impl = q_impl
        ctx.r_impl = r_impl
        return _wrap(q_impl)

    @staticmethod
    def backward(ctx: FunctionCtx, G_Q: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        Q = _wrap(ctx.q_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        R = _wrap(ctx.r_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        m, n = int(Q.shape[-2]), int(Q.shape[-1])
        # Stiefel gradient: project G_Q onto tangent space at Q
        QtGQ = Q.mT @ G_Q  # n×n
        sym_QtGQ = (QtGQ + QtGQ.mT) * 0.5
        numerator = G_Q - Q @ sym_QtGQ  # m×n: tangent-space component
        dA = solve_triangular(R, numerator.mT, upper=True).mT
        if m > n:
            # Off-range: (I - QQ^T) G_Q R^{-T}
            proj = G_Q - Q @ (Q.mT @ G_Q)
            dA = dA + solve_triangular(R, proj.mT, upper=True).mT
        return dA


def qr(x: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    r"""QR decomposition of a matrix.

    Factorizes a matrix :math:`A \in \mathbb{R}^{m \times n}` as

    .. math::

        A \,=\, Q\,R,

    where :math:`Q` has orthonormal columns (:math:`Q^\top Q = I`) and
    :math:`R` is upper-triangular.  Used for orthogonalizing a basis,
    solving least-squares (:math:`\min \|Ax - b\|_2`), and computing
    eigenvalues via QR iteration.

    Parameters
    ----------
    x : Tensor
        Input matrix of shape ``(*, m, n)``.
    mode : {"reduced", "complete", "r"}, optional
        ``"reduced"`` (default): :math:`Q` is ``(m, k)`` and :math:`R`
        is ``(k, n)`` with :math:`k = \min(m, n)`.
        ``"complete"``: :math:`Q` is ``(m, m)`` and :math:`R` is
        ``(m, n)``.
        ``"r"``: return only :math:`R` (``Q`` is an empty tensor).

    Returns
    -------
    Q : Tensor
        Orthogonal factor.
    R : Tensor
        Upper-triangular factor.

    Notes
    -----
    Computed via Householder reflections (LAPACK ``geqrf``).  Cost is
    :math:`O(2 m n^2 - \tfrac{2}{3} n^3)` for :math:`m \ge n`.

    The diagonal of :math:`R` may carry arbitrary signs (LAPACK
    convention) — the factorization is unique only up to a diagonal
    sign matrix.  Backward routes :math:`R` through a Cholesky of
    :math:`A^\top A` (sign-robust) and :math:`Q` through the
    Stiefel-manifold tangent projection.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import qr
    >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> Q, R = qr(A)
    >>> Q.T @ Q
    Tensor([[1.0000, 0.0000],
            [0.0000, 1.0000]])
    """
    q_impl, r_impl = _la.qr(_unwrap(x))
    if not _C_engine.grad_enabled() or not x.requires_grad:
        return _wrap(q_impl), _wrap(r_impl)
    Q = _QRQGrad.apply(x, q_impl, r_impl)
    R = _QRRGradWithA.apply(x, r_impl)
    return Q, R  # type: ignore[return-value]


def matrix_power(x: Tensor, n: int) -> Tensor:
    r"""Raise a square matrix to an integer power.

    Computes :math:`A^n` for an integer exponent :math:`n`:

    .. math::

        A^n \,=\, \begin{cases}
            \underbrace{A A \cdots A}_{n\ \text{times}} & n > 0 \\
            I & n = 0 \\
            (A^{-1})^{|n|} & n < 0
        \end{cases}

    Parameters
    ----------
    x : Tensor
        Square matrix of shape ``(*, m, m)``.
    n : int
        Integer exponent.  Negative values require :math:`A` to be
        invertible.

    Returns
    -------
    Tensor
        :math:`A^n`, shape ``(*, m, m)``.

    Notes
    -----
    Uses binary exponentiation (repeated squaring) so the cost is
    :math:`O(\log |n|)` matrix multiplies rather than :math:`|n|`.
    Implemented as a Python composite over :func:`matmul` and
    :func:`inv` so autograd flows naturally — the engine
    ``matrix_power_op`` is not differentiable.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_power
    >>> A = lucid.tensor([[1.0, 1.0], [0.0, 1.0]])
    >>> matrix_power(A, 5)
    Tensor([[1.0000, 5.0000],
            [0.0000, 1.0000]])
    """
    if not isinstance(n, int):
        raise TypeError(f"matrix_power exponent must be int, got {type(n).__name__}")

    sh: tuple[int, ...] = tuple(_unwrap(x).shape)
    if len(sh) < 2 or sh[-1] != sh[-2]:
        raise ValueError(
            f"matrix_power requires a square matrix in the last two dims, got {sh}"
        )

    if n == 0:
        # Identity broadcast to the input's batch shape.
        eye_2d: Tensor = lucid.eye(int(sh[-1]), dtype=x.dtype)
        if len(sh) == 2:
            return eye_2d
        return lucid.broadcast_to(eye_2d, tuple(sh))

    base: Tensor = cast(Tensor, inv(x)) if n < 0 else x
    exponent: int = -n if n < 0 else n
    if exponent == 1:
        return base

    # Standard binary exponentiation: result starts at base if the lowest
    # bit is set, otherwise it gets multiplied in on the first set bit.
    result: Tensor | None = None
    cur: Tensor = base
    while exponent > 0:
        if exponent & 1:
            result = cur if result is None else lucid.matmul(result, cur)
        exponent >>= 1
        if exponent:
            cur = lucid.matmul(cur, cur)
    assert result is not None  # exponent was non-zero on entry
    return result


def pinv(x: Tensor) -> Tensor:
    r"""Moore-Penrose pseudo-inverse of a matrix.

    Returns the unique matrix :math:`A^+` satisfying the four
    Moore-Penrose conditions

    .. math::

        A A^+ A = A, \quad A^+ A A^+ = A^+, \quad
        (A A^+)^\top = A A^+, \quad (A^+ A)^\top = A^+ A.

    For a thin SVD :math:`A = U\Sigma V^\top`, the pseudo-inverse is

    .. math::

        A^+ \,=\, V\,\Sigma^+\,U^\top, \qquad
        \Sigma^+_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > \tau \\ 0 & \text{else} \end{cases}.

    Parameters
    ----------
    x : Tensor
        Input matrix of shape ``(*, m, n)``.  Need not be square or
        full-rank.

    Returns
    -------
    Tensor
        Pseudo-inverse of shape ``(*, n, m)``.

    Notes
    -----
    For square, full-rank matrices ``pinv`` is equivalent to :func:`inv`
    — Lucid routes that case through ``inv`` to keep autograd active.
    For rectangular or rank-deficient inputs the SVD-based engine
    kernel is invoked (no backward).

    The pseudo-inverse provides the least-squares solution of
    :math:`Ax = b` even when :math:`A` is singular: :math:`x = A^+ b`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import pinv
    >>> A = lucid.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    >>> pinv(A) @ A
    Tensor([[1.0000, 0.0000],
            [0.0000, 1.0000]])
    """
    sh: tuple[int, ...] = tuple(_unwrap(x).shape)
    if len(sh) >= 2 and sh[-1] == sh[-2]:
        # Square matrix → ``pinv ≡ inv`` for full-rank input. Falling back to
        # the engine pinv would lose autograd; ``inv`` keeps it.
        return cast(Tensor, inv(x))
    return _wrap(_la.pinv(_unwrap(x)))


@_linalg_op
def eig(x: Tensor) -> tuple[Tensor, Tensor]:
    r"""Eigenvalue decomposition of a general square matrix.

    For a general (not necessarily symmetric) square matrix :math:`A`
    returns eigenvalues :math:`\lambda_i` and right eigenvectors
    :math:`v_i` such that

    .. math::

        A v_i \,=\, \lambda_i\, v_i,
        \qquad A \,=\, V\,\Lambda\,V^{-1}.

    For matrices with complex eigenvalues, the result is in general
    complex-valued.  When :math:`A` is known to be symmetric / Hermitian
    prefer :func:`eigh` — it is faster, more stable, and produces real
    eigenvalues with orthogonal eigenvectors.

    Parameters
    ----------
    x : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    eigenvalues : Tensor
        Tensor of shape ``(*, n)``.  Real for real-spectrum matrices,
        complex otherwise.
    eigenvectors : Tensor
        Tensor of shape ``(*, n, n)`` whose columns are the right
        eigenvectors.

    Notes
    -----
    Backed by LAPACK ``geev``.  Cost is :math:`O(n^3)`.  This op
    currently has **no autograd support** — gradients through
    eigendecomposition of a general matrix are notoriously unstable
    near defective spectra and not implemented.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import eig
    >>> A = lucid.tensor([[2.0, 0.0], [0.0, 3.0]])
    >>> w, V = eig(A)
    >>> w
    Tensor([2.0000, 3.0000])
    """
    vals, vecs = _la.eig(_unwrap(x))
    return _wrap(vals), _wrap(vecs)


def eigvals(x: Tensor) -> Tensor:
    r"""Compute only the eigenvalues of a general square matrix.

    Returns the roots :math:`\lambda_1, \ldots, \lambda_n` of the
    characteristic polynomial :math:`\det(A - \lambda I) = 0` without
    computing the eigenvectors.  Equivalent to ``eig(A)[0]`` but skips
    the eigenvector assembly.

    Parameters
    ----------
    x : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    Tensor
        Eigenvalues of shape ``(*, n)``.

    Notes
    -----
    Cost is :math:`O(n^3)` and dominated by the Hessenberg reduction.
    No autograd support — use :func:`eigvalsh` for symmetric / Hermitian
    matrices when gradients are required.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import eigvals
    >>> eigvals(lucid.tensor([[2.0, 0.0], [0.0, 3.0]]))
    Tensor([2.0000, 3.0000])
    """
    vals, _ = _la.eig(_unwrap(x))
    return _wrap(vals)


# ── Eigh with backward ────────────────────────────────────────────────────────
#
# For symmetric A = V diag(w) V^T, given G_w and G_V:
#   F[i,j] = 1/(w_i − w_j)  for i≠j,  F[i,i] = 0    (Loewner matrix)
#   dA from w: V diag(G_w) V^T
#   dA from V: V (F ⊙ (V^T G_V)) V^T  (then symmetrised)
#
# Split into two Function wrappers so the engine accumulates contributions.


class _EighWGrad(_AutogradFunction):
    """Backward: eigenvalue contribution  dA = V diag(G_w) V^T.

    w_impl / V_impl are passed as TensorImpl (not Tensor) so _make_apply
    skips them in the differentiable-input scan — only A gets a gradient edge.
    """

    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        ctx: FunctionCtx,
        A: Tensor,
        w_impl: _C_engine.TensorImpl,
        V_impl: _C_engine.TensorImpl,
    ) -> Tensor:
        ctx.V_impl = V_impl  # store TensorImpl, not Tensor
        return _wrap(w_impl)

    @staticmethod
    def backward(ctx: FunctionCtx, G_w: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        V = _wrap(ctx.V_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        return V @ lucid.diag_embed(G_w) @ V.mT


class _EighVGrad(_AutogradFunction):
    """Backward: eigenvector contribution  dA = sym(V (F ⊙ V^T G_V) V^T)."""

    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        ctx: FunctionCtx,
        A: Tensor,
        w_impl: _C_engine.TensorImpl,
        V_impl: _C_engine.TensorImpl,
    ) -> Tensor:
        ctx.w_impl = w_impl
        ctx.V_impl = V_impl
        return _wrap(V_impl)

    @staticmethod
    def backward(ctx: FunctionCtx, G_V: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        w = _wrap(ctx.w_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        V = _wrap(ctx.V_impl)  # type: ignore[arg-type]  # ctx attr is TensorImpl at runtime
        k = int(w.shape[-1])
        Wi = w.unsqueeze(-1)
        Wj = w.unsqueeze(-2)
        denom = Wi - Wj
        eye_k = lucid.eye(k, dtype=w.dtype, device=w.device)
        safe_denom = denom + eye_k
        F = (1.0 / safe_denom) * (1.0 - eye_k)
        VtGV = V.mT @ G_V
        inner = F * VtGV
        dA = V @ inner @ V.mT
        return (dA + dA.mT) * 0.5


def eigh(x: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    r"""Eigendecomposition of a Hermitian / symmetric matrix.

    Returns the eigenvalues and orthonormal eigenvectors of a real
    symmetric (or complex Hermitian) matrix :math:`A`.  Eigenvalues are
    returned in **ascending order** and eigenvectors form an orthogonal
    matrix :math:`V` such that

    .. math::

        A \,=\, V \,\Lambda\, V^\top,

    where :math:`\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)`
    with :math:`\lambda_1 \le \cdots \le \lambda_n`.

    Parameters
    ----------
    x : Tensor
        Square Hermitian / symmetric matrix of shape ``(*, n, n)``.
    UPLO : str, optional
        ``"L"`` (default) reads only the lower triangle of ``x``;
        ``"U"`` reads only the upper triangle.  The other triangle is
        ignored, so a non-Hermitian input is accepted as long as the
        chosen triangle holds the correct values.

    Returns
    -------
    eigenvalues : Tensor
        Real-valued tensor of shape ``(*, n)`` in ascending order.
    eigenvectors : Tensor
        Tensor of shape ``(*, n, n)`` whose columns are the
        corresponding orthonormal eigenvectors.

    Notes
    -----
    Prefer this over :func:`eig` whenever ``A`` is symmetric / Hermitian
    — it is faster, numerically more stable, and guarantees real
    eigenvalues with orthogonal eigenvectors (:math:`V^\top V = I`).

    Backward uses the Loewner-matrix formula
    :math:`F_{ij} = 1/(\lambda_i - \lambda_j)`:

    .. math::

        \frac{\partial \mathcal{L}}{\partial A}
        \,=\, V\,\big(\mathrm{diag}(G_\lambda) + F \odot (V^\top G_V)\big)\,V^\top,

    symmetrised to enforce the symmetry of :math:`A`.  Gradients blow up
    near repeated eigenvalues.

    Implementation: LAPACK ``syevd`` on the CPU stream (via Apple
    Accelerate); MLX on the GPU stream.  Both run in :math:`O(n^3)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import eigh
    >>> A = lucid.tensor([[2.0, 1.0], [1.0, 3.0]])
    >>> w, V = eigh(A)
    >>> w
    Tensor([1.3820, 3.6180])
    """
    w_impl, V_impl = _la.eigh(_unwrap(x))
    if not _C_engine.grad_enabled() or not x.requires_grad:
        return _wrap(w_impl), _wrap(V_impl)
    w = _EighWGrad.apply(x, w_impl, V_impl)
    V = _EighVGrad.apply(x, w_impl, V_impl)
    return w, V  # type: ignore[return-value]


def eigvalsh(x: Tensor, UPLO: str = "L") -> Tensor:
    r"""Eigenvalues of a Hermitian / symmetric matrix.

    Returns only the real eigenvalues :math:`\lambda_1 \le \cdots \le
    \lambda_n` of a symmetric (real) or Hermitian (complex) matrix
    without forming the eigenvectors.  Equivalent to ``eigh(A)[0]``
    but skips eigenvector assembly when no gradients are requested.

    Parameters
    ----------
    x : Tensor
        Symmetric / Hermitian matrix of shape ``(*, n, n)``.
    UPLO : str, optional
        ``"L"`` reads the lower triangle (default), ``"U"`` the upper.

    Returns
    -------
    Tensor
        Eigenvalues in ascending order, shape ``(*, n)``.

    Notes
    -----
    When ``x.requires_grad`` is true the call routes through
    :func:`eigh` so that backward via the Loewner formula remains
    available.  Otherwise the engine kernel is invoked directly.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import eigvalsh
    >>> eigvalsh(lucid.tensor([[2.0, 1.0], [1.0, 3.0]]))
    Tensor([1.3820, 3.6180])
    """
    if _C_engine.grad_enabled() and x.requires_grad:
        w, _ = eigh(x, UPLO)
        return w
    vals, _ = _la.eigh(_unwrap(x))
    return _wrap(vals)


@_linalg_op
def lu_factor(A: Tensor) -> tuple[Tensor, Tensor]:
    r"""LU factorization with partial pivoting (packed form).

    Computes the packed LU factorization

    .. math::

        P\,A \,=\, L\,U,

    where :math:`P` is a row-permutation matrix, :math:`L` is
    unit-lower-triangular, and :math:`U` is upper-triangular.  The
    result is returned in LAPACK's packed format suitable for repeated
    solves via :func:`lu_solve`.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    LU : Tensor
        Packed factorization of shape ``(*, n, n)``.  :math:`U`
        occupies the diagonal and above; :math:`L` (without its unit
        diagonal) occupies the strict lower triangle.
    pivots : Tensor
        ``int32`` tensor of shape ``(*, n)`` containing 1-based pivot
        indices (LAPACK convention).

    Notes
    -----
    Backed by LAPACK ``getrf``.  Cost is :math:`O(\tfrac{2}{3} n^3)`.

    Use :func:`lu_factor` + :func:`lu_solve` instead of :func:`solve`
    when the same coefficient matrix :math:`A` is reused with many
    right-hand sides — factorization is shared.  For the explicit
    :math:`(P, L, U)` triple see :func:`lu`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lu_factor, lu_solve
    >>> A = lucid.tensor([[2.0, 1.0], [4.0, 7.0]])
    >>> LU, piv = lu_factor(A)
    >>> b = lucid.tensor([[3.0], [13.0]])
    >>> lu_solve(LU, piv, b)
    Tensor([[0.8000],
            [1.4000]])
    """
    lu, pivots = _la.lu_factor(_unwrap(A))
    return _wrap(lu), _wrap(pivots)


# ── Pure-Python compositions ───────────────────────────────────────────────────


def slogdet(A: Tensor) -> tuple[Tensor, Tensor]:
    r"""Sign and natural log of the absolute determinant.

    Returns the pair :math:`(\mathrm{sign},\, \log|\det A|)` such that

    .. math::

        \det(A) \,=\, \mathrm{sign} \cdot \exp(\log|\det A|).

    Numerically stable for matrices whose determinant would overflow
    or underflow if computed directly — for instance, large covariance
    matrices in log-likelihood calculations.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    sign : Tensor
        :math:`\pm 1` (or :math:`0` for singular matrices), shape
        ``(*,)``.
    logabsdet : Tensor
        :math:`\log|\det A|`, shape ``(*,)``.  Returns :math:`-\infty`
        for singular inputs.

    Notes
    -----
    Computed via LU as :math:`\log|\det A| = \sum_i \log|U_{ii}|` with
    the sign accumulated from row swaps.  Cost is :math:`O(n^3)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import slogdet
    >>> sign, logabs = slogdet(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
    >>> sign, logabs
    (Tensor(-1.0), Tensor(0.6931))
    """
    d = cast(Tensor, det(A))
    sign = _wrap(_C_engine.sign(_unwrap(d)))
    logabsdet = _wrap(_C_engine.log(_C_engine.abs(_unwrap(d))))
    return sign, logabsdet


def matrix_rank(
    A: Tensor,
    tol: float | None = None,
    hermitian: bool = False,
) -> Tensor:
    r"""Compute the numerical rank of a matrix.

    The numerical rank is the number of singular values of :math:`A`
    that exceed a tolerance:

    .. math::

        \mathrm{rank}(A) \,=\, |\{\,i : \sigma_i > \tau\,\}|.

    Parameters
    ----------
    A : Tensor
        Input matrix of shape ``(*, m, n)``.
    tol : float or None, optional
        Threshold below which singular values are treated as zero.
        ``None`` (default) uses :math:`\max(m, n) \cdot \varepsilon
        \cdot \sigma_{\max}` with :math:`\varepsilon` the float32
        machine epsilon (:math:`\approx 1.19 \times 10^{-7}`).
    hermitian : bool, optional
        If ``True`` exploit Hermitian structure for a cheaper
        eigen-based computation.  Currently unused (SVD path is always
        taken).

    Returns
    -------
    Tensor
        Integer scalar (or batched scalars) holding the rank.

    Notes
    -----
    Computed via :func:`svd` so cost is :math:`O(\min(m,n)^2 \cdot
    \max(m,n))`.  Choosing the tolerance is application-specific —
    consider scaling by :math:`\|A\|` when the singular values span
    many orders of magnitude.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_rank
    >>> matrix_rank(lucid.tensor([[1.0, 2.0], [2.0, 4.0]]))
    Tensor(1)
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
    r"""Compute the condition number of a matrix.

    The condition number under norm :math:`\|\cdot\|_p` is

    .. math::

        \kappa_p(A) \,=\, \|A\|_p \,\|A^{-1}\|_p.

    For the spectral (:math:`p = 2`) norm this simplifies to the ratio
    of the largest to smallest singular value:

    .. math::

        \kappa_2(A) \,=\, \sigma_{\max}(A) \,/\, \sigma_{\min}(A).

    The condition number quantifies how sensitive the solution of
    :math:`Ax = b` is to perturbations in :math:`A` or :math:`b`.

    Parameters
    ----------
    A : Tensor
        Input matrix of shape ``(*, m, n)``.
    p : int, float, str or None, optional
        Norm order.  ``None`` (default) and ``2`` use the spectral
        norm via SVD; ``-2`` returns the reciprocal :math:`\sigma_{\min}
        / \sigma_{\max}`.  Other orders dispatch to :func:`norm`.

    Returns
    -------
    Tensor
        Condition number, shape ``(*,)``.

    Notes
    -----
    A condition number near :math:`1/\varepsilon_{\mathrm{mach}}`
    indicates numerical singularity.  Non-spectral orders require an
    explicit :func:`inv`, so prefer ``p = 2`` for rank-deficient or
    rectangular matrices.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cond
    >>> cond(lucid.tensor([[1.0, 0.0], [0.0, 1e-6]]))
    Tensor(1000000.0)
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
    return _wrap(
        _C_engine.mul(
            _unwrap(norm(A, ord=p)),
            _unwrap(norm(inv(A), ord=p)),  # type: ignore[arg-type]
        )
    )


def multi_dot(tensors: list[Tensor]) -> Tensor:
    r"""Multiply a sequence of matrices as a single chained product.

    Computes the product of a list of matrices

    .. math::

        A_1 \, A_2 \, \cdots \, A_n,

    associating left-to-right.  Optimal parenthesization can
    substantially reduce flops for chains with widely varying inner
    dimensions; the current implementation associates left-to-right
    (the most common case is already locally optimal).

    Parameters
    ----------
    tensors : list of Tensor
        Sequence of at least one matrix.  Inner dimensions must agree
        (``tensors[i].shape[-1] == tensors[i+1].shape[-2]``).

    Returns
    -------
    Tensor
        The chained matrix product.

    Notes
    -----
    For long chains, choosing the optimal split can reduce work from
    :math:`O(\sum_i d_i d_{i+1} d_{i+2})` (left-to-right) to a
    significantly smaller bound found via dynamic programming.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import multi_dot
    >>> A = lucid.tensor([[1.0, 2.0]])
    >>> B = lucid.tensor([[3.0], [4.0]])
    >>> C = lucid.tensor([[5.0]])
    >>> multi_dot([A, B, C])
    Tensor([[55.0000]])
    """
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
    r"""Solve a triangular linear system by back/forward substitution.

    Solves the system :math:`AX = B` (or :math:`XA = B`) in which the
    coefficient matrix :math:`A` is triangular.  For upper-triangular
    :math:`A` the system is solved by back-substitution starting from
    the last row; for lower-triangular :math:`A` by forward
    substitution from the first row.  Either direction runs in
    :math:`O(n^2 k)` time and is numerically stable when the diagonal
    of :math:`A` is well-conditioned.

    Parameters
    ----------
    A : Tensor
        Triangular coefficient matrix of shape ``(*, n, n)``.  Only
        the relevant triangle is read; the other half is ignored.
    B : Tensor
        Right-hand side of shape ``(*, n, k)`` (or ``(*, n)``).
    upper : bool, keyword-only, optional
        If ``True`` (default) :math:`A` is upper-triangular; if
        ``False`` lower-triangular.
    left : bool, keyword-only, optional
        If ``True`` (default) solves :math:`AX = B`; if ``False``
        solves :math:`XA = B` via transposition.
    unitriangular : bool, keyword-only, optional
        If ``True`` the diagonal of :math:`A` is treated as all-ones
        regardless of its stored values (LAPACK's "unit-diagonal"
        mode).

    Returns
    -------
    Tensor
        Solution :math:`X` shaped like :math:`B`.

    Notes
    -----
    Backed by LAPACK ``trsm``.  Triangular solves are the workhorse
    used inside Cholesky, LU, and QR back-substitution paths.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import solve_triangular
    >>> A = lucid.tensor([[2.0, 1.0], [0.0, 3.0]])  # upper
    >>> b = lucid.tensor([[5.0], [9.0]])
    >>> solve_triangular(A, b, upper=True)
    Tensor([[1.0000],
            [3.0000]])
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
    r"""Construct a Vandermonde matrix from a 1-D vector.

    Given a 1-D input :math:`x = (x_1, \ldots, x_n)`, returns the
    :math:`n \times N` matrix whose :math:`j`-th column is a power of
    :math:`x`:

    .. math::

        V_{ij} \,=\, x_i^{\,j} \quad (\text{increasing})
        \qquad\text{or}\qquad
        V_{ij} \,=\, x_i^{\,N-1-j} \quad (\text{decreasing, default}).

    Vandermonde matrices arise naturally in polynomial fitting and
    interpolation: the columns are the basis :math:`\{1, x, x^2,
    \ldots\}` evaluated at the data points.

    Parameters
    ----------
    x : Tensor
        1-D input of length :math:`n`.
    N : int or None, optional
        Number of columns.  Defaults to :math:`n` (square output).
    increasing : bool, optional
        If ``True`` powers increase left-to-right (column 0 is
        :math:`x^0`).  Default ``False`` matches the classical
        convention used in polynomial regression.

    Returns
    -------
    Tensor
        :math:`n \times N` Vandermonde matrix.

    Notes
    -----
    Vandermonde matrices become highly ill-conditioned as
    :math:`N` grows (condition number grows exponentially) — for
    polynomial regression beyond degree :math:`\sim 10` prefer an
    orthogonal-polynomial basis or QR-based fitting.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import vander
    >>> vander(lucid.tensor([1.0, 2.0, 3.0]), N=3, increasing=True)
    Tensor([[1.0000, 1.0000, 1.0000],
            [1.0000, 2.0000, 4.0000],
            [1.0000, 3.0000, 9.0000]])
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
    r"""Compute a vector :math:`p`-norm along an axis.

    Reduces along ``dim`` (or all elements if ``dim is None``) using

    .. math::

        \|x\|_p \,=\, \Big(\sum_i |x_i|^{\,p}\Big)^{1/p}

    for any positive real :math:`p`.  Special-cased values:

    * :math:`p = 0` — count of non-zero entries (not a true norm).
    * :math:`p = 1` — :math:`\sum_i |x_i|`.
    * :math:`p = 2` — Euclidean norm.
    * :math:`p = +\infty` — :math:`\max_i |x_i|`.
    * :math:`p = -\infty` — :math:`\min_i |x_i|`.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    ord : int or float, optional
        Order of the norm.  Default ``2``.
    dim : int, list of int or None, optional
        Axis or axes to reduce.  ``None`` flattens first.
    keepdim : bool, optional
        If ``True``, reduced dimensions are retained with size 1.
    dtype : optional
        Currently unused; reserved for future accumulation-dtype
        control.

    Returns
    -------
    Tensor
        Norm along the specified axes.

    Notes
    -----
    All operations are routed through autograd-aware engine kernels, so
    gradients flow naturally even for non-integer :math:`p` (via
    :math:`p`-power and root).

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import vector_norm
    >>> vector_norm(lucid.tensor([3.0, 4.0]))
    Tensor(5.0)
    >>> vector_norm(lucid.tensor([1.0, -2.0, 3.0]), ord=1)
    Tensor(6.0)
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
    r"""Compute the cross product of two 3-D vectors.

    Returns the standard 3-D vector cross product

    .. math::

        x \times y \,=\, \big(x_2 y_3 - x_3 y_2,\;
                              x_3 y_1 - x_1 y_3,\;
                              x_1 y_2 - x_2 y_1\big).

    Parameters
    ----------
    x : Tensor
        First operand.  Must have size 3 along ``dim``.
    y : Tensor
        Second operand.  Same shape as ``x``.
    dim : int, optional
        Axis along which the 3 components are stored.  Default ``-1``.

    Returns
    -------
    Tensor
        Cross product, same shape as inputs.

    Notes
    -----
    Implemented via :func:`gather` + element-wise products so that
    autograd flows through naturally.  Recall that
    :math:`x \times y` is orthogonal to both :math:`x` and :math:`y`,
    with magnitude :math:`\|x\|\,\|y\|\,\sin\theta`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cross
    >>> cross(lucid.tensor([1.0, 0.0, 0.0]), lucid.tensor([0.0, 1.0, 0.0]))
    Tensor([0.0000, 0.0000, 1.0000])
    """
    xi = _unwrap(x)
    yi = _unwrap(y)
    ndim = len(xi.shape)
    d = dim if dim >= 0 else ndim + dim

    def _idx(t: _C_engine.TensorImpl, i: int) -> _C_engine.TensorImpl:
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
    r"""Compute a batched vector dot product along an axis.

    Reduces the chosen axis with a sum of element-wise products:

    .. math::

        (x \cdot y)_{\ldots} \,=\, \sum_{k} x_{\ldots, k, \ldots}\,
                                              y_{\ldots, k, \ldots}.

    Parameters
    ----------
    x : Tensor
        First operand.
    y : Tensor
        Second operand, broadcast-compatible with ``x``.
    dim : int, optional
        Axis to contract.  Default ``-1``.

    Returns
    -------
    Tensor
        Reduced tensor with ``dim`` removed.

    Notes
    -----
    Equivalent to ``(x * y).sum(dim=dim)``.  Useful for computing many
    independent dot products in one shot (e.g., per-row inner products
    of two matrices).

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import vecdot
    >>> x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> y = lucid.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    >>> vecdot(x, y)
    Tensor([-2.0000,  5.0000])
    """
    prod = _C_engine.mul(_unwrap(x), _unwrap(y))
    return _wrap(_C_engine.sum(prod, [dim], False))


def dot(x: Tensor, y: Tensor) -> Tensor:
    r"""Dot product of two 1-D tensors.

    Computes the scalar inner product

    .. math::

        x \cdot y \,=\, \sum_{i} x_i\, y_i.

    Parameters
    ----------
    x, y : Tensor
        1-D tensors of equal length.

    Returns
    -------
    Tensor
        Scalar dot product.

    Notes
    -----
    For higher-rank tensors use :func:`inner` (last-dim contraction)
    or :func:`vecdot` (explicit axis), and :func:`matmul` for matrix
    products.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import dot
    >>> dot(lucid.tensor([1.0, 2.0, 3.0]), lucid.tensor([4.0, 5.0, 6.0]))
    Tensor(32.0)
    """
    return _wrap(_C_engine.dot(_unwrap(x), _unwrap(y)))


def inner(x: Tensor, y: Tensor) -> Tensor:
    r"""Inner product over the last axes of two tensors.

    Behaves like :func:`dot` for 1-D inputs and like a generalised
    outer-then-contract for higher ranks.

    Parameters
    ----------
    x, y : Tensor
        Tensors whose last dimensions match.

    Returns
    -------
    Tensor
        Result with shape ``x.shape[:-1] + y.shape[:-1]``.

    Notes
    -----
    Contracts the last axis of both operands and sums:

    .. math::

        \mathrm{inner}(x, y)_{\ldots, \ldots'} \,=\,
            \sum_{k} x_{\ldots, k}\, y_{\ldots', k}.

    For 1-D inputs, this is the standard vector dot product.  For higher
    ranks the leading axes of ``x`` and ``y`` are kept independent and
    appear consecutively in the output — handy for batched dot products
    when the contraction axis is the trailing one.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import inner
    >>> inner(lucid.tensor([1.0, 2.0]), lucid.tensor([3.0, 4.0]))
    Tensor(11.0)
    """
    return _wrap(_C_engine.inner(_unwrap(x), _unwrap(y)))


def outer(x: Tensor, y: Tensor) -> Tensor:
    r"""Outer product of two 1-D tensors.

    Parameters
    ----------
    x : Tensor
        1-D tensor of length :math:`m`.
    y : Tensor
        1-D tensor of length :math:`n`.

    Returns
    -------
    Tensor
        Matrix of shape ``(m, n)``.

    Notes
    -----
    Computes the rank-1 matrix

    .. math::

        (x \otimes y)_{ij} \,=\, x_i\, y_j,

    yielding the same result as :math:`x\,y^\top` viewed as a 2-D array.
    Outer products underpin rank-1 updates (BFGS), Kronecker-product
    factorisations, and certain attention patterns.  Unlike
    :func:`matmul`, the inputs are not contracted — every pair
    :math:`(x_i, y_j)` produces an independent entry.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import outer
    >>> outer(lucid.tensor([1.0, 2.0]), lucid.tensor([3.0, 4.0]))
    Tensor([[3.0000, 4.0000],
            [6.0000, 8.0000]])
    """
    return _wrap(_C_engine.outer(_unwrap(x), _unwrap(y)))


def matrix_norm(
    x: Tensor,
    ord: int | float | str = "fro",
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
) -> Tensor:
    r"""Compute a matrix norm.

    Reduces the trailing two axes of an input to a scalar matrix norm.
    Supported orders:

    * ``"fro"`` — Frobenius norm
      :math:`\|A\|_F = \big(\sum_{ij} |A_{ij}|^2\big)^{1/2}`.
    * ``"nuc"`` — nuclear norm
      :math:`\|A\|_* = \sum_i \sigma_i(A)` (sum of singular values).
    * ``1`` / ``-1`` — max / min absolute column sum.
    * ``inf`` / ``-inf`` — max / min absolute row sum.
    * ``2`` / ``-2`` — largest / smallest singular value (spectral
      norm and its reciprocal).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(*, m, n)``.
    ord : int, float or str, optional
        Norm order.  Default ``"fro"``.
    dim : tuple of two ints, optional
        Axis pair identifying the matrix dimensions.  Default
        ``(-2, -1)``.
    keepdim : bool, optional
        If ``True``, reduced dims are retained with size 1.

    Returns
    -------
    Tensor
        Matrix norm of each batch.

    Notes
    -----
    Spectral and nuclear norms require an SVD and so cost
    :math:`O(\min(m,n)^2 \max(m,n))`.  Entry-wise norms reduce in a
    single pass.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_norm
    >>> A = lucid.tensor([[3.0, 4.0], [0.0, 0.0]])
    >>> matrix_norm(A, ord="fro")
    Tensor(5.0)
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
    r"""Solve a least-squares linear system.

    Finds :math:`X` minimising the squared residual

    .. math::

        \min_X \,\|A\,X - B\|_2^2,

    where :math:`A \in \mathbb{R}^{m \times n}` may be over- or
    underdetermined.  For full-rank :math:`A` and :math:`m \ge n` the
    solution is unique:

    .. math::

        X \,=\, (A^\top A)^{-1} A^\top B,

    obtained more stably via QR or SVD without forming the normal
    equations.

    Parameters
    ----------
    A : Tensor
        Coefficient matrix of shape ``(*, m, n)``.
    B : Tensor
        Right-hand side of shape ``(*, m, k)`` (or ``(*, m)``).
    rcond : float or None, optional
        Cutoff for small singular values (passed to the underlying
        driver).  ``None`` selects the default driver heuristic.
    driver : str or None, optional
        Solver choice (``"gels"``, ``"gelsy"``, ``"gelsd"``, ...).
        ``None`` lets the engine pick (currently ``gels``).

    Returns
    -------
    solution : Tensor
        Least-squares solution of shape ``(*, n, k)``.
    residuals : Tensor
        Sum-of-squared residuals.  Currently an empty placeholder for
        API compatibility.
    rank : Tensor
        Effective rank of :math:`A`.  Currently empty placeholder.
    singular_values : Tensor
        Singular values of :math:`A`.  Currently empty placeholder.

    Notes
    -----
    Backed by LAPACK ``gels`` / ``gelsd`` on the CPU stream; GPU calls
    fall back to CPU.  Only ``solution`` is fully populated in the
    current implementation; the remaining outputs exist for shape
    compatibility with the reference API.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lstsq
    >>> A = lucid.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    >>> b = lucid.tensor([[6.0], [9.0], [12.0]])
    >>> sol, *_ = lstsq(A, b)
    >>> sol
    Tensor([[3.0000],
            [3.0000]])
    """
    sol = _wrap(_la.lstsq(_unwrap(A), _unwrap(B)))
    dev = _unwrap(A).device
    dt = _unwrap(A).dtype
    empty = _wrap(_C_engine.zeros([0], dt, dev))
    return sol, empty, empty, empty


def lu_solve(LU: Tensor, pivots: Tensor, B: Tensor) -> Tensor:
    r"""Solve a linear system from a precomputed LU factorization.

    Given the packed factorization :math:`PA = LU` returned by
    :func:`lu_factor`, solves

    .. math::

        A\,X \,=\, B

    by applying the permutation and performing two triangular solves
    (forward + back substitution).

    Parameters
    ----------
    LU : Tensor
        Packed LU factor of shape ``(*, n, n)`` from :func:`lu_factor`.
    pivots : Tensor
        Pivot indices of shape ``(*, n)`` from :func:`lu_factor`
        (1-based, LAPACK convention).
    B : Tensor
        Right-hand side of shape ``(*, n, k)`` (or ``(*, n)``).

    Returns
    -------
    Tensor
        Solution :math:`X`, same shape as ``B``.

    Notes
    -----
    Backed by LAPACK ``getrs``.  Cost per solve is :math:`O(n^2 k)` —
    much cheaper than a fresh :func:`solve` (:math:`O(n^3)`) when the
    same :math:`A` is reused.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lu_factor, lu_solve
    >>> A = lucid.tensor([[3.0, 1.0], [1.0, 2.0]])
    >>> LU, piv = lu_factor(A)
    >>> b = lucid.tensor([[9.0], [8.0]])
    >>> lu_solve(LU, piv, b)
    Tensor([[2.0000],
            [3.0000]])
    """
    return _wrap(_la.lu_solve(_unwrap(LU), _unwrap(pivots), _unwrap(B)))


def householder_product(H: Tensor, tau: Tensor) -> Tensor:
    r"""Reconstruct an orthogonal matrix from Householder reflectors.

    Computes the implicit product

    .. math::

        Q \,=\, H_1\,H_2\,\cdots\,H_k,
        \qquad H_i \,=\, I - \tau_i\, v_i\, v_i^\top,

    where each :math:`v_i` is a Householder vector stored in the
    :math:`i`-th column of the packed input ``H`` and :math:`\tau_i`
    is its scalar factor.  This is the standard way to materialise the
    :math:`Q` factor from a packed QR (``geqrf``) result.

    Parameters
    ----------
    H : Tensor
        Packed reflector matrix of shape ``(*, m, k)`` — columns
        contain the Householder vectors (typically the output of an
        unpacked ``geqrf``).
    tau : Tensor
        Scalar factors of shape ``(*, k)``.

    Returns
    -------
    Tensor
        Orthogonal matrix :math:`Q` of shape ``(*, m, k)``.

    Notes
    -----
    Backed by LAPACK ``orgqr``.  Cost is :math:`O(m k^2)`.  Useful
    when a routine returns the packed Householder form (cheaper to
    store) but the explicit :math:`Q` is needed downstream.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import qr, householder_product
    >>> A = lucid.randn(4, 3)
    >>> Q, R = qr(A, mode="reduced")
    >>> # The same Q can be reconstructed from packed Householder reflectors
    >>> # returned by lower-level geqrf-style factorisations.
    >>> Q.shape
    (4, 3)
    """
    return _wrap(_la.householder_product(_unwrap(H), _unwrap(tau)))


def ldl_factor(
    A: Tensor,
    hermitian: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""LDL factorization of a symmetric (or Hermitian) matrix.

    For a real symmetric matrix :math:`A` (possibly indefinite),
    computes a Bunch-Kaufman block factorization

    .. math::

        A \,=\, L\,D\,L^\top,

    where :math:`L` is unit-lower-triangular and :math:`D` is
    block-diagonal with :math:`1 \times 1` or :math:`2 \times 2`
    blocks.  Unlike :func:`cholesky`, this factorization exists for
    *indefinite* symmetric matrices (e.g., saddle-point systems).

    Parameters
    ----------
    A : Tensor
        Symmetric / Hermitian matrix of shape ``(*, n, n)``.
    hermitian : bool, optional
        If ``True`` (default), treat ``A`` as Hermitian (conjugate
        symmetric in the complex case).

    Returns
    -------
    LD : Tensor
        Packed factor of shape ``(*, n, n)``.  The strict lower
        triangle holds :math:`L`; the diagonal holds :math:`D`'s
        entries (:math:`2 \times 2` blocks are stored in the
        sub-diagonal).
    pivots : Tensor
        ``int32`` pivot indices.  Positive entries indicate
        :math:`1 \times 1` blocks; pairs of negative entries flag a
        :math:`2 \times 2` block.

    Notes
    -----
    Backed by LAPACK ``sytrf``.  Cost is :math:`O(n^3 / 3)`.  Pair
    with :func:`ldl_solve` for solving symmetric indefinite systems.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import ldl_factor
    >>> A = lucid.tensor([[1.0, 2.0], [2.0, 3.0]])
    >>> LD, piv = ldl_factor(A)
    """
    ld_impl, piv_impl = _la.ldl_factor(_unwrap(A))
    return _wrap(ld_impl), _wrap(piv_impl)


# ── *_ex variants — return (result, info) instead of raising ───────────────
#
# LAPACK's ``*_ex`` family writes a non-zero ``info`` integer when the
# matrix is singular / not positive definite / etc., instead of erroring.
# Lucid's existing ``cholesky`` / ``inv`` / ``solve`` raise ``LucidError``
# in those cases (translated from the LAPACK status by the engine
# layer).  We re-shape that into the ``info`` return contract by catching
# the engine error and emitting a non-zero ``info`` tensor.
#
# ``info == 0``  → success; ``result`` is meaningful.
# ``info != 0``  → numerical failure; ``result`` is a *shape-correct
#                  placeholder* (zeros).  Callers must check ``info``
#                  before trusting the result, exactly as in LAPACK.


def _info_zero(A: Tensor) -> Tensor:
    """Build a scalar (or batched) int32 ``info`` tensor of zeros aligned
    with the leading-batch dims of ``A`` (everything except the trailing
    two matrix dims).  Mirrors LAPACK's batched-info contract."""
    batch = list(A.shape[:-2])
    if not batch:
        return lucid.zeros(tuple(), dtype=lucid.int32, device=A.device)
    return lucid.zeros(*batch, dtype=lucid.int32, device=A.device)


def cholesky_ex(
    A: Tensor,
    *,
    upper: bool = False,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Cholesky factorization with an explicit success flag.

    Variant of :func:`cholesky` that, instead of raising when the
    input fails to be positive-definite, returns the factor together
    with an integer ``info`` code following LAPACK's convention:

    * ``info == 0`` — success; :math:`L` (or :math:`U`) is meaningful.
    * ``info != 0`` — numerical failure; :math:`L` is zero-filled.

    Parameters
    ----------
    A : Tensor
        Candidate SPD matrix of shape ``(*, n, n)``.
    upper : bool, keyword-only, optional
        If ``True`` return the upper-triangular factor :math:`U` such
        that :math:`A = U^\top U`.  Default ``False``.
    check_errors : bool, keyword-only, optional
        If ``True``, re-raise the underlying engine error instead of
        emitting a non-zero ``info`` — useful while debugging.

    Returns
    -------
    L : Tensor
        Cholesky factor (or zeros on failure), shape ``(*, n, n)``.
    info : Tensor
        ``int32`` status, scalar or shape ``(*,)`` matching the batch
        of ``A``.

    Notes
    -----
    Designed for code paths where a failed Cholesky is an expected
    event (e.g., trial steps in trust-region optimisers).  Callers
    must inspect ``info`` before trusting ``L``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cholesky_ex
    >>> A = lucid.tensor([[4.0, 2.0], [2.0, 3.0]])
    >>> L, info = cholesky_ex(A)
    >>> int(info)
    0
    """
    try:
        L = cholesky(A, upper=upper)
        return L, _info_zero(A)
    except Exception:
        if check_errors:
            raise
        zero_L = lucid.zeros(*A.shape, dtype=A.dtype, device=A.device)
        info = _info_zero(A) + 1  # non-zero sentinel
        return zero_L, info


def inv_ex(A: Tensor, *, check_errors: bool = False) -> tuple[Tensor, Tensor]:
    r"""Matrix inverse with an explicit success flag.

    Variant of :func:`inv` that returns an ``info`` code instead of
    raising on a singular input.

    * ``info == 0`` — success; ``Ainv`` is :math:`A^{-1}`.
    * ``info != 0`` — :math:`A` was singular; ``Ainv`` is zero-filled.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(*, n, n)``.
    check_errors : bool, keyword-only, optional
        If ``True``, re-raise the underlying engine error instead of
        emitting a non-zero ``info``.

    Returns
    -------
    Ainv : Tensor
        Inverse (or zero placeholder) of shape ``(*, n, n)``.
    info : Tensor
        ``int32`` status flag.

    Notes
    -----
    Useful in algorithms that occasionally probe near-singular
    matrices (e.g., iterative refinement, regularisation grid
    searches) without wanting to wrap every call in a ``try``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import inv_ex
    >>> Ainv, info = inv_ex(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
    >>> int(info)
    0
    """
    try:
        return cast(Tensor, inv(A)), _info_zero(A)
    except Exception:
        if check_errors:
            raise
        zero_inv = lucid.zeros(*A.shape, dtype=A.dtype, device=A.device)
        info = _info_zero(A) + 1
        return zero_inv, info


def solve_ex(
    A: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Solve :math:`AX = B` with an explicit success flag.

    Variant of :func:`solve` that returns an ``info`` code instead of
    raising on a singular coefficient matrix.

    * ``info == 0`` — success; :math:`X` is the unique solution.
    * ``info != 0`` — :math:`A` was singular; :math:`X` is zero-filled.

    Parameters
    ----------
    A : Tensor
        Coefficient matrix of shape ``(*, n, n)``.
    B : Tensor
        Right-hand side of shape ``(*, n, k)`` (or ``(*, n)``).
    left : bool, keyword-only, optional
        Currently must be ``True``.  ``False`` (``X A = B``) is not yet
        implemented and raises ``NotImplementedError``.
    check_errors : bool, keyword-only, optional
        If ``True``, re-raise the underlying engine error instead of
        emitting a non-zero ``info``.

    Returns
    -------
    X : Tensor
        Solution (or zero placeholder) shaped like ``B``.
    info : Tensor
        ``int32`` status flag.

    Notes
    -----
    Implementation calls :func:`solve` under the hood and converts the
    raised exception into the ``info`` flag.  This is the recommended
    form when calling from inside a batched / jit-compiled / vmapped
    routine where raising would break control flow; manually inspect
    ``info`` afterwards and decide whether to recover.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import solve_ex
    >>> A = lucid.tensor([[3.0, 1.0], [1.0, 2.0]])
    >>> b = lucid.tensor([9.0, 8.0])
    >>> X, info = solve_ex(A, b)
    >>> int(info)
    0
    """
    if not left:
        raise NotImplementedError("solve_ex: only left=True is supported")
    try:
        return cast(Tensor, solve(A, B)), _info_zero(A)
    except Exception:
        if check_errors:
            raise
        zero_X = lucid.zeros(*B.shape, dtype=B.dtype, device=B.device)
        info = _info_zero(A) + 1
        return zero_X, info


# ── Full LU decomposition (P, L, U) ────────────────────────────────────────


def lu(A: Tensor, *, pivot: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    r"""Full LU decomposition with explicit factors :math:`(P, L, U)`.

    Decomposes a square matrix :math:`A` as

    .. math::

        A \,=\, P\,L\,U,

    where :math:`P` is a row permutation, :math:`L` is unit-lower
    triangular (ones on the diagonal), and :math:`U` is upper
    triangular.  Unlike :func:`lu_factor` (which returns a packed
    factor + integer pivots), this routine returns the three factors
    as explicit dense tensors — convenient for inspection or for
    re-using :math:`P,L,U` in downstream linear-algebra expressions.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(n, n)``.  Batched inputs are not
        yet exposed through the Python wrapper (will raise).
    pivot : bool, keyword-only, optional
        Must be ``True`` (the default).  ``False`` would request an
        unpivoted LU; Lucid does not currently ship that kernel and
        raises ``NotImplementedError``.

    Returns
    -------
    P : Tensor
        Permutation matrix of shape ``(n, n)``.
    L : Tensor
        Unit-lower-triangular factor of shape ``(n, n)``.
    U : Tensor
        Upper-triangular factor of shape ``(n, n)``.

    Notes
    -----
    Implemented as a Python composite over :func:`lu_factor` followed
    by explicit triangular masking and pivot-to-matrix conversion.
    Cost is :math:`O(\tfrac{2}{3} n^3)`, dominated by the underlying
    :math:`LU` factorization.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lu
    >>> A = lucid.tensor([[2.0, 1.0], [4.0, 7.0]])
    >>> P, L, U = lu(A)
    >>> (P @ L @ U).numpy()
    array([[2., 1.],
           [4., 7.]], dtype=float32)
    """
    if not pivot:
        raise NotImplementedError("lu: pivot=False is not supported")
    sh = tuple(_unwrap(A).shape)
    if len(sh) < 2 or sh[-1] != sh[-2]:
        raise ValueError(f"lu requires a square matrix in the last two dims, got {sh}")
    n = int(sh[-1])

    _lu_result = cast(tuple["Tensor", "Tensor"], lu_factor(A))
    LU, pivots = _lu_result

    # Split the packed LU into L (unit-lower) and U (upper).
    eye_n = lucid.eye(n, dtype=A.dtype, device=A.device)
    L_strict = lucid.tril(LU) - lucid.tril(LU) * eye_n  # zero out diagonal
    L = L_strict + eye_n  # add unit diagonal
    U = lucid.triu(LU)

    # Reconstruct the permutation matrix from the LAPACK pivot vector.
    # LAPACK pivots are 1-based: ``pivots[i]`` holds the row swapped with
    # row ``i`` during step ``i``.  We start from the identity and apply
    # the swap sequence in reverse to recover ``P`` such that ``P · A``
    # equals the *unpermuted* LU product.
    P = _build_permutation_matrix(pivots, n, A.dtype, A.device)
    return P, L, U


def _build_permutation_matrix(
    pivots: Tensor,
    n: int,
    dtype: lucid.dtype | type[lucid.dtype] | _C_engine.Dtype | None,
    device: lucid.device | _C_engine.Device | str | None,
) -> Tensor:
    """Convert LAPACK's 1-based pivot vector to an explicit (n × n) P
    matrix such that ``A = P · L · U`` (LAPACK's contract is
    ``P · A = L · U``; we transpose at the end so callers can use the
    factor product directly)."""
    perm: list[int] = list(range(n))
    pv = pivots.numpy()
    # If batched, only the leading instance is exposed here — caller
    # should iterate.  Lucid's lu_factor on a non-batched 2-D input
    # gives a length-n pivot vector; that's what we handle.
    if pv.ndim != 1:
        raise NotImplementedError("lu: batched LU is not yet exposed")
    for i in range(n):
        j = int(pv[i]) - 1  # 1-based → 0-based
        perm[i], perm[j] = perm[j], perm[i]
    # Build the explicit matrix.  P[i, perm[i]] = 1.
    P_np = [[0.0] * n for _ in range(n)]
    for i in range(n):
        P_np[i][perm[i]] = 1.0
    return lucid.tensor(P_np, dtype=dtype, device=device).mT  # transpose: A = P·L·U


# ── ldl_solve — back-substitution using the LDL factorization ──────────────


def ldl_solve(LD: Tensor, pivots: Tensor, B: Tensor) -> Tensor:
    r"""Solve a symmetric linear system using an LDL factorization.

    Given :math:`A = L\,D\,L^\top` produced by :func:`ldl_factor`,
    solves

    .. math::

        A\,X \,=\, B

    by chaining three substitutions:

    .. math::

        L\,Y = P B, \quad D\,Z = Y, \quad L^\top X = Z,

    followed by an inverse permutation to undo the Bunch-Kaufman row
    swaps.

    Parameters
    ----------
    LD : Tensor
        Packed LDL factor from :func:`ldl_factor`, shape ``(n, n)``.
    pivots : Tensor
        Pivot indices from :func:`ldl_factor`.  This implementation
        only supports **1×1 (simple) pivots** — every entry must be
        strictly positive.  Mixed 2×2 block pivots raise
        ``NotImplementedError``.
    B : Tensor
        Right-hand side of shape ``(n, k)`` (or ``(n,)``).

    Returns
    -------
    Tensor
        Solution :math:`X`, same shape as ``B``.

    Notes
    -----
    Supports indefinite symmetric :math:`A` (unlike Cholesky), so it
    is appropriate for KKT / saddle-point systems where Cholesky
    would fail.  Cost per solve is :math:`O(n^2 k)` once the LDL
    factor is in hand.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import ldl_factor, ldl_solve
    >>> A = lucid.tensor([[4.0, 1.0], [1.0, 3.0]])
    >>> LD, piv = ldl_factor(A)
    >>> b = lucid.tensor([[5.0], [4.0]])
    >>> ldl_solve(LD, piv, b)  # doctest: +SKIP
    """
    pv = pivots.numpy()
    if pv.ndim != 1:
        raise NotImplementedError("ldl_solve: batched solve not yet exposed")
    if int(pv.min()) <= 0:
        raise NotImplementedError(
            "ldl_solve: 2x2 block pivots from Bunch-Kaufman are not yet "
            "supported.  All pivot entries must be > 0 (1x1 simple pivots)."
        )
    n = int(LD.shape[-1])
    eye_n = lucid.eye(n, dtype=LD.dtype, device=LD.device)
    # L is the strictly lower triangle of LD with 1s on the diagonal;
    # D's diagonal lives in LD's diagonal.
    L_strict = lucid.tril(LD) - lucid.tril(LD) * eye_n
    L = L_strict + eye_n
    diag = lucid.diagonal(LD)  # length-n vector

    # Apply LAPACK's pivot permutation to B before the triangular solves.
    perm: list[int] = list(range(n))
    pv_l = pv.tolist()
    for i in range(n):
        j = int(pv_l[i]) - 1
        perm[i], perm[j] = perm[j], perm[i]
    B_perm = B.index_select(-2, lucid.tensor(perm, dtype=lucid.int64, device=B.device))

    y = solve_triangular(L, B_perm, upper=False, unitriangular=True)
    # Diagonal solve via element-wise division along the leading dim of y.
    diag_col = diag.reshape(n, 1)
    z = y / diag_col
    X_perm = solve_triangular(L.mT, z, upper=True, unitriangular=True)

    # Inverse permutation to restore the original row order.
    inv_perm: list[int] = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return X_perm.index_select(
        -2, lucid.tensor(inv_perm, dtype=lucid.int64, device=B.device)
    )


# ── linalg.diagonal — batched-view alias of lucid.diagonal ─────────────────


def matrix_exp(A: Tensor) -> Tensor:
    r"""Matrix exponential :math:`e^A`.

    Returns the matrix exponential

    .. math::

        e^A \,=\, \sum_{k=0}^{\infty} \frac{A^k}{k!},

    which solves the matrix ODE :math:`\dot Y = A Y` with initial
    condition :math:`Y(0) = I`.  Computed by the scaling-and-squaring
    algorithm of Higham (2005) with a Padé [6/6] rational approximant:

    1. Scale :math:`A' = A / 2^s` so that :math:`\|A'\| \le \theta_6`.
    2. Approximate :math:`R \approx e^{A'}` via Padé [6/6] using the
       even/odd polynomial splitting :math:`R = D^{-1} N`.
    3. Square :math:`R` a total of :math:`s` times to recover
       :math:`e^A = R^{2^s}`.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    Tensor
        :math:`e^A`, shape ``(*, n, n)``.

    Notes
    -----
    Computational cost is :math:`O((s + 6)\, n^3)` where the scaling
    parameter :math:`s = \lceil \log_2 (\|A\| / \theta_6) \rceil`.  The
    Frobenius norm is used here as a (conservative) proxy for the
    1-norm — this may add one extra squaring step but keeps the result
    correct.

    Differentiable: built entirely on :func:`matmul` and :func:`inv`,
    so autograd flows naturally.  Batched inputs (``ndim > 2``) are
    handled element-wise.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_exp
    >>> A = lucid.tensor([[0.0, 1.0], [-1.0, 0.0]])  # 90-deg rotation generator
    >>> matrix_exp(A)
    Tensor([[ 0.5403,  0.8415],
            [-0.8415,  0.5403]])
    """
    import math as _math

    sh: tuple[int, ...] = tuple(A.shape)
    if len(sh) < 2 or sh[-1] != sh[-2]:
        raise ValueError(
            f"matrix_exp requires a square matrix in the last two dims, got shape {sh}"
        )
    n: int = int(sh[-1])

    # -- Padé [6/6] coefficients c_j = (12-j)! * 6! / (12! * j! * (6-j)!) ----
    _c0: float = 1.0
    _c1: float = 0.5
    _c2: float = 5.0 / 44.0
    _c3: float = 1.0 / 66.0
    _c4: float = 1.0 / 792.0
    _c5: float = 1.0 / 15840.0
    _c6: float = 1.0 / 665280.0
    # theta_6: Frobenius-norm threshold at which Padé [6/6] is machine-accurate
    _theta: float = 2.0

    # -- Scaling ---------------------------------------------------------------
    # Upper bound for the 1-norm: max element × n (safe over-estimate).
    norm_bound: float = float(A.abs().max().item()) * n
    s: int = (
        max(0, _math.ceil(_math.log2(norm_bound / _theta)))
        if norm_bound > _theta
        else 0
    )
    A_sc: Tensor = A * (2.0 ** (-s))

    # -- Matrix powers ---------------------------------------------------------
    A2: Tensor = A_sc @ A_sc
    A4: Tensor = A2 @ A2
    A6: Tensor = A2 @ A4

    # -- Identity (broadcasts over any batch dims) -----------------------------
    eye_2d: Tensor = lucid.eye(n, dtype=A.dtype, device=A.device)
    if len(sh) > 2:
        # Use arithmetic broadcast: eye_2d + zeros of batch shape keeps autograd.
        I: Tensor = eye_2d + lucid.zeros(*sh[:-2], n, n, dtype=A.dtype, device=A.device)
    else:
        I = eye_2d

    # -- Padé polynomials (even/odd split for efficiency) ----------------------
    # Even = c0*I + c2*A2 + c4*A4 + c6*A6
    Even: Tensor = _c0 * I + _c2 * A2 + _c4 * A4 + _c6 * A6
    # Odd  = A * (c1*I + c3*A2 + c5*A4)
    Odd: Tensor = A_sc @ (_c1 * I + _c3 * A2 + _c5 * A4)

    # N = Even + Odd,  D = Even - Odd
    N: Tensor = Even + Odd
    D: Tensor = Even - Odd

    # -- Solve D @ R = N  →  R = D^{-1} @ N -----------------------------------
    R: Tensor = inv(D) @ N

    # -- Squaring step ---------------------------------------------------------
    for _ in range(s):
        R = R @ R

    return R


def diagonal(
    A: Tensor,
    *,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Tensor:
    r"""Extract a (off-)diagonal from each matrix in a batched input.

    Returns the elements of :math:`A` lying on the diagonal selected by
    ``offset`` from the matrix slice formed by ``(dim1, dim2)``.  For
    a 2-D matrix and ``offset=0`` this is

    .. math::

        d_i \,=\, A_{i,\,i}.

    Positive ``offset`` selects super-diagonals (:math:`A_{i, i +
    \text{offset}}`); negative offsets select sub-diagonals.

    Parameters
    ----------
    A : Tensor
        Input of shape ``(*, m, n)`` (or higher rank).
    offset : int, keyword-only, optional
        Diagonal index relative to the main diagonal.  Default ``0``.
    dim1 : int, keyword-only, optional
        First matrix dimension.  Default ``-2``.
    dim2 : int, keyword-only, optional
        Second matrix dimension.  Default ``-1``.

    Returns
    -------
    Tensor
        Diagonal values with the two matrix axes replaced by a single
        axis of length :math:`\min(m, n) - |\text{offset}|`.

    Notes
    -----
    Shares the engine kernel with the top-level :func:`lucid.diagonal`;
    the only difference is the keyword-only / matrix-aware defaults
    (``dim1=-2``, ``dim2=-1``) that match the standard linalg
    convention.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import diagonal
    >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> diagonal(A)
    Tensor([1.0000, 4.0000])
    >>> diagonal(A, offset=1)
    Tensor([2.0000])
    """
    return lucid.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)  # type: ignore[arg-type]


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
    "ldl_solve",
    "lu",
    "cholesky_ex",
    "inv_ex",
    "solve_ex",
    "matrix_exp",
    "diagonal",
]
