"""The ``*_ex`` variants: LAPACK's ``info`` per matrix instead of an error.

``cholesky_ex``, ``inv_ex``, ``solve_ex``, ``lu_factor_ex`` and
``ldl_factor_ex`` report a matrix the routine could not factor through an
``int32`` ``info`` code — LAPACK's own number for it — and raise
:class:`LinAlgError` only under ``check_errors=True``.  ``lu_factor`` and
``ldl_factor`` are the checked forms of the last two; ``ldl_factor`` lives
here beside its variant.  Re-exported by :mod:`lucid.linalg`, which owns
the factorisations they wrap.
"""

from collections.abc import Callable
from typing import final, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._dtype import finfo as _finfo
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.linalg import (
    LinAlgError,
    _la,
    _lu_factor_packed,
    _Pinv,
    cholesky,
    eigh,
    inv,
    solve,
    solve_triangular,
    svd,
)

# ── *_ex variants — return (result, info) instead of raising ───────────────
#
# LAPACK's ``*_ex`` contract is one ``info`` integer per matrix of the
# batch: ``0`` on success, otherwise the routine's own code — for ``getrf``
# the 1-based index of the first exactly-zero pivot of ``U``, for ``potrf``
# the order of the leading minor that is not positive definite.  A matrix
# that succeeded keeps its answer whatever its neighbours did.
#
# ``lu_factor`` and ``ldl_factor`` finish on a singular matrix, and their
# kernels return LAPACK's ``info`` with the factor.  ``cholesky``, ``inv``
# and ``solve`` have no answer to return: their kernels raise
# ``LinAlgError`` at the first failed matrix of a batch.  So those wrappers
# locate each failure per matrix, swap the failed matrices for the
# identity so the kernel runs on the batch, and replace only their results
# (by zeros — LAPACK leaves them undefined).  The successes, and their
# gradients, are the kernel's.


def _info_zero(A: Tensor) -> Tensor:
    """Build a scalar (or batched) int32 ``info`` tensor of zeros aligned
    with the leading-batch dims of ``A`` (everything except the trailing
    two matrix dims).  Mirrors LAPACK's batched-info contract."""
    batch = list(A.shape[:-2])
    if not batch:
        return lucid.zeros(tuple(), dtype=lucid.int32, device=A.device)
    return lucid.zeros(*batch, dtype=lucid.int32, device=A.device)


def _getrf_info(A: Tensor) -> Tensor:
    """LAPACK ``getrf``'s ``info`` for each matrix of ``A`` — the pivot
    ``getri`` / ``getrs`` would divide by.  Factorising finishes on a
    singular matrix, so this answers for every matrix of the batch at once;
    the ``inv`` / ``solve`` kernels stop at the first failure."""
    return _wrap(_la.lu_factor(_unwrap(A))[2])


def _cholesky_succeeds(M: Tensor, upper: bool) -> bool:
    """Whether the Cholesky kernel factorises the single matrix ``M``."""
    try:
        _la.cholesky(_unwrap(M.contiguous()), upper)
    except LinAlgError:
        return False
    return True


def _potrf_info(A: Tensor, upper: bool) -> Tensor:
    """LAPACK ``potrf``'s ``info`` for each matrix of ``A``.

    The order of the first leading minor that is not positive definite, or
    ``0``.  The kernel stops at the first failed matrix of a batch, so the
    failures are found by bisection twice over.
    Across the batch: a slice that factorises clears every matrix in it at
    once, so ``f`` failures among ``B`` matrices cost about ``2 f log2 B``
    calls rather than ``B``.  Within a failed matrix, over its leading
    blocks: the first ``j`` steps of a Cholesky factorisation read nothing
    outside the leading ``j × j`` block, so that block fails exactly when
    the whole matrix fails within ``j`` steps.  Reached only once the
    batched call has failed, so the whole batch is known to.
    """
    n = int(A.shape[-1])
    batch = tuple(int(s) for s in A.shape[:-2])
    with lucid.no_grad():
        flat = A.detach().reshape(-1, n, n)
        codes = [0] * int(flat.shape[0])
        # (start, stop, known to fail) slices of the batch still to settle.
        pending = [(0, len(codes), True)]
        while pending:
            start, stop, failed = pending.pop()
            if not failed and _cholesky_succeeds(flat[start:stop], upper):
                continue
            if stop - start > 1:
                middle = (start + stop) // 2
                pending += [(start, middle, False), (middle, stop, False)]
                continue
            M = flat[start]
            lo, hi = 1, n  # order n fails; find the smallest order that does
            while lo < hi:
                mid = (lo + hi) // 2
                if _cholesky_succeeds(M[:mid, :mid], upper):
                    lo = mid + 1
                else:
                    hi = mid
            codes[start] = lo
    if not batch:
        return lucid.tensor(codes[0], dtype=lucid.int32, device=A.device)
    return lucid.tensor(codes, dtype=lucid.int32, device=A.device).reshape(*batch)


def _raise_ex_failure(op: str, info: Tensor, reason: Callable[[int], str]) -> None:
    """Raise for the first failed matrix, naming it as the reference does."""
    flat = info.reshape(-1)
    failed = lucid.nonzero(flat).reshape(-1)
    if failed.numel() == 0:
        return
    b = int(failed[0].item())
    where = f"(Batch element {b}): " if info.ndim else ""
    raise LinAlgError(f"{op}: {where}{reason(int(flat[b].item()))}")


def _singular_reason(action: str) -> Callable[[int], str]:
    """The ``getrf`` failure message for an ``inv`` / ``solve`` variant."""

    def reason(code: int) -> str:
        """The message for a zero pivot at 1-based position ``code``."""
        return (
            f"The diagonal element {code} is zero, the {action} could not be "
            f"completed because the input matrix is singular."
        )

    return reason


def _zero_pivot_reason(code: int) -> str:
    """The ``getrf`` failure message for an exactly-zero pivot ``U[k, k]``."""
    return (
        f"U[{code},{code}] is zero and using it on lu_solve would result in a "
        "division by zero.  To factor the matrix anyway, call "
        "lucid.linalg.lu(A) or lucid.linalg.lu_factor_ex(A)."
    )


def _zero_block_reason(code: int) -> str:
    """The ``sytrf`` failure message for an exactly-zero block ``D[k, k]``."""
    return (
        f"D[{code},{code}] is zero: the factorization completed, but D is "
        "singular and using it on ldl_solve would result in a division by "
        "zero.  To factor the matrix anyway, call lucid.linalg.ldl_factor_ex(A)."
    )


def _not_positive_definite(code: int) -> str:
    """The ``potrf`` failure message for a leading minor of order ``code``."""
    return (
        "The factorization could not be completed because the input is not "
        f"positive-definite (the leading minor of order {code} is not "
        "positive-definite)."
    )


def _on_the_rest(fn: Callable[[Tensor], Tensor], A: Tensor, info: Tensor) -> Tensor:
    """``fn(A)`` with the failed matrices of ``A`` left out.

    Each matrix whose ``info`` is non-zero is swapped for the identity, so
    the kernel can run on the batch, and its result is zeroed afterwards.
    Both swaps are ``where`` selections, so the matrices that succeeded keep
    their gradient and the failed ones receive none.
    """
    failed = info != 0
    batch = tuple(failed.shape)
    eye = lucid.eye(int(A.shape[-1]), dtype=A.dtype, device=A.device)
    out = fn(lucid.where(failed.reshape(*batch, 1, 1), eye, A))
    # A solve against a vector right-hand side drops one trailing axis.
    trailing = (1, 1) if out.ndim >= A.ndim else (1,)
    return lucid.where(failed.reshape(*batch, *trailing), lucid.zeros_like(out), out)


def _partial_cholesky(M: Tensor, order: int, upper: bool) -> Tensor:
    """The defined part of the factor of ``M``, whose minor ``order`` failed.

    LAPACK defines only the leading ``(order - 1)``-square block: the factor
    of the minor that succeeded.  What ``potrf`` leaves outside it depends
    on its blocking — partly factored, partly the input's own — so it is
    not reproduced; it is zero here.  Real input only, as :func:`cholesky`.
    """
    done = order - 1
    out = lucid.zeros_like(M)
    if done:
        out[:done, :done] = cholesky(M[:done, :done], upper=upper)
    return out


def _with_partial_factors(out: Tensor, A: Tensor, info: Tensor, upper: bool) -> Tensor:
    """``out`` with each failed matrix's leading factor put back.

    Taken without a gradient: the failed matrices received none before, and
    a partial factor is not a function the caller asked to differentiate.
    """
    n = int(A.shape[-1])
    with lucid.no_grad():
        flat = A.detach().reshape(-1, n, n)
        codes = [int(c.item()) for c in info.reshape(-1)]
        partials = lucid.stack(
            [
                _partial_cholesky(flat[b], code, upper) if code else flat[b]
                for b, code in enumerate(codes)
            ]
        ).reshape(*out.shape)
    failed = (info != 0).reshape(*info.shape, 1, 1)
    return lucid.where(failed, partials, out)


def cholesky_ex(
    input: Tensor,
    *,
    upper: bool = False,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Cholesky factorization with an explicit success flag.

    Variant of :func:`cholesky` that, instead of raising when the
    input fails to be positive-definite, returns the factor together
    with an integer ``info`` code following LAPACK's convention:

    * ``info == 0`` — success; :math:`L` (or :math:`U`) is meaningful.
    * ``info == k > 0`` — the leading minor of order :math:`k` is not
      positive-definite; that matrix's factor holds only the part LAPACK
      defines, below.

    Parameters
    ----------
    input : Tensor
        Candidate SPD matrix of shape ``(*, n, n)``.
    upper : bool, keyword-only, optional
        If ``True`` return the upper-triangular factor :math:`U` such
        that :math:`A = U^\top U`.  Default ``False``.
    check_errors : bool, keyword-only, optional
        If ``True``, raise for the first matrix that fails instead of
        reporting it through ``info`` — useful while debugging.

    Returns
    -------
    L : Tensor
        Cholesky factor of shape ``(*, n, n)``.  For a matrix that failed
        at order :math:`k`, the leading :math:`(k-1) \times (k-1)` block is
        the factor of the minor that succeeded, the one part LAPACK
        defines; the rest is zero (LAPACK leaves it to its blocking).
    info : Tensor
        ``int32`` status, scalar or shape ``(*,)`` matching the batch
        of ``input``.

    Notes
    -----
    Designed for code paths where a failed Cholesky is an expected
    event (e.g., trial steps in trust-region optimisers).  Callers
    must inspect ``info`` before trusting ``L``.

    Each matrix of a batch is judged on its own: one failure leaves the
    other factors, and their gradients, intact.  The batch is factorised
    once; only when that fails is each matrix tried alone, and a failed
    one bisected over its leading blocks to find the order LAPACK reports.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cholesky_ex
    >>> A = lucid.tensor([[4.0, 2.0], [2.0, 3.0]])
    >>> L, info = cholesky_ex(A)
    >>> int(info)
    0
    >>> _, info = cholesky_ex(lucid.tensor([[1.0, 0.0], [0.0, -1.0]]))
    >>> int(info)
    2
    """
    A = input
    try:
        return cholesky(A, upper=upper), _info_zero(A)
    except LinAlgError:
        pass  # located per matrix below
    info = _potrf_info(A, upper)
    if check_errors:
        _raise_ex_failure("cholesky_ex", info, _not_positive_definite)
    out = _on_the_rest(lambda M: cholesky(M, upper=upper), A, info)
    return _with_partial_factors(out, A, info, upper), info


def inv_ex(A: Tensor, *, check_errors: bool = False) -> tuple[Tensor, Tensor]:
    r"""Matrix inverse with an explicit success flag.

    Variant of :func:`inv` that returns an ``info`` code instead of
    raising on a singular input.

    * ``info == 0`` — success; ``Ainv`` is :math:`A^{-1}`.
    * ``info == k > 0`` — :math:`U_{kk}` of :math:`PA = LU` is exactly
      zero, so :math:`A` is singular; that matrix's ``Ainv`` is
      zero-filled.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(*, n, n)``.
    check_errors : bool, keyword-only, optional
        If ``True``, raise for the first singular matrix instead of
        reporting it through ``info``.

    Returns
    -------
    Ainv : Tensor
        Inverse of shape ``(*, n, n)`` — zeros for the singular matrices.
    info : Tensor
        ``int32`` status, scalar or shape ``(*,)`` matching the batch of
        ``A``.

    Notes
    -----
    Useful in algorithms that occasionally probe near-singular
    matrices (e.g., iterative refinement, regularisation grid
    searches) without wanting to wrap every call in a ``try``.

    Each matrix of a batch is judged on its own: a singular one leaves the
    other inverses, and their gradients, intact.  The batch is inverted
    directly; only when that fails is it factorised again to find which
    matrices failed and at which pivot.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import inv_ex
    >>> Ainv, info = inv_ex(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
    >>> int(info)
    0
    >>> _, info = inv_ex(lucid.tensor([[1.0, 2.0], [2.0, 4.0]]))
    >>> int(info)
    2
    """
    try:
        return inv(A), _info_zero(A)
    except LinAlgError:
        pass  # located per matrix below
    info = _getrf_info(A)
    if check_errors:
        _raise_ex_failure("inv_ex", info, _singular_reason("inversion"))
    return _on_the_rest(lambda M: inv(M), A, info), info


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
    * ``info == k > 0`` — :math:`U_{kk}` of :math:`PA = LU` is exactly
      zero, so :math:`A` is singular; that system's :math:`X` is
      zero-filled.

    Parameters
    ----------
    A : Tensor
        Coefficient matrix of shape ``(*, n, n)``.
    B : Tensor
        Right-hand side of shape ``(*, n, k)`` (or ``(*, n)``).
    left : bool, keyword-only, optional
        If ``True`` (default), solve :math:`A X = B`; ``False`` solves
        :math:`X A = B`.
    check_errors : bool, keyword-only, optional
        If ``True``, raise for the first singular matrix instead of
        reporting it through ``info``.

    Returns
    -------
    X : Tensor
        Solution shaped like :func:`solve`'s — zeros for the systems whose
        ``A`` was singular.
    info : Tensor
        ``int32`` status, scalar or shape ``(*,)`` matching the batch of
        ``A``.

    Notes
    -----
    This is the recommended form when calling from inside a batched /
    jit-compiled / vmapped routine where raising would break control
    flow; manually inspect ``info`` afterwards and decide whether to
    recover.

    Each system of a batch is judged on its own: a singular ``A`` leaves
    the other solutions, and their gradients, intact.  The batch is solved
    directly; only when that fails is ``A`` factorised again to find which
    systems failed and at which pivot.

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
    try:
        return solve(A, B, left=left), _info_zero(A)
    except LinAlgError:
        pass  # located per matrix below
    info = _getrf_info(A if left else A.mT)
    if check_errors:
        _raise_ex_failure("solve_ex", info, _singular_reason("solve"))
    return _on_the_rest(lambda M: solve(M, B, left=left), A, info), info


def lu_factor_ex(
    A: Tensor,
    *,
    pivot: bool = True,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""LU factorization with partial pivoting and an explicit status.

    Variant of :func:`lu_factor` that returns the factors of a singular
    matrix too, with an ``info`` code following LAPACK ``getrf``:

    * ``info == 0`` — every pivot :math:`U_{kk}` is non-zero.
    * ``info == k > 0`` — :math:`U_{kk}` is the first exactly-zero pivot
      (1-based).  The factorization is still complete, but :math:`A` is
      singular and :func:`lu_solve` divides by that pivot.

    Parameters
    ----------
    A : Tensor
        Matrix of shape ``(*, m, n)``.  Need not be square.
    pivot : bool, keyword-only, optional
        Must be ``True`` (the default); unpivoted LU is not implemented
        and raises ``NotImplementedError``.
    check_errors : bool, keyword-only, optional
        If ``True``, raise :class:`LinAlgError` for the first singular
        matrix instead of reporting it through ``info``.

    Returns
    -------
    LU : Tensor
        Packed factorization of shape ``(*, m, n)``, as :func:`lu_factor`.
    pivots : Tensor
        ``int32`` 1-based pivots of shape ``(*, min(m, n))``.
    info : Tensor
        ``int32`` status, scalar or shape ``(*,)`` matching the batch of
        ``A``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lu_factor_ex
    >>> LU, piv, info = lu_factor_ex(lucid.tensor([[1.0, 2.0], [2.0, 4.0]]))
    >>> int(info)
    2
    """
    if not pivot:
        raise NotImplementedError("lu_factor_ex: pivot=False is not supported")
    return _lu_factor_ex(A, check_errors=check_errors, op="lu_factor_ex")


def _lu_factor_ex(
    A: Tensor, *, check_errors: bool, op: str
) -> tuple[Tensor, Tensor, Tensor]:
    """``(LU, pivots, info)``, ``info`` being ``getrf``'s own code from the
    kernel; ``check_errors`` raises for the first matrix it marks."""
    LU, pivots, info = _lu_factor_packed(A)
    if check_errors:
        _raise_ex_failure(op, info, _zero_pivot_reason)
    return LU, pivots, info


def ldl_factor(
    input: Tensor,
    *,
    hermitian: bool = False,
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
    input : Tensor
        Symmetric / Hermitian matrix of shape ``(*, n, n)``.
    hermitian : bool, keyword-only, optional
        Whether a complex ``input`` is Hermitian rather than symmetric.  A real
        ``input`` is both, so either value factors it the same.  Default
        ``False``, as the reference's.

    Returns
    -------
    LD : Tensor
        Packed factor of shape ``(*, n, n)``.  The strict lower
        triangle holds :math:`L`; the diagonal holds :math:`D`'s
        entries (:math:`2 \times 2` blocks are stored in the
        sub-diagonal).  The strict upper triangle is zero.
    pivots : Tensor
        ``int32`` pivot indices, shape ``(*, n)``, 1-based (LAPACK
        convention).  A positive entry :math:`p` at :math:`k` marks a
        :math:`1 \times 1` block, with row :math:`k` interchanged with
        row :math:`p`; a pair of equal negative entries :math:`-p` at
        :math:`k, k+1` marks a :math:`2 \times 2` block, with row
        :math:`k+1` interchanged with row :math:`p`.

    Notes
    -----
    Backed by LAPACK ``sytrf``.  Cost is :math:`O(n^3 / 3)`.  Pair
    with :func:`ldl_solve` for solving symmetric indefinite systems.

    ``sytrf`` writes the factor into the lower triangle and leaves the
    upper one as it found it — the input's own entries.  Those are
    cleared here, so ``LD`` holds the factorization and nothing else.

    ``LD`` is differentiable in ``input``, to any order, wherever the pivots
    do not change — an addition over the reference framework, which
    defines no derivative for this factorization and raises on backward.
    The gradient lands on ``input``'s lower triangle, the part ``sytrf``
    reads, as :func:`cholesky`'s does.

    Raises
    ------
    LinAlgError
        If a :math:`1 \times 1` block of :math:`D` is exactly zero — the
        factorization completes, but :math:`D` is singular.
        :func:`ldl_factor_ex` returns it with a non-zero ``info`` instead.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import ldl_factor
    >>> A = lucid.tensor([[4.0, 1.0], [1.0, 3.0]])
    >>> LD, piv = ldl_factor(A)
    >>> LD
    tensor([[4., 0.], [0.25, 2.75]])
    >>> piv
    tensor([1, 2], dtype=lucid.int32)
    """
    del hermitian  # only real input is factored, where it changes nothing
    LD, pivots, _ = _ldl_factor_ex(input, check_errors=True, op="ldl_factor")
    return LD, pivots


def ldl_factor_ex(
    input: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""LDL factorization with an explicit status.

    Variant of :func:`ldl_factor` that returns the factors of a matrix
    whose :math:`D` is singular too, with an ``info`` code following
    LAPACK ``sytrf``:

    * ``info == 0`` — :math:`D` is non-singular.
    * ``info == k > 0`` — :math:`D_{kk}`, a :math:`1 \times 1` block, is
      the first one that is exactly zero (or NaN), 1-based.  The
      factorization is complete, but :func:`ldl_solve` would divide by
      that block.

    Parameters
    ----------
    input : Tensor
        Symmetric / Hermitian matrix of shape ``(*, n, n)``.
    hermitian : bool, keyword-only, optional
        As :func:`ldl_factor`.  Default ``False``.
    check_errors : bool, keyword-only, optional
        If ``True``, raise :class:`LinAlgError` for the first failed matrix
        instead of reporting it through ``info``.

    Returns
    -------
    LD, pivots : Tensor
        The packed factor ``(*, n, n)`` and ``int32`` pivots ``(*, n)``,
        as :func:`ldl_factor` returns them.
    info : Tensor
        One ``int32`` code per matrix — a scalar for a single matrix.

    Notes
    -----
    ``LD`` differentiates as :func:`ldl_factor`'s does — the reference
    framework does not differentiate it.  A matrix whose ``info`` is
    non-zero receives no gradient, as a failed matrix of the other
    ``*_ex`` variants does.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import ldl_factor_ex
    >>> LD, piv, info = ldl_factor_ex(lucid.zeros(2, 2))
    >>> int(info)
    1
    """
    del hermitian  # only real input is factored, where it changes nothing
    return _ldl_factor_ex(input, check_errors=check_errors, op="ldl_factor_ex")


def _ldl_factor_ex(
    A: Tensor, *, check_errors: bool, op: str
) -> tuple[Tensor, Tensor, Tensor]:
    """``(LD, pivots, info)``, ``info`` being ``sytrf``'s own code from the
    kernel; ``check_errors`` raises for the first matrix it marks.  The
    kernel takes real input only, for which symmetric and Hermitian are one
    and the same, so ``hermitian`` has nothing to choose here."""
    ld_impl, piv_impl, info_impl = _la.ldl_factor(_unwrap(A))
    LD, pivots, info = lucid.tril(_wrap(ld_impl)), _wrap(piv_impl), _wrap(info_impl)
    if check_errors:
        _raise_ex_failure(op, info, _zero_block_reason)
    if A.requires_grad and lucid.is_grad_enabled():
        tracked = _LDLFactorGrad.apply(A, LD, pivots, info)
        assert isinstance(tracked, Tensor)
        LD = tracked
    return LD, pivots, info


# ── The derivative of the packed LDLᵀ factor ───────────────────────────────
#
# ``sytrf`` stores ``A = P₁L₁ ⋯ PₘLₘ D Lₘᵀ ⋯``: step ``j`` swaps one pair of
# rows of the trailing matrix, then eliminates a 1×1 or 2×2 block, and its
# multipliers are stored in the row order of that moment — the later swaps
# are not applied to them.  Applying them gives one unit block-lower ``L̂``
# with ``Π A Πᵀ = L̂ D L̂ᵀ``.  Between two changes of pivot the pivots are
# constant and the factors smooth, so with ``X = L̂⁻¹ dÂ L̂⁻ᵀ``
#
#     dD = blockdiag(X),    dL̂ = L̂ · strictly_block_lower(X) · D⁻¹,
#
# whose adjoint is the backward below.  The reference framework defines no
# derivative here and raises on backward; a detached factor would instead
# train nothing without a word, so this one is given.


def _ldl_steps(piv: list[int]) -> tuple[list[int], list[list[int]], list[int]]:
    """Unpack one matrix's ``sytrf`` pivots.

    Returns each row's block (the block's first row), the row of ``LD``
    each entry of ``L̂`` is read from — ``L̂[i][c] = LD[order[i][c]][c]`` —
    and ``Π`` as the row of ``A`` that becomes each row of ``Π A Πᵀ``.
    """
    n = len(piv)
    first = list(range(n))
    order = [[i] * n for i in range(n)]
    perm = list(range(n))
    k = 0
    while k < n:
        size = 1 if piv[k] > 0 else 2
        last, p = k + size - 1, abs(piv[k]) - 1
        first[last] = k
        # This step's swap reaches the multipliers of the earlier columns.
        order[last][:k], order[p][:k] = order[p][:k], order[last][:k]
        perm[last], perm[p] = perm[p], perm[last]
        k += size
    return first, order, perm


@final
class _LDLFactorGrad(Function):
    """The packed factor of ``ldl_factor``, differentiable in ``A``.

    ``sytrf`` reads ``A``'s lower triangle, so that is where the gradient
    goes, as :func:`cholesky`'s does.  A matrix whose ``D`` is singular
    (``info != 0``) receives none, as a failed matrix of the other ``*_ex``
    variants does.  ``L̂`` and ``D`` are read from this Function's own
    output, so the result differentiates again.
    """

    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx, A: Tensor, LD: Tensor, pivots: Tensor, info: Tensor
    ) -> Tensor:
        out = LD.detach()
        ctx.save_for_backward(out, pivots, info)
        return out

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> tuple[Tensor, None, None, None]:
        LD, pivots, info = ctx.saved_tensors
        if LD.numel() == 0:
            return lucid.zeros_like(grad_out), None, None, None
        n, batch = int(LD.shape[-1]), tuple(LD.shape[:-2])
        steps = [
            _ldl_steps([int(p.item()) for p in row])
            for row in pivots.to("cpu").reshape(-1, n)
        ]
        dev = LD.device
        first = lucid.tensor([s[0] for s in steps], device=dev).reshape(*batch, n)
        order = lucid.tensor([s[1] for s in steps], dtype=lucid.int64, device=dev)
        perm = lucid.tensor([s[2] for s in steps], dtype=lucid.int64, device=dev)
        eye = lucid.eye(n, dtype=LD.dtype, device=dev)
        P = eye[perm.reshape(*batch, n)]
        # Below each block, and D's lower triangle.
        below = (first[..., :, None] > first[..., None, :]).to(LD.dtype)
        block = lucid.tril((first[..., :, None] == first[..., None, :]).to(LD.dtype))
        order = order.reshape(*batch, n, n)

        L = lucid.gather(LD * below, -2, order) + eye
        D_low = LD * block
        failed = (info != 0).reshape(*batch, 1, 1)
        D = lucid.where(failed, eye, D_low + lucid.tril(D_low, -1).mT)

        gL = lucid.gather(grad_out * below, -2, order)
        X = below * lucid.matmul(lucid.matmul(L.mT, gL), inv(D)) + grad_out * block
        Y = solve_triangular(L.mT, X, upper=True, unitriangular=True)
        Z = solve_triangular(L, Y, upper=False, left=False, unitriangular=True)
        G = lucid.matmul(lucid.matmul(P.mT, 0.5 * (Z + Z.mT)), P)
        grad = 2.0 * lucid.tril(G) - lucid.diag_embed(G.diagonal(dim1=-2, dim2=-1))
        return lucid.where(failed, lucid.zeros_like(grad), grad), None, None, None


def _complete_svd_basis(
    factor: Tensor, full_dim: int, k: int, *, columns: bool
) -> Tensor:
    """Complete a thin orthogonal factor ``factor`` with its null-space basis."""
    batch = tuple(int(s) for s in factor.shape[:-2])
    if 0 in batch:
        return lucid.zeros(
            *batch, full_dim, full_dim, dtype=factor.dtype, device=factor.device
        )
    if k == 0:
        eye = lucid.eye(full_dim, dtype=factor.dtype, device=factor.device)
        return lucid.broadcast_to(eye, (*batch, full_dim, full_dim)).contiguous()
    with lucid.no_grad():
        fd = factor.detach() if columns else factor.detach().mT
        bad = ~lucid.isfinite(fd).all(dim=(-2, -1), keepdim=True)
        safe = lucid.where(bad, lucid.zeros_like(fd), fd)
        eye = lucid.eye(full_dim, dtype=factor.dtype, device=factor.device)
        proj = eye - lucid.matmul(safe, safe.mT)
        _, v_impl = _la.eigh(_unwrap(proj))
        null = _wrap(v_impl)[..., :, k:]
        null = lucid.where(bad, lucid.full_like(null, float("nan")), null)
        if not columns:
            null = null.mT.contiguous()
    return lucid.cat([factor, null], dim=-1 if columns else -2)


def _complete_svd_factors(
    A: Tensor, U: object, S: object, Vh: object, full_matrices: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(U, S, Vh)`` with full orthogonal bases when ``full_matrices=True``."""
    assert isinstance(U, Tensor) and isinstance(S, Tensor) and isinstance(Vh, Tensor)
    if full_matrices:
        m, n = int(A.shape[-2]), int(A.shape[-1])
        k = min(m, n)
        if m > n:
            U = _complete_svd_basis(U, m, k, columns=True)
        elif m < n:
            Vh = _complete_svd_basis(Vh, n, k, columns=False)
    return U, S, Vh


def _tol_tensor(
    val: float | Tensor, A: Tensor, dtype: lucid.dtype, op: str = "matrix_rank"
) -> Tensor:
    """Coerce a scalar or tensor tolerance ``val`` for ``matrix_rank`` / ``pinv``."""
    if isinstance(val, Tensor):
        if val.is_complex():
            raise TypeError(
                f"{op}: tolerance tensor of complex dtype is not supported, got {val.dtype}"
            )
        return val.to(dtype).unsqueeze(-1)
    return lucid.full((1,), float(val), dtype=dtype, device=A.device)


def _sv_cutoff(
    A: Tensor,
    sv: Tensor,
    *,
    atol: float | Tensor | None,
    rtol: float | Tensor | None,
    op: str = "matrix_rank",
) -> Tensor:
    """Compute per-matrix singular-value cutoff ``max(atol, max_sv * rtol)``."""
    atol_t = _tol_tensor(0.0 if atol is None else atol, A, sv.dtype, op)
    eps = float(_finfo(sv.dtype).eps)
    default_rtol = max(int(A.shape[-2]), int(A.shape[-1])) * eps
    if rtol is not None:
        rtol_t = _tol_tensor(rtol, A, sv.dtype, op)
    elif atol is None:
        rtol_t = _tol_tensor(default_rtol, A, sv.dtype, op)
    else:
        zero = lucid.zeros_like(atol_t)
        rtol_t = lucid.where(
            atol_t > zero, zero, _tol_tensor(default_rtol, A, sv.dtype, op)
        )
    if sv.numel() == 0:
        return atol_t
    max_sv = _wrap(_C_engine.max(_unwrap(sv), [-1], True))
    return lucid.maximum(atol_t, max_sv * rtol_t)


def _pinv_with_tol(
    A: Tensor,
    *,
    atol: float | Tensor | None,
    rtol: float | Tensor | None,
    hermitian: bool,
) -> Tensor:
    """Compute ``pinv(A)`` under custom ``atol``/``rtol`` or ``hermitian=True``."""
    if A.ndim < 2 or (hermitian and A.shape[-1] != A.shape[-2]):
        raise ValueError(f"pinv: invalid input shape {tuple(A.shape)}")
    with lucid.no_grad():
        if hermitian:
            w, V = eigh(A)
            sv = w.abs()
            cutoff = _sv_cutoff(A, sv, atol=atol, rtol=rtol, op="pinv")
            inv_w = lucid.where(sv > cutoff, 1.0 / w, lucid.zeros_like(w))
            P = lucid.matmul(V * inv_w.unsqueeze(-2), V.mT)
        else:
            U, sv, Vh = svd(A, full_matrices=False)
            cutoff = _sv_cutoff(A, sv, atol=atol, rtol=rtol, op="pinv")
            inv_s = lucid.where(sv > cutoff, 1.0 / sv, lucid.zeros_like(sv))
            P = lucid.matmul(Vh.mT * inv_s.unsqueeze(-2), U.mT)
    if A.requires_grad and lucid.is_grad_enabled():
        out = _Pinv.apply(A, _unwrap(P))
        assert isinstance(out, Tensor)
        return out
    return P


