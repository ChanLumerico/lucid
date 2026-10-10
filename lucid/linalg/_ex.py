"""The ``*_ex`` variants: LAPACK's ``info`` per matrix instead of an error.

``cholesky_ex``, ``inv_ex``, ``solve_ex``, ``lu_factor_ex`` and
``ldl_factor_ex`` report a matrix the routine could not factor through an
``int32`` ``info`` code — LAPACK's own number for it — and raise
:class:`LinAlgError` only under ``check_errors=True``.  ``lu_factor`` and
``ldl_factor`` are the checked forms of the last two.  Re-exported by
:mod:`lucid.linalg`, which owns the factorisations they wrap.
"""

from collections.abc import Callable

import lucid
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import Tensor
from lucid._unsupported import unsupported_if
from lucid.linalg import (
    LinAlgError,
    _la,
    _lu_factor_packed,
    cholesky,
    inv,
    solve,
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
    with lucid.no_grad():
        return _lu_factor_ex(A.detach(), check_errors=False, op="getrf")[2]


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
    * ``info == k > 0`` — the leading minor of order :math:`k` is not
      positive-definite; that matrix's factor is zero-filled.

    Parameters
    ----------
    A : Tensor
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
        Cholesky factor of shape ``(*, n, n)`` — zeros for the matrices
        that failed.
    info : Tensor
        ``int32`` status, scalar or shape ``(*,)`` matching the batch
        of ``A``.

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
    try:
        return cholesky(A, upper=upper), _info_zero(A)
    except LinAlgError:
        pass  # located per matrix below
    info = _potrf_info(A, upper)
    if check_errors:
        _raise_ex_failure("cholesky_ex", info, _not_positive_definite)
    return _on_the_rest(lambda M: cholesky(M, upper=upper), A, info), info


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
        Currently must be ``True``.  ``False`` (``X A = B``) is not yet
        implemented and raises ``NotImplementedError``.
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
    if not left:
        raise NotImplementedError("solve_ex: only left=True is supported")
    try:
        return solve(A, B), _info_zero(A)
    except LinAlgError:
        pass  # located per matrix below
    info = _getrf_info(A)
    if check_errors:
        _raise_ex_failure("solve_ex", info, _singular_reason("solve"))
    return _on_the_rest(lambda M: solve(M, B), A, info), info


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


def ldl_factor_ex(
    A: Tensor,
    *,
    hermitian: bool = True,
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
    A : Tensor
        Symmetric / Hermitian matrix of shape ``(*, n, n)``.
    hermitian : bool, keyword-only, optional
        As :func:`ldl_factor`.  Default ``True``.
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

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import ldl_factor_ex
    >>> LD, piv, info = ldl_factor_ex(lucid.zeros(2, 2))
    >>> int(info)
    1
    """
    return _ldl_factor_ex(
        A, hermitian=hermitian, check_errors=check_errors, op="ldl_factor_ex"
    )


def _ldl_factor_ex(
    A: Tensor, *, hermitian: bool, check_errors: bool, op: str
) -> tuple[Tensor, Tensor, Tensor]:
    """``(LD, pivots, info)``, ``info`` being ``sytrf``'s own code from the
    kernel; ``check_errors`` raises for the first matrix it marks."""
    unsupported_if(
        not hermitian,
        op,
        "hermitian",
        hermitian,
        detail="Only the Hermitian case is computed.",
    )
    ld_impl, piv_impl, info_impl = _la.ldl_factor(_unwrap(A))
    LD, pivots, info = lucid.tril(_wrap(ld_impl)), _wrap(piv_impl), _wrap(info_impl)
    if check_errors:
        _raise_ex_failure(op, info, _zero_block_reason)
    return LD, pivots, info
