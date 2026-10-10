// lucid/_C/backend/LapackInfo.h
//
// The one reading of a factorisation's status, for both streams.
//
// LAPACK reports through ``info``: ``0`` succeeded, ``-k`` means argument
// ``k`` was illegal (always Lucid's bug — it marshalled the call wrongly),
// and ``k > 0`` means the matrix itself could not be factored at step ``k``
// (a zero pivot, a leading minor that is not positive definite, an
// iteration that did not converge).  The last is ``LinAlgError``; the first
// stays a plain ``LucidError``.
//
// The CPU stream gets ``info`` from LAPACK.  MLX reports nothing, so the
// Metal stream reads the failed step off the factor (a zero pivot of
// ``U``, a non-positive diagonal entry of ``L``) and comes here with it —
// the two streams raise the same class with the same text, naming the same
// matrix of a batch.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "../core/ErrorBuilder.h"

namespace lucid::backend {

// Raise for a non-zero LAPACK ``info`` of matrix ``index`` among ``batch``.
// The matrix is named only when there is more than one.  Call it inside the
// batch loop: ``info`` is overwritten per matrix, so checked once after the
// loop it reports only the last matrix, and an earlier failure leaves its
// output block unwritten and returned as if it were an answer.
inline void
check_lapack_info(int info, const char* op, std::int64_t index = 0, std::int64_t batch = 1) {
    if (info == 0)
        return;
    const std::string where =
        batch > 1 ? " in matrix " + std::to_string(index) + " of the batch" : std::string{};
    if (info < 0)
        ErrorBuilder(op).fail("LAPACK invalid argument index" + std::to_string(-info) + where);
    ErrorBuilder(op).linalg_error("LAPACK numerical failure (info=" + std::to_string(info) + ")" +
                                  where);
}

// Raise for the first set flag of ``flags``, a row-major ``(batch, n)``
// grid with one flag per factorisation step, as LAPACK would have: the
// first failed matrix, and ``info`` the 1-based step it failed at.  Returns
// when no flag is set.
inline void check_step_flags(const bool* flags, std::size_t count, std::size_t n, const char* op) {
    if (n == 0)
        return;
    std::size_t first = 0;
    while (first < count && !flags[first])
        ++first;
    if (first == count)
        return;
    check_lapack_info(static_cast<int>(first % n) + 1, op, static_cast<std::int64_t>(first / n),
                      static_cast<std::int64_t>(count / n));
}

}  // namespace lucid::backend
