// lucid/_C/ops/linalg/LUSolve.cpp
#include "LUSolve.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"
namespace lucid {

TensorImplPtr
lu_solve_op(const TensorImplPtr& LU, const TensorImplPtr& pivots, const TensorImplPtr& b) {
    using namespace linalg_detail;
    Validator::input(LU, "lu_solve.LU").float_only().non_null();
    // LAPACK's ``ipiv`` is ``const int*``, so the buffer is read 32 bits at
    // a time whatever dtype it arrived as.  Only checking for null let
    // three separate failures through: an int8 or bool pivot vector was
    // read past its own allocation and took the process down with SIGBUS,
    // an int16 one produced 1e+133 instead of a solution, and an int64 one
    // returned a different — silently wrong — answer.  Only I32 is the
    // width this reads, so only I32 may be passed.
    Validator::input(pivots, "lu_solve.pivots").non_null().dtype_eq(Dtype::I32);
    Validator::input(b, "lu_solve.b").float_only().non_null();

    // The factor has to be square.  ``?getrs`` solves ``A X = B`` from an
    // LU of A, and only a square A has a solve; ``lu_factor`` accepts any
    // shape, so a rectangular factor can now reach this call.  It used to
    // be handed to LAPACK anyway, which read ``n`` rows out of a matrix
    // that had fewer and answered ``[nan, nan, -inf]``.
    const auto& lu_sh = LU->shape();
    if (lu_sh.size() < 2)
        ErrorBuilder("lu_solve.LU").fail("LU must be at least 2-D");
    if (lu_sh[lu_sh.size() - 1] != lu_sh[lu_sh.size() - 2])
        ErrorBuilder("lu_solve.LU")
            .fail("LU must be square to solve with — lu_factor accepts a "
                  "rectangular matrix, but only a square system has a solution");

    // A degenerate matrix has an answer; LAPACK just will not be the one
    // to compute it — see ``empty_matrix``.
    if (empty_matrix(LU->shape()))
        return zeros_op(b->shape(), LU->dtype(), LU->device());

    auto result = backend::Dispatcher::for_device(LU->device())
                      .linalg_lu_solve(LU->storage(), pivots->storage(), b->storage(), LU->shape(),
                                       b->shape(), LU->dtype());

    return fresh(std::move(result), b->shape(), LU->dtype(), LU->device());
}

}  // namespace lucid
