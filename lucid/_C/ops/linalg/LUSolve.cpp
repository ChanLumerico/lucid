// lucid/_C/ops/linalg/LUSolve.cpp
#include "LUSolve.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
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

    auto result = backend::Dispatcher::for_device(LU->device())
                      .linalg_lu_solve(LU->storage(), pivots->storage(), b->storage(), LU->shape(),
                                       b->shape(), LU->dtype());

    return fresh(std::move(result), b->shape(), LU->dtype(), LU->device());
}

}  // namespace lucid
