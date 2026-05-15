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
    Validator::input(pivots, "lu_solve.pivots").non_null();
    Validator::input(b, "lu_solve.b").float_only().non_null();

    auto result = backend::Dispatcher::for_device(LU->device())
                      .linalg_lu_solve(LU->storage(), pivots->storage(), b->storage(), LU->shape(),
                                       b->shape(), LU->dtype());

    return fresh(std::move(result), b->shape(), LU->dtype(), LU->device());
}

}  // namespace lucid
