#include "MatrixPower.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr matrix_power_op(const TensorImplPtr& a, int p) {
    using namespace linalg_detail;
    Validator::input(a, "matrix_power.a").non_null();
    require_float(a->dtype(), "matrix_power");
    require_square_2d(a->shape(), "matrix_power");
    OpScopeFull scope{"matrix_power", a->device(), a->dtype(), a->shape()};

    Storage out = backend::Dispatcher::for_device(a->device())
                      .linalg_matrix_power(a->storage(), a->shape(), p, a->dtype());
    return fresh(std::move(out), a->shape(), a->dtype(), a->device());
}

}  // namespace lucid
