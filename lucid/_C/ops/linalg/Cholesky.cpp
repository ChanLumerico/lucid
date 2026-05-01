#include "Cholesky.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr cholesky_op(const TensorImplPtr& a, bool upper) {
    Validator::input(a, "cholesky.a").float_only().square_2d();
    OpScopeFull scope{"cholesky", a->device(), a->dtype(), a->shape()};

    Storage out = backend::Dispatcher::for_device(a->device())
                      .linalg_cholesky(a->storage(), a->shape(), upper, a->dtype());
    return linalg_detail::fresh(std::move(out), a->shape(), a->dtype(), a->device());
}

}  // namespace lucid
