// lucid/_C/ops/complex/Real.cpp
//
// Forward implementation of real_op.  Validates the input is C64, then
// dispatches through ``IBackend::complex_real`` (CPU = stride-2 walk,
// GPU = ``mlx::core::real``).

#include "Real.h"

#include "../../backend/Dispatcher.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr real_op(const TensorImplPtr& a) {
    Validator::input(a, "real.a").non_null();
    complex_detail::require_complex(a->dtype(), "real");
    OpScopeFull scope{"real", a->device(), a->dtype(), a->shape()};

    Storage out =
        backend::Dispatcher::for_device(a->device()).complex_real(a->storage(), a->shape());
    return complex_detail::fresh(std::move(out), a->shape(), Dtype::F32, a->device());
}

}  // namespace lucid
