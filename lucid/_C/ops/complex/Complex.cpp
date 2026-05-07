// lucid/_C/ops/complex/Complex.cpp
//
// Forward implementation of complex_op.  Validates that both inputs are
// real-floating dtype, the shapes match, and the devices match — then
// dispatches through ``IBackend::complex_combine``.

#include "Complex.h"

#include "../../backend/Dispatcher.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr complex_op(const TensorImplPtr& re, const TensorImplPtr& im) {
    Validator::input(re, "complex.re").non_null();
    Validator::input(im, "complex.im").non_null();
    complex_detail::require_real_float(re->dtype(), "complex.re");
    complex_detail::require_real_float(im->dtype(), "complex.im");
    if (re->shape() != im->shape())
        ErrorBuilder("complex").shape_mismatch(
            re->shape(), im->shape(),
            "real and imag must have the same shape");
    if (re->device() != im->device())
        ErrorBuilder("complex").device_mismatch(
            re->device(), im->device(),
            "real and imag must live on the same device");

    OpScopeFull scope{"complex", re->device(), re->dtype(), re->shape()};

    Storage out = backend::Dispatcher::for_device(re->device())
                      .complex_combine(re->storage(), im->storage(), re->shape());
    return complex_detail::fresh(std::move(out), re->shape(), Dtype::C64, re->device());
}

}  // namespace lucid
