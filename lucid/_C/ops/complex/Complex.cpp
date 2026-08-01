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
#include "../../kernel/BinaryKernel.h"  // detail::broadcast_shapes
#include "../utils/Layout.h"            // broadcast_to_op
#include "_Detail.h"

namespace lucid {

TensorImplPtr complex_op(const TensorImplPtr& re_in, const TensorImplPtr& im_in) {
    Validator::input(re_in, "complex.re").non_null();
    Validator::input(im_in, "complex.im").non_null();
    complex_detail::require_real_float(re_in->dtype(), "complex.re");
    complex_detail::require_real_float(im_in->dtype(), "complex.im");

    // The two parts broadcast against each other, so building a complex
    // tensor from a real field and one scalar imaginary part is spelled
    // the obvious way.  Materialised only when the shapes differ.
    TensorImplPtr re = re_in;
    TensorImplPtr im = im_in;
    if (re_in->shape() != im_in->shape()) {
        const Shape out = detail::broadcast_shapes(re_in->shape(), im_in->shape());
        if (re_in->shape() != out)
            re = broadcast_to_op(re_in, out);
        if (im_in->shape() != out)
            im = broadcast_to_op(im_in, out);
    }

    if (re->device() != im->device())
        ErrorBuilder("complex").device_mismatch(re->device(), im->device(),
                                                "real and imag must live on the same device");

    OpScopeFull scope{"complex", re->device(), re->dtype(), re->shape()};

    Storage out = backend::Dispatcher::for_device(re->device())
                      .complex_combine(re->storage(), im->storage(), re->shape());
    return complex_detail::fresh(std::move(out), re->shape(), Dtype::C64, re->device());
}

}  // namespace lucid
