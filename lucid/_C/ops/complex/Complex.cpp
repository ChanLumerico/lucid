// lucid/_C/ops/complex/Complex.cpp
//
// Forward implementation of complex_op.  Validates that both inputs are
// real-floating dtype, the shapes match, and the devices match — then
// dispatches through ``IBackend::complex_combine``.

#include "Complex.h"

#include "../../backend/Dispatcher.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"  // detail::broadcast_shapes
#include "../../kernel/NaryKernel.h"
#include "../ufunc/Astype.h"  // astype_op
#include "../utils/Layout.h"  // broadcast_to_op
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

    // There is no half-precision complex, so the half formats widen here
    // rather than in the backend — which used to read their bytes as
    // float32 and assemble a number out of the wrong halves.
    if (is_half_float(re->dtype()))
        re = astype_op(re, Dtype::F32);
    if (is_half_float(im->dtype()))
        im = astype_op(im, Dtype::F32);
    if (re->dtype() != im->dtype())
        ErrorBuilder("complex").fail("real and imag must have the same dtype");

    Storage out = backend::Dispatcher::for_device(re->device())
                      .complex_combine(re->storage(), im->storage(), re->shape());
    auto result =
        complex_detail::fresh(std::move(out), re->shape(), complex_for(re->dtype()), re->device());

    auto bwd = std::make_shared<ComplexBackward>();
    bwd->shape_ = re->shape();
    bwd->device_ = re->device();
    kernel::NaryKernel<ComplexBackward, 2>::wire_autograd(std::move(bwd), {re, im}, result, false);
    return result;
}

const OpSchema ComplexBackward::schema_v1{"complex", 1, AmpPolicy::KeepInput, true};

std::vector<Storage> ComplexBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.complex_real(grad_out, shape_), be.complex_imag(grad_out, shape_)};
}

LUCID_REGISTER_OP(ComplexBackward)

}  // namespace lucid
