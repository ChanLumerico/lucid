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
#include "../ufunc/Astype.h"   // astype_op
#include "../utils/Layout.h"   // broadcast_to_op
#include "../utils/Promote.h"  // AmpScope
#include "Imag.h"
#include "Real.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr complex_op(const TensorImplPtr& re_in, const TensorImplPtr& im_in) {
    Validator::input(re_in, "complex.re").non_null();
    Validator::input(im_in, "complex.im").non_null();
    complex_detail::require_real_float(re_in->dtype(), "complex.re");
    complex_detail::require_real_float(im_in->dtype(), "complex.im");

    auto [re, im] = detail::align_binary_devices(re_in, im_in, "complex");

    // The two parts broadcast against each other, so building a complex
    // tensor from a real field and one scalar imaginary part is spelled
    // the obvious way.  Materialised only when the shapes differ.
    if (re->shape() != im->shape()) {
        const Shape out = detail::broadcast_shapes(re->shape(), im->shape());
        if (re->shape() != out)
            re = broadcast_to_op(re, out);
        if (im->shape() != out)
            im = broadcast_to_op(im, out);
    }

    // There is no half-precision complex, so the half formats widen here
    // rather than in the backend — which used to read their bytes as
    // float32 and assemble a number out of the wrong halves.
    if (is_half_float(re->dtype()))
        re = astype_op(re, Dtype::F32);
    if (is_half_float(im->dtype()))
        im = astype_op(im, Dtype::F32);
    if (re->dtype() != im->dtype())
        ErrorBuilder("complex").fail("real and imag must have the same dtype");

    // Opened after the widening, for two reasons: the scope's dtype is
    // the *output's*, so it has to be the complex type this actually
    // produces; and ``astype_op`` traces a node of its own, which would
    // otherwise be pushed after this one and take the I/O that
    // ``wire_autograd`` records below.
    OpScopeFull scope{"complex", re->device(), complex_for(re->dtype()), re->shape()};

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

TensorImplPtr view_as_complex_op(const TensorImplPtr& a_in) {
    Validator::input(a_in, "view_as_complex.a").non_null();
    TensorImplPtr a = AmpScope{"view_as_complex", a_in->device()}.cast(a_in);
    complex_detail::require_real_float(a->dtype(), "view_as_complex");
    if (a->dtype() == Dtype::BF16)
        ErrorBuilder("view_as_complex")
            .not_implemented("view_as_complex is only supported for float16, float32 and float64 "
                             "tensors, got bfloat16");
    if (a->shape().empty() || a->shape().back() != 2)
        ErrorBuilder("view_as_complex").invalid_argument("view_as_complex: expected last dim == 2");
    if (a->dtype() == Dtype::F16)
        a = astype_op(a, Dtype::F32);

    const std::size_t lane_bytes = dtype_size(a->dtype());
    const auto lane_elem = static_cast<std::int64_t>(lane_bytes);
    const auto complex_elem = lane_elem * 2;
    if (a->stride().back() != lane_elem)
        ErrorBuilder("view_as_complex")
            .invalid_argument("Tensor must have a last dimension with stride 1");
    if (static_cast<std::int64_t>(a->allocation_offset()) % complex_elem != 0)
        ErrorBuilder("view_as_complex")
            .invalid_argument("Tensor must have a storage_offset divisible by 2");

    const bool empty = shape_numel(a->shape()) == 0;
    Shape out_shape(a->shape().begin(), a->shape().end() - 1);
    Stride out_stride;
    out_stride.reserve(out_shape.size());
    for (std::size_t d = 0; d + 1 < a->shape().size(); ++d) {
        const std::int64_t s = a->stride()[d];
        if (s % complex_elem != 0) {
            if (empty || a->shape()[d] != 1)
                ErrorBuilder("view_as_complex")
                    .invalid_argument(
                        "Tensor must have a stride divisible by 2 for all but last dimension");
            out_stride.push_back(complex_elem);
        } else {
            out_stride.push_back(s);
        }
    }

    const Dtype out_dt = complex_for(a->dtype());
    OpScopeFull scope{"view_as_complex", a->device(), out_dt, out_shape};
    auto result =
        TensorImpl::make_dtype_view(a, out_dt, std::move(out_shape), std::move(out_stride), 0);

    auto bwd = std::make_shared<ViewAsComplexBackward>();
    bwd->shape_ = result->shape();
    bwd->lane_ = a->dtype();
    bwd->device_ = a->device();
    kernel::NaryKernel<ViewAsComplexBackward, 1>::wire_autograd(std::move(bwd), {a}, result, false);
    return result;
}

const OpSchema ComplexBackward::schema_v1{"complex", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema ComplexUnaryBackwardBase<ViewAsComplexBackward>::schema_v1{
    "view_as_complex", 1, AmpPolicy::KeepInput, true};

std::vector<Storage> ComplexBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.complex_real(grad_out, shape_), be.complex_imag(grad_out, shape_)};
}

std::vector<TensorImplPtr> ComplexBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    return {real_op(grad_out), imag_op(grad_out)};
}

template <>
std::vector<Storage> ComplexUnaryBackwardBase<ViewAsComplexBackward>::apply(Storage grad_out) {
    NoGradGuard no_grad;
    auto g = std::make_shared<TensorImpl>(std::move(grad_out), shape_, complex_for(lane_), device_,
                                          false);
    return {view_as_real_op(g)->storage()};
}

template <>
std::vector<TensorImplPtr>
ComplexUnaryBackwardBase<ViewAsComplexBackward>::apply_for_graph(const TensorImplPtr& grad_out) {
    return {view_as_real_op(grad_out)};
}

LUCID_REGISTER_OP(ComplexBackward)
LUCID_REGISTER_OP(ViewAsComplexBackward)

}  // namespace lucid
