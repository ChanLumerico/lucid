// lucid/_C/ops/complex/Real.cpp
//
// Forward implementation of real_op. Validates the input is complex, then
// dispatches through ``IBackend::complex_real`` (CPU = stride-2 walk,
// GPU = ``mlx::core::real``).

#include "Real.h"

#include "../../backend/Dispatcher.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../gfunc/Gfunc.h"
#include "Complex.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr real_op(const TensorImplPtr& a) {
    Validator::input(a, "real.a").non_null();
    complex_detail::require_complex(a->dtype(), "real");
    const Dtype lane = real_lane_of(a->dtype());
    // The scope's dtype is the output's: this strips a lane, so it is
    // the real one, not the complex input's.
    OpScopeFull scope{"real", a->device(), lane, a->shape()};

    auto result = TensorImpl::make_dtype_view(a, lane, a->shape(), a->stride(), 0);

    auto bwd = std::make_shared<RealBackward>();
    bwd->shape_ = a->shape();
    bwd->lane_ = lane;
    bwd->device_ = a->device();
    kernel::NaryKernel<RealBackward, 1>::wire_autograd(std::move(bwd), {a}, result, false);
    return result;
}

TensorImplPtr view_as_real_op(const TensorImplPtr& a) {
    Validator::input(a, "view_as_real.a").non_null();
    complex_detail::require_complex(a->dtype(), "view_as_real");
    const Dtype lane = real_lane_of(a->dtype());
    Shape out_shape = a->shape();
    out_shape.push_back(2);
    Stride out_stride = a->stride();
    out_stride.push_back(static_cast<std::int64_t>(dtype_size(lane)));
    OpScopeFull scope{"view_as_real", a->device(), lane, out_shape};

    auto result =
        TensorImpl::make_dtype_view(a, lane, std::move(out_shape), std::move(out_stride), 0);

    auto bwd = std::make_shared<ViewAsRealBackward>();
    bwd->shape_ = a->shape();
    bwd->lane_ = lane;
    bwd->device_ = a->device();
    kernel::NaryKernel<ViewAsRealBackward, 1>::wire_autograd(std::move(bwd), {a}, result, false);
    return result;
}

template <>
const OpSchema ComplexUnaryBackwardBase<RealBackward>::schema_v1{"real", 1, AmpPolicy::KeepInput,
                                                                 true};
template <>
const OpSchema ComplexUnaryBackwardBase<ViewAsRealBackward>::schema_v1{"view_as_real", 1,
                                                                       AmpPolicy::KeepInput, true};

template <>
std::vector<Storage> ComplexUnaryBackwardBase<RealBackward>::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    auto zero = zeros_op(shape_, lane_, device_);
    return {be.complex_combine(grad_out, zero->storage(), shape_)};
}

template <>
std::vector<TensorImplPtr>
ComplexUnaryBackwardBase<RealBackward>::apply_for_graph(const TensorImplPtr& grad_out) {
    return {complex_op(grad_out, zeros_op(shape_, lane_, device_))};
}

template <>
std::vector<Storage> ComplexUnaryBackwardBase<ViewAsRealBackward>::apply(Storage grad_out) {
    NoGradGuard no_grad;
    Shape in_shape = shape_;
    in_shape.push_back(2);
    auto g = std::make_shared<TensorImpl>(std::move(grad_out), std::move(in_shape), lane_, device_,
                                          false);
    return {view_as_complex_op(g)->storage()};
}

template <>
std::vector<TensorImplPtr>
ComplexUnaryBackwardBase<ViewAsRealBackward>::apply_for_graph(const TensorImplPtr& grad_out) {
    return {view_as_complex_op(grad_out)};
}

LUCID_REGISTER_OP(RealBackward)
LUCID_REGISTER_OP(ViewAsRealBackward)

}  // namespace lucid
