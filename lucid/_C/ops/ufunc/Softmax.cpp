// lucid/_C/ops/ufunc/Softmax.cpp
//
// Softmax forward and Jacobian-vector backward.  The backward is fully
// delegated to the backend (IBackend::softmax_backward) which computes:
//   dL/dx_i = p_i * (dL/dy_i - sum_j p_j * dL/dy_j)
// in a single fused pass to avoid materialising the N×N Jacobian.

#include "Softmax.h"

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"

namespace lucid {

// ForceFP32 prevents probability underflow; softmax on float16 is numerically
// unreliable for large logit magnitudes.
const OpSchema SoftmaxBackward::schema_v1{"softmax", 1, AmpPolicy::ForceFP32, true};

// Normalise the axis, dispatch the forward softmax, and wire the backward node.
// The output tensor is saved on the node (saved_output_) so apply() can use p
// directly without re-running the forward pass.
// set_flops(*5): each element needs roughly exp + div + two adds + a compare.
TensorImplPtr SoftmaxBackward::forward(const TensorImplPtr& a, int axis) {
    Validator::input(a, "softmax.a").non_null();

    const int ndim = static_cast<int>(a->shape().size());
    // Normalise negative axis to a canonical non-negative index.
    const int wrapped = axis < 0 ? axis + ndim : axis;
    if (wrapped < 0 || wrapped >= ndim)
        ErrorBuilder("softmax").index_error("axis out of range");

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .softmax(a->storage(), a->shape(), wrapped, a->dtype());

    auto result = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                               a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()) * 5);

    if (!GradMode::is_enabled() || !a->requires_grad())
        return result;

    auto bwd = std::make_shared<SoftmaxBackward>();
    bwd->saved_output_ = result->storage();  // p = softmax(x)
    bwd->axis_ = wrapped;
    // wire_autograd called with save_output=false because we already set
    // saved_output_ manually above.
    kernel::NaryKernel<SoftmaxBackward, 1>::wire_autograd(std::move(bwd), {a}, result, false);
    return result;
}

// dL/dx = p * (dL/dy - dot(dL/dy, p)) along axis_, computed by the backend.
std::vector<Storage> SoftmaxBackward::apply(Storage grad_out) {
    return {backend::Dispatcher::for_device(device_).softmax_backward(
        saved_output_, grad_out, input_shapes_[0], axis_, dtype_)};
}

TensorImplPtr softmax_op(const TensorImplPtr& a, int axis) {
    return SoftmaxBackward::forward(a, axis);
}
LUCID_REGISTER_OP(SoftmaxBackward)

// log_softmax — ForceFP32 for the same numerical stability reason as softmax.
const OpSchema LogSoftmaxBackward::schema_v1{"log_softmax", 1, AmpPolicy::ForceFP32, true};

// Forward: compute log_softmax via the numerically-stable backend,
// save the log_softmax output for use in the backward pass.
TensorImplPtr LogSoftmaxBackward::forward(const TensorImplPtr& a, int axis) {
    Validator::input(a, "log_softmax.a").non_null();

    const int ndim = static_cast<int>(a->shape().size());
    const int wrapped = axis < 0 ? axis + ndim : axis;
    if (wrapped < 0 || wrapped >= ndim)
        ErrorBuilder("log_softmax").index_error("axis out of range");

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .log_softmax(a->storage(), a->shape(), wrapped, a->dtype());

    auto result = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                               a->device(), false);
    if (!GradMode::is_enabled() || !a->requires_grad())
        return result;

    auto bwd = std::make_shared<LogSoftmaxBackward>();
    bwd->saved_output_ = result->storage();  // y = log_softmax(x)
    bwd->axis_ = wrapped;
    kernel::NaryKernel<LogSoftmaxBackward, 1>::wire_autograd(std::move(bwd), {a}, result, false);
    return result;
}

// Backward: dL/dx = dL/dy - exp(y) * sum(dL/dy, axis)
// where exp(y) = softmax(x) (probabilities).  Delegated to the backend so
// that broadcasting (sum_g expanded back to input shape) is handled correctly.
std::vector<Storage> LogSoftmaxBackward::apply(Storage grad_out) {
    const Shape& shape = input_shapes_[0];
    return {backend::Dispatcher::for_device(device_).log_softmax_backward(saved_output_, grad_out,
                                                                          shape, axis_, dtype_)};
}

TensorImplPtr log_softmax_op(const TensorImplPtr& a, int axis) {
    return LogSoftmaxBackward::forward(a, axis);
}
LUCID_REGISTER_OP(LogSoftmaxBackward)

}  // namespace lucid
