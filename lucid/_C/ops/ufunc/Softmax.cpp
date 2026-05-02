#include "Softmax.h"

#include "../../autograd/AccumulateGrad.h"
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

const OpSchema SoftmaxBackward::schema_v1{"softmax", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr SoftmaxBackward::forward(const TensorImplPtr& a, int axis) {
    Validator::input(a, "softmax.a").non_null();

    const int ndim = static_cast<int>(a->shape().size());
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
    bwd->saved_output_ = result->storage();
    bwd->axis_ = wrapped;
    kernel::NaryKernel<SoftmaxBackward, 1>::wire_autograd(std::move(bwd), {a}, result,
                                                          /*save_ins=*/false);
    return result;
}

std::vector<Storage> SoftmaxBackward::apply(Storage grad_out) {
    return {backend::Dispatcher::for_device(device_).softmax_backward(
        saved_output_, grad_out, input_shapes_[0], axis_, dtype_)};
}

TensorImplPtr softmax_op(const TensorImplPtr& a, int axis) {
    return SoftmaxBackward::forward(a, axis);
}
LUCID_REGISTER_OP(SoftmaxBackward)

}  // namespace lucid
