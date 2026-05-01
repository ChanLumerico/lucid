#include "RMSNorm.h"

#include <vector>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/NaryKernel.h"

namespace lucid {

const OpSchema RMSNormBackward::schema_v1{"rms_norm", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr RMSNormBackward::forward(const TensorImplPtr& x,
                                       const TensorImplPtr& gamma,
                                       double eps) {
    if (!x || !gamma)
        ErrorBuilder("rms_norm").fail("null input");
    if (x->dtype() != gamma->dtype())
        throw DtypeMismatch(std::string(dtype_name(x->dtype())),
                            std::string(dtype_name(gamma->dtype())), "rms_norm");
    if (x->device() != gamma->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "rms_norm");

    // γ shape must match trailing dims of x.
    if (gamma->shape().size() > x->shape().size())
        throw ShapeMismatch(x->shape(), gamma->shape(), "rms_norm: γ has more dims than x");
    const std::size_t Dn = gamma->shape().size();
    const std::size_t lead = x->shape().size() - Dn;
    for (std::size_t i = 0; i < Dn; ++i) {
        if (x->shape()[lead + i] != gamma->shape()[i]) {
            throw ShapeMismatch(x->shape(), gamma->shape(),
                                "rms_norm: γ must match trailing dims of x");
        }
    }
    std::size_t outer = 1, N = 1;
    for (std::size_t i = 0; i < lead; ++i)
        outer *= static_cast<std::size_t>(x->shape()[i]);
    for (std::size_t i = 0; i < Dn; ++i)
        N *= static_cast<std::size_t>(gamma->shape()[i]);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};

    auto forward = backend::Dispatcher::for_device(x->device())
                       .rms_norm_forward(x->storage(), gamma->storage(), outer, N, eps, x->shape(),
                                         x->dtype());
    scope.set_flops(static_cast<std::int64_t>(outer * N) * 4);

    auto out = std::make_shared<TensorImpl>(std::move(forward.first), x->shape(), x->dtype(),
                                            x->device(), /*requires_grad=*/false);

    auto bwd = std::make_shared<RMSNormBackward>();
    bwd->saved_rstd_ = std::move(forward.second);
    bwd->outer_ = outer;
    bwd->N_ = N;
    kernel::NaryKernel<RMSNormBackward, 2>::wire_autograd(std::move(bwd), {x, gamma}, out);
    return out;
}

std::vector<Storage> RMSNormBackward::apply(Storage grad_out) {
    auto grads = backend::Dispatcher::for_device(device_).rms_norm_backward(
        saved_inputs_[0], saved_inputs_[1], saved_rstd_, grad_out, outer_, N_, input_shapes_[0],
        input_shapes_[1], dtype_);
    return {std::move(grads.first), std::move(grads.second)};
}

TensorImplPtr rms_norm_op(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps) {
    return RMSNormBackward::forward(x, gamma, eps);
}

LUCID_REGISTER_OP(RMSNormBackward)

}  // namespace lucid
