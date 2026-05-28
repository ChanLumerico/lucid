// lucid/_C/nn/RMSNorm.cpp
//
// Implementation of Root Mean Square Layer Normalization.
//
// Shape resolution is performed inline (not via a helper): gamma's shape must
// match the trailing Dn dims of x, and the outer/N split is computed manually.
// Forward calls IBackend::rms_norm_forward, returning {y, rstd}.
// Backward calls IBackend::rms_norm_backward, returning {dx, d_gamma}.
// FLOP estimate: 4 * outer * N (rms + scale).

#include "RMSNorm.h"

#include <vector>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../core/AmpPolicy.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/NaryKernel.h"
#include "../ops/ufunc/Astype.h"

namespace lucid {

const OpSchema RMSNormBackward::schema_v1{"rms_norm", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr
RMSNormBackward::forward(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps) {
    if (!x || !gamma)
        ErrorBuilder("rms_norm").fail("null input");
    if (x->device() != gamma->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "rms_norm");

    // 3.4+ Phase A.8: AMP plumbing (same as BatchNorm / LayerNorm).
    // schema_v1.amp_policy = ForceFP32 — under autocast(F16) we upcast x
    // and gamma to F32 before the kernel so RMS variance over normalised
    // dims doesn't lose precision.
    SchemaGuard sg{RMSNormBackward::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);

    if (x_eff->dtype() != gamma_eff->dtype())
        throw DtypeMismatch(std::string(dtype_name(x_eff->dtype())),
                            std::string(dtype_name(gamma_eff->dtype())), "rms_norm");

    // Validate that gamma's shape matches the trailing dims of x.
    if (gamma_eff->shape().size() > x_eff->shape().size())
        throw ShapeMismatch(x_eff->shape(), gamma_eff->shape(), "rms_norm: γ has more dims than x");
    const std::size_t Dn = gamma_eff->shape().size();
    const std::size_t lead = x_eff->shape().size() - Dn;
    for (std::size_t i = 0; i < Dn; ++i) {
        if (x_eff->shape()[lead + i] != gamma_eff->shape()[i]) {
            throw ShapeMismatch(x_eff->shape(), gamma_eff->shape(),
                                "rms_norm: γ must match trailing dims of x");
        }
    }
    std::size_t outer = 1, N = 1;
    for (std::size_t i = 0; i < lead; ++i)
        outer *= static_cast<std::size_t>(x_eff->shape()[i]);
    for (std::size_t i = 0; i < Dn; ++i)
        N *= static_cast<std::size_t>(gamma_eff->shape()[i]);

    OpScopeFull scope{schema_v1.name, x_eff->device(), eff_dt, x_eff->shape()};
    // 3.5 Phase 1.2: report eps for the compile-path RMSNorm emitter.
    scope.set_attr("eps", eps);

    // rms_norm_forward returns {y, rstd}.
    auto forward = backend::Dispatcher::for_device(x_eff->device())
                       .rms_norm_forward(x_eff->storage(), gamma_eff->storage(), outer, N, eps,
                                         x_eff->shape(), eff_dt);
    scope.set_flops(static_cast<std::int64_t>(outer * N) * 4);

    auto out = std::make_shared<TensorImpl>(std::move(forward.first), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);

    auto bwd = std::make_shared<RMSNormBackward>();
    bwd->saved_rstd_ = std::move(forward.second);
    bwd->outer_ = outer;
    bwd->N_ = N;
    // saved_inputs_[0..1] hold {x, gamma} at eff_dt.
    kernel::NaryKernel<RMSNormBackward, 2>::wire_autograd(std::move(bwd), {x_eff, gamma_eff}, out);
    return out;
}

std::vector<Storage> RMSNormBackward::apply(Storage grad_out) {
    // rms_norm_backward returns {dx, d_gamma} as a pair.
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
