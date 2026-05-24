// lucid/_C/nn/GroupNorm.cpp
//
// Implementation of Group Normalization.
//
// Layout expected: (B, C, S_0, ..., S_{ndim-3}) where ndim >= 2.
// The C channels are partitioned into G groups of (C/G) channels each.
// Statistics are computed per (b, g) slice spanning all channels in the
// group plus all spatial positions.
//
// Forward calls IBackend::group_norm_forward, returning [y, mean, rstd].
// Backward calls IBackend::group_norm_backward, returning [dx, d_gamma, d_beta].

#include "GroupNorm.h"

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
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/ufunc/Astype.h"

namespace lucid {

const OpSchema GroupNormBackward::schema_v1{"group_norm", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr GroupNormBackward::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& gamma,
                                         const TensorImplPtr& beta,
                                         int G,
                                         double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("group_norm").fail("null input");
    if (x->device() != gamma->device() || x->device() != beta->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "group_norm");

    // 3.4+ Phase A.8: AMP plumbing (same as BatchNorm / LayerNorm / RMSNorm).
    // schema_v1.amp_policy = ForceFP32 — cast x / γ / β to F32 under autocast(F16).
    SchemaGuard sg{GroupNormBackward::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);
    const TensorImplPtr beta_eff = astype_op(beta, eff_dt);

    if (x_eff->dtype() != gamma_eff->dtype() || x_eff->dtype() != beta_eff->dtype())
        throw DtypeMismatch(std::string(dtype_name(x_eff->dtype())),
                            std::string(dtype_name(gamma_eff->dtype())), "group_norm");
    // Rank >= 2 required (at least batch and channel dims).
    if (x_eff->device() == Device::CPU &&
        (!x_eff->is_contiguous() || !gamma_eff->is_contiguous() || !beta_eff->is_contiguous()))
        if (x_eff->shape().size() < 2)
            throw ShapeMismatch(x_eff->shape(), Shape{}, "group_norm: x must be at least (B, C, ...)");
    if (gamma_eff->shape().size() != 1 || beta_eff->shape().size() != 1)
        throw ShapeMismatch(gamma_eff->shape(), beta_eff->shape(),
                            "group_norm: γ, β must be 1-D");

    const int B = static_cast<int>(x_eff->shape()[0]);
    const int C = static_cast<int>(x_eff->shape()[1]);
    if (C % G != 0)
        ErrorBuilder("group_norm").fail("C must be divisible by num_groups");
    if (gamma_eff->shape()[0] != C || beta_eff->shape()[0] != C)
        throw ShapeMismatch(gamma_eff->shape(), x_eff->shape(), "group_norm: γ/β must have length C");

    const int N_spatial = static_cast<int>(x_eff->shape().size()) - 2;
    std::vector<int> S(N_spatial);
    int spatial_total = 1;
    for (int i = 0; i < N_spatial; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        spatial_total *= S[i];
    }
    OpScopeFull scope{schema_v1.name, x_eff->device(), eff_dt, x_eff->shape()};
    // 3.5 Phase 1.2: report eps + num_groups for the compile-path GN emitter.
    scope.set_attr("eps", eps);
    scope.set_attr("num_groups", static_cast<std::int64_t>(G));

    // group_norm_forward returns [y, mean, rstd].
    auto forward = backend::Dispatcher::for_device(x_eff->device())
                       .group_norm_forward(x_eff->storage(), gamma_eff->storage(),
                                           beta_eff->storage(), B, C, spatial_total, G, S, eps,
                                           x_eff->shape(), eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);
    auto bwd = std::make_shared<GroupNormBackward>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    bwd->B_ = B;
    bwd->C_ = C;
    bwd->G_ = G;
    bwd->spatial_dims_ = std::move(S);
    // saved_inputs_[0..2] hold {x, gamma, beta} at eff_dt.
    kernel::NaryKernel<GroupNormBackward, 3>::wire_autograd(std::move(bwd),
                                                            {x_eff, gamma_eff, beta_eff}, out);
    return out;
}

std::vector<Storage> GroupNormBackward::apply(Storage grad_out) {
    int spatial_total = 1;
    for (int s : spatial_dims_)
        spatial_total *= s;

    // Returns [dx, d_gamma, d_beta].
    return backend::Dispatcher::for_device(device_).group_norm_backward(
        saved_inputs_[0], saved_inputs_[1], saved_mean_, saved_rstd_, grad_out, B_, C_,
        spatial_total, G_, spatial_dims_, input_shapes_[0], dtype_);
}

TensorImplPtr group_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            int num_groups,
                            double eps) {
    return GroupNormBackward::forward(x, gamma, beta, num_groups, eps);
}

LUCID_REGISTER_OP(GroupNormBackward)

}  // namespace lucid
