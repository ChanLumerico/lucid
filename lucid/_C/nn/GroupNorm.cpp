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
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema GroupNormBackward::schema_v1{"group_norm", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr GroupNormBackward::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& gamma,
                                         const TensorImplPtr& beta,
                                         int G,
                                         double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("group_norm").fail("null input");
    if (x->dtype() != gamma->dtype() || x->dtype() != beta->dtype())
        throw DtypeMismatch(std::string(dtype_name(x->dtype())),
                            std::string(dtype_name(gamma->dtype())), "group_norm");
    if (x->device() != gamma->device() || x->device() != beta->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "group_norm");
    // Rank >= 2 required (at least batch and channel dims).
    if (x->device() == Device::CPU &&
        (!x->is_contiguous() || !gamma->is_contiguous() || !beta->is_contiguous()))
        if (x->shape().size() < 2)
            throw ShapeMismatch(x->shape(), Shape{}, "group_norm: x must be at least (B, C, ...)");
    if (gamma->shape().size() != 1 || beta->shape().size() != 1)
        throw ShapeMismatch(gamma->shape(), beta->shape(), "group_norm: γ, β must be 1-D");

    const int B = static_cast<int>(x->shape()[0]);
    const int C = static_cast<int>(x->shape()[1]);
    if (C % G != 0)
        ErrorBuilder("group_norm").fail("C must be divisible by num_groups");
    if (gamma->shape()[0] != C || beta->shape()[0] != C)
        throw ShapeMismatch(gamma->shape(), x->shape(), "group_norm: γ/β must have length C");

    const int N_spatial = static_cast<int>(x->shape().size()) - 2;
    std::vector<int> S(N_spatial);
    int spatial_total = 1;
    for (int i = 0; i < N_spatial; ++i) {
        S[i] = static_cast<int>(x->shape()[2 + i]);
        spatial_total *= S[i];
    }
    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};

    // group_norm_forward returns [y, mean, rstd].
    auto forward = backend::Dispatcher::for_device(x->device())
                       .group_norm_forward(x->storage(), gamma->storage(), beta->storage(), B, C,
                                           spatial_total, G, S, eps, x->shape(), x->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x->shape(), x->dtype(),
                                            x->device(), false);
    auto bwd = std::make_shared<GroupNormBackward>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    bwd->B_ = B;
    bwd->C_ = C;
    bwd->G_ = G;
    bwd->spatial_dims_ = std::move(S);
    // saved_inputs_[0..2] will hold {x, gamma, beta}.
    kernel::NaryKernel<GroupNormBackward, 3>::wire_autograd(std::move(bwd), {x, gamma, beta}, out);
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
