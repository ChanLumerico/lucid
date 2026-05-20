// lucid/_C/nn/BatchNorm.cpp
//
// Training-mode Batch Normalization for 1-D, 2-D, and 3-D inputs.
//
// For each channel c, statistics are computed over the (B, S_total) axes:
//   mean_c  = mean over all (b, spatial) pairs
//   rstd_c  = 1 / sqrt(var_c + eps)
//   y_{b,c,...} = (x_{b,c,...} - mean_c) * rstd_c * gamma_c + beta_c
//
// The forward delegates to IBackend::batch_norm_forward, which returns
// [y, mean, rstd].  The backward delegates to IBackend::batch_norm_backward,
// which returns [dx, d_gamma, d_beta].  Running-stats inference is handled by
// BatchNormEvalBackward in NormExt.h.

#include "BatchNorm.h"

#include <vector>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/BinaryKernel.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/ufunc/Astype.h"

namespace lucid {

template <>
const OpSchema BatchNorm1dBackward::schema_v1{"batch_norm1d", 1, AmpPolicy::ForceFP32, true};
template <>
const OpSchema BatchNorm2dBackward::schema_v1{"batch_norm", 1, AmpPolicy::ForceFP32, true};
template <>
const OpSchema BatchNorm3dBackward::schema_v1{"batch_norm3d", 1, AmpPolicy::ForceFP32, true};

template <int N>
TensorImplPtr BatchNormNdBackward<N>::forward(const TensorImplPtr& x,
                                              const TensorImplPtr& gamma,
                                              const TensorImplPtr& beta,
                                              double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("batch_norm").fail("null input");
    if (x->device() != gamma->device() || x->device() != beta->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "batch_norm");
    // Rank check: x must be (B, C, S_0, ..., S_{N-1}).
    if (x->device() == Device::CPU &&
        (!x->is_contiguous() || !gamma->is_contiguous() || !beta->is_contiguous()))
        if (static_cast<int>(x->shape().size()) != N + 2)
            throw ShapeMismatch(x->shape(), Shape{}, "batch_norm: x rank mismatch");
    if (gamma->shape().size() != 1 || beta->shape().size() != 1)
        throw ShapeMismatch(gamma->shape(), beta->shape(), "batch_norm: γ, β must be 1-D");

    // 3.3 AMP plumbing: schema_v1.amp_policy == ForceFP32.  Under
    // ``AutocastGuard(F16)`` SchemaGuard returns ``Dtype::F32`` regardless
    // of input dtype — BN's batch statistics are numerically sensitive
    // and running them in F16 risks catastrophic cancellation of the
    // mean/variance reductions.  The cast MUST happen before the strict
    // ``x->dtype() != gamma->dtype()`` check below: under autocast, the
    // surrounding Conv has cast x to F16 while gamma / beta on the
    // ``nn.BatchNorm`` Parameter slots are still F32.  After the cast
    // all three operands share ``eff_dt`` and the dtype-match invariant
    // holds.  Outside an autocast scope this is a no-op (``astype_op``
    // returns the input unchanged when dtypes already match).
    //
    // ``astype_op`` (not ``maybe_cast_for_kernel``) is used so the cast
    // tensors carry an ``AstypeBackward`` grad_fn.  Without this the
    // F32-cast x_eff has requires_grad=false and ``wire_autograd`` would
    // drop the entire BN backward chain under AMP.
    SchemaGuard sg{BatchNormNdBackward<N>::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);
    const TensorImplPtr beta_eff = astype_op(beta, eff_dt);

    // Trivially true after the AMP cast above — kept as a defensive
    // assertion in case ``astype_op`` ever returns an input whose dtype
    // doesn't match eff_dt (would indicate a backend bug).
    if (x_eff->dtype() != gamma_eff->dtype() || x_eff->dtype() != beta_eff->dtype())
        throw DtypeMismatch(std::string(dtype_name(x_eff->dtype())),
                            std::string(dtype_name(gamma_eff->dtype())), "batch_norm");

    const int B = static_cast<int>(x_eff->shape()[0]);
    const int C = static_cast<int>(x_eff->shape()[1]);
    int S[N > 0 ? N : 1];
    int spatial_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        spatial_total *= S[i];
    }
    if (gamma_eff->shape()[0] != C || beta_eff->shape()[0] != C)
        throw ShapeMismatch(gamma_eff->shape(), x_eff->shape(),
                            "batch_norm: γ/β must have length C");

    OpScopeFull scope{BatchNormNdBackward<N>::schema_v1.name, x_eff->device(), eff_dt,
                      x_eff->shape()};

    // batch_norm_forward returns [y, mean, rstd].
    auto forward = backend::Dispatcher::for_device(x_eff->device())
                       .batch_norm_forward(x_eff->storage(), gamma_eff->storage(),
                                           beta_eff->storage(), B, C, spatial_total, N, eps,
                                           x_eff->shape(), eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);
    auto bwd = std::make_shared<BatchNormNdBackward<N>>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    bwd->B_ = B;
    bwd->C_ = C;
    for (int i = 0; i < N; ++i)
        bwd->S_[i] = S[i];
    // saved_inputs_[0..2] hold {x, gamma, beta} at eff_dt.
    kernel::NaryKernel<BatchNormNdBackward<N>, 3>::wire_autograd(
        std::move(bwd), {x_eff, gamma_eff, beta_eff}, out);
    return out;
}

template <int N>
std::vector<Storage> BatchNormNdBackward<N>::apply(Storage grad_out) {
    int spatial_total = 1;
    for (int i = 0; i < N; ++i)
        spatial_total *= this->S_[i];

    // Returns [dx, d_gamma, d_beta].
    return backend::Dispatcher::for_device(this->device_)
        .batch_norm_backward(this->saved_inputs_[0], this->saved_inputs_[1], this->saved_mean_,
                             this->saved_rstd_, grad_out, this->B_, this->C_, spatial_total, N,
                             this->input_shapes_[0], this->dtype_);
}

template class BatchNormNdBackward<1>;
template class BatchNormNdBackward<2>;
template class BatchNormNdBackward<3>;

TensorImplPtr batch_norm1d_op(const TensorImplPtr& x,
                              const TensorImplPtr& gamma,
                              const TensorImplPtr& beta,
                              double eps) {
    return BatchNorm1dBackward::forward(x, gamma, beta, eps);
}
TensorImplPtr batch_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            double eps) {
    return BatchNorm2dBackward::forward(x, gamma, beta, eps);
}
TensorImplPtr batch_norm3d_op(const TensorImplPtr& x,
                              const TensorImplPtr& gamma,
                              const TensorImplPtr& beta,
                              double eps) {
    return BatchNorm3dBackward::forward(x, gamma, beta, eps);
}

LUCID_REGISTER_OP(BatchNorm1dBackward)
LUCID_REGISTER_OP(BatchNorm2dBackward)
LUCID_REGISTER_OP(BatchNorm3dBackward)

}  // namespace lucid
