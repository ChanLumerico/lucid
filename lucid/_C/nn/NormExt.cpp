// lucid/_C/nn/NormExt.cpp
//
// Implementations of BatchNormEval, LpNormalize, and GlobalResponseNorm.
//
// BatchNormEval: uses externally supplied running statistics; does not update
//   them.  IBackend::batch_norm_eval_forward returns [y, rstd].
//   Backward: IBackend::batch_norm_eval_backward returns [dx, d_mean, d_var,
//   d_gamma, d_beta] (mean/var grads are zero in standard BN eval).
//
// LpNormalize: IBackend::lp_normalize_forward returns [y, norm].
//   Backward: IBackend::lp_normalize_backward returns dx.
//
// GlobalResponseNorm: IBackend::global_response_norm_forward returns [y, Nx].
//   Backward: IBackend::global_response_norm_backward returns [dx, d_gamma, d_beta].

#include "NormExt.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema BatchNormEvalBackward::schema_v1{"batch_norm_eval", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BatchNormEvalBackward::forward(const TensorImplPtr& x,
                                             const TensorImplPtr& mean,
                                             const TensorImplPtr& var,
                                             const TensorImplPtr& gamma,
                                             const TensorImplPtr& beta,
                                             double eps) {
    if (!x || !mean || !var || !gamma || !beta)
        ErrorBuilder("batch_norm_eval").fail("null input");
    if (x->shape().size() < 2)
        throw ShapeMismatch(x->shape(), Shape{}, "batch_norm_eval: expected >=2-D x");

    const int C = static_cast<int>(x->shape()[1]);
    // All stat tensors must be 1-D with length C.
    if (mean->shape().size() != 1 || mean->shape()[0] != C || var->shape().size() != 1 ||
        var->shape()[0] != C || gamma->shape().size() != 1 || gamma->shape()[0] != C ||
        beta->shape().size() != 1 || beta->shape()[0] != C) {
        throw ShapeMismatch(mean->shape(), x->shape(),
                            "batch_norm_eval: 1-D (C,) tensors required");
    }
    int spatial = 1;
    for (std::size_t i = 2; i < x->shape().size(); ++i)
        spatial *= static_cast<int>(x->shape()[i]);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};

    auto& be = backend::Dispatcher::for_device(x->device());
    // batch_norm_eval_forward returns [y, rstd].
    auto fwd =
        be.batch_norm_eval_forward(x->storage(), mean->storage(), var->storage(), gamma->storage(),
                                   beta->storage(), x->shape(), C, spatial, eps, x->dtype());
    Storage out_storage = std::move(fwd[0]);
    Storage rstd_storage = std::move(fwd[1]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x->shape(), x->dtype(),
                                            x->device(), false);

    auto bwd = std::make_shared<BatchNormEvalBackward>();
    bwd->eps_ = eps;
    bwd->rstd_ = std::move(rstd_storage);
    // saved_inputs_[0..4] will hold {x, mean, var, gamma, beta}.
    kernel::NaryKernel<BatchNormEvalBackward, 5>::wire_autograd(std::move(bwd),
                                                                {x, mean, var, gamma, beta}, out);
    return out;
}

std::vector<Storage> BatchNormEvalBackward::apply(Storage grad_out) {
    const Shape& xs = this->input_shapes_[0];
    const int C = static_cast<int>(xs[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < xs.size(); ++i)
        spatial *= static_cast<int>(xs[i]);
    auto& be = backend::Dispatcher::for_device(this->device_);
    // saved_inputs_[3] is gamma (index 3 of the five saved tensors).
    return be.batch_norm_eval_backward(this->saved_inputs_[0], this->saved_inputs_[1],
                                       this->saved_inputs_[3], this->rstd_, grad_out, xs, C,
                                       spatial, this->dtype_);
}

TensorImplPtr batch_norm_eval_op(const TensorImplPtr& x,
                                 const TensorImplPtr& mean,
                                 const TensorImplPtr& var,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps) {
    return BatchNormEvalBackward::forward(x, mean, var, gamma, beta, eps);
}

LUCID_REGISTER_OP(BatchNormEvalBackward)

const OpSchema LpNormalizeBackward::schema_v1{"lp_normalize", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr
LpNormalizeBackward::forward(const TensorImplPtr& x, double ord, int axis, double eps) {
    Validator::input(x, "lp_normalize.x").non_null();
    const int rank = static_cast<int>(x->shape().size());
    // Resolve negative axis to a non-negative index.
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
        ErrorBuilder("lp_normalize").fail("axis out of range");

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};

    auto& be = backend::Dispatcher::for_device(x->device());
    // lp_normalize_forward returns [y, norm].
    auto fwd = be.lp_normalize_forward(x->storage(), x->shape(), ord, axis, eps, x->dtype());
    Storage y_storage = std::move(fwd[0]);
    Storage norm_storage = std::move(fwd[1]);

    auto out = std::make_shared<TensorImpl>(std::move(y_storage), x->shape(), x->dtype(),
                                            x->device(), false);
    {
        auto bwd = std::make_shared<LpNormalizeBackward>();
        bwd->ord_ = ord;
        bwd->axis_ = axis;
        bwd->eps_ = eps;
        bwd->saved_norm_ = std::move(norm_storage);
        kernel::NaryKernel<LpNormalizeBackward, 1>::wire_autograd(std::move(bwd), {x}, out);
    }
    return out;
}

std::vector<Storage> LpNormalizeBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(this->device_);
    return {be.lp_normalize_backward(this->saved_inputs_[0], this->saved_norm_, grad_out,
                                     this->out_shape_, ord_, axis_, this->dtype_)};
}

TensorImplPtr lp_normalize_op(const TensorImplPtr& x, double ord, int axis, double eps) {
    return LpNormalizeBackward::forward(x, ord, axis, eps);
}

LUCID_REGISTER_OP(LpNormalizeBackward)

const OpSchema GlobalResponseNormBackward::schema_v1{"global_response_norm", 1,
                                                     AmpPolicy::ForceFP32, true};

TensorImplPtr GlobalResponseNormBackward::forward(const TensorImplPtr& x,
                                                  const TensorImplPtr& gamma,
                                                  const TensorImplPtr& beta,
                                                  double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("global_response_norm").fail("null input");
    if (x->shape().size() != 4)
        throw ShapeMismatch(x->shape(), Shape{}, "global_response_norm: x must be 4-D");
    const int C = static_cast<int>(x->shape()[1]);
    if (gamma->numel() != static_cast<std::size_t>(C) ||
        beta->numel() != static_cast<std::size_t>(C))
        throw ShapeMismatch(gamma->shape(), x->shape(),
                            "global_response_norm: gamma/beta must have C elements");

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};

    auto& be = backend::Dispatcher::for_device(x->device());
    // global_response_norm_forward returns [y, Nx] where Nx is the
    // normalized global response used for backprop.
    auto fwd = be.global_response_norm_forward(x->storage(), gamma->storage(), beta->storage(),
                                               x->shape(), eps, x->dtype());
    Storage out_storage = std::move(fwd[0]);
    Storage nx_storage = std::move(fwd[1]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x->shape(), x->dtype(),
                                            x->device(), false);
    auto bwd = std::make_shared<GlobalResponseNormBackward>();
    bwd->eps_ = eps;
    bwd->saved_Nx_ = std::move(nx_storage);
    // saved_inputs_[0..2] will hold {x, gamma, beta}.
    kernel::NaryKernel<GlobalResponseNormBackward, 3>::wire_autograd(std::move(bwd),
                                                                     {x, gamma, beta}, out);
    return out;
}

std::vector<Storage> GlobalResponseNormBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(this->device_);
    // Returns [dx, d_gamma, d_beta].
    return be.global_response_norm_backward(this->saved_inputs_[0], this->saved_inputs_[1],
                                            this->saved_inputs_[2], this->saved_Nx_, grad_out,
                                            this->input_shapes_[0], eps_, this->dtype_);
}

TensorImplPtr global_response_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps) {
    return GlobalResponseNormBackward::forward(x, gamma, beta, eps);
}

LUCID_REGISTER_OP(GlobalResponseNormBackward)

}  // namespace lucid
