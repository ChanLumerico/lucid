#include "BatchNorm.h"

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

template <>
const OpSchema BatchNorm1dBackward::schema_v1{"batch_norm1d", 1, AmpPolicy::ForceFP32, true};
template <>
const OpSchema BatchNorm2dBackward::schema_v1{"batch_norm", 1, AmpPolicy::ForceFP32,
                                              true};  // keep "batch_norm" for backwards compat
template <>
const OpSchema BatchNorm3dBackward::schema_v1{"batch_norm3d", 1, AmpPolicy::ForceFP32, true};

template <int N>
TensorImplPtr BatchNormNdBackward<N>::forward(const TensorImplPtr& x,
                                              const TensorImplPtr& gamma,
                                              const TensorImplPtr& beta,
                                              double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("batch_norm").fail("null input");
    if (x->dtype() != gamma->dtype() || x->dtype() != beta->dtype())
        throw DtypeMismatch(std::string(dtype_name(x->dtype())),
                            std::string(dtype_name(gamma->dtype())), "batch_norm");
    if (x->device() != gamma->device() || x->device() != beta->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "batch_norm");
    if (x->device() == Device::CPU &&
        (!x->is_contiguous() || !gamma->is_contiguous() || !beta->is_contiguous()))
        if (static_cast<int>(x->shape().size()) != N + 2)
            throw ShapeMismatch(x->shape(), Shape{}, "batch_norm: x rank mismatch");
    if (gamma->shape().size() != 1 || beta->shape().size() != 1)
        throw ShapeMismatch(gamma->shape(), beta->shape(), "batch_norm: γ, β must be 1-D");

    const int B = static_cast<int>(x->shape()[0]);
    const int C = static_cast<int>(x->shape()[1]);
    int S[N > 0 ? N : 1];
    int spatial_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape()[2 + i]);
        spatial_total *= S[i];
    }
    if (gamma->shape()[0] != C || beta->shape()[0] != C)
        throw ShapeMismatch(gamma->shape(), x->shape(), "batch_norm: γ/β must have length C");

    OpScopeFull scope{BatchNormNdBackward<N>::schema_v1.name, x->device(), x->dtype(), x->shape()};

    auto forward = backend::Dispatcher::for_device(x->device())
                       .batch_norm_forward(x->storage(), gamma->storage(), beta->storage(), B, C,
                                           spatial_total, N, eps, x->shape(), x->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x->shape(), x->dtype(),
                                            x->device(), false);
    auto bwd = std::make_shared<BatchNormNdBackward<N>>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    bwd->B_ = B;
    bwd->C_ = C;
    for (int i = 0; i < N; ++i)
        bwd->S_[i] = S[i];
    kernel::NaryKernel<BatchNormNdBackward<N>, 3>::wire_autograd(std::move(bwd), {x, gamma, beta},
                                                                 out);
    return out;
}

template <int N>
std::vector<Storage> BatchNormNdBackward<N>::apply(Storage grad_out) {
    int spatial_total = 1;
    for (int i = 0; i < N; ++i)
        spatial_total *= this->S_[i];

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
