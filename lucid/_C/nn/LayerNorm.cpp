// lucid/_C/nn/LayerNorm.cpp
//
// Implementation of the Layer Normalization forward pass and its autograd node.
//
// Shape resolution: given x of shape (d0, d1, ..., dk, n0, n1, ..., nm) and
// gamma of shape (n0, n1, ..., nm), the operation normalizes each of the
// outer = d0*...*dk slices of length N = n0*...*nm independently.
//
// Forward calls IBackend::layer_norm_forward, which returns [y, mean, rstd].
// Backward calls IBackend::layer_norm_backward, which computes [dx, d_gamma, d_beta].
// FLOP estimate: 5 * outer * N (normalize + affine).

#include "LayerNorm.h"

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

const OpSchema LayerNormBackward::schema_v1{"layer_norm", 1, AmpPolicy::ForceFP32, true};

namespace {

// Decomposed shape into (outer, N) where outer is the product of leading dims
// that are not normalized, and N is the product of normalized trailing dims.
struct LayerNormShapes {
    std::size_t outer;
    std::size_t N;
};

// Verify that gamma_shape matches the trailing Dn dims of x_shape and
// compute the outer/N split.
// Throws ShapeMismatch on any mismatch.
LayerNormShapes resolve_shapes(const Shape& x_shape, const Shape& gamma_shape) {
    if (gamma_shape.size() > x_shape.size()) {
        throw ShapeMismatch(x_shape, gamma_shape, "layer_norm: γ has more dims than x");
    }
    const std::size_t Dn = gamma_shape.size();
    const std::size_t lead = x_shape.size() - Dn;
    for (std::size_t i = 0; i < Dn; ++i) {
        if (x_shape[lead + i] != gamma_shape[i]) {
            throw ShapeMismatch(x_shape, gamma_shape,
                                "layer_norm: γ shape must match trailing dims of x");
        }
    }
    std::size_t outer = 1;
    for (std::size_t i = 0; i < lead; ++i)
        outer *= static_cast<std::size_t>(x_shape[i]);
    std::size_t N = 1;
    for (std::size_t i = 0; i < Dn; ++i)
        N *= static_cast<std::size_t>(gamma_shape[i]);
    return {outer, N};
}

}  // namespace

TensorImplPtr LayerNormBackward::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& gamma,
                                         const TensorImplPtr& beta,
                                         double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("layer_norm").fail("null input");
    if (x->dtype() != gamma->dtype() || x->dtype() != beta->dtype())
        throw DtypeMismatch(std::string(dtype_name(x->dtype())),
                            std::string(dtype_name(gamma->dtype())), "layer_norm");
    if (x->device() != gamma->device() || x->device() != beta->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "layer_norm");
    // Contiguity required on CPU; gamma/beta shape equality checked here.
    if (x->device() == Device::CPU &&
        (!x->is_contiguous() || !gamma->is_contiguous() || !beta->is_contiguous()))
        if (gamma->shape() != beta->shape())
            throw ShapeMismatch(gamma->shape(), beta->shape(),
                                "layer_norm: γ and β must have the same shape");

    const auto [outer, N] = resolve_shapes(x->shape(), gamma->shape());

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};

    // layer_norm_forward returns {y, mean, rstd}.
    auto forward = backend::Dispatcher::for_device(x->device())
                       .layer_norm_forward(x->storage(), gamma->storage(), beta->storage(), outer,
                                           N, eps, x->shape(), x->dtype());
    scope.set_flops(static_cast<std::int64_t>(outer * N) * 5);

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x->shape(), x->dtype(),
                                            x->device(), false);

    auto bwd = std::make_shared<LayerNormBackward>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    bwd->outer_ = outer;
    bwd->N_ = N;
    // saved_inputs_[0..2] will hold {x, gamma, beta} for backward.
    kernel::NaryKernel<LayerNormBackward, 3>::wire_autograd(std::move(bwd), {x, gamma, beta}, out);
    return out;
}

std::vector<Storage> LayerNormBackward::apply(Storage grad_out) {
    // Returns [dx, d_gamma, d_beta].
    return backend::Dispatcher::for_device(device_).layer_norm_backward(
        saved_inputs_[0], saved_inputs_[1], saved_mean_, saved_rstd_, grad_out, outer_, N_,
        input_shapes_[0], input_shapes_[1], input_shapes_[2], dtype_);
}

TensorImplPtr layer_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            double eps) {
    return LayerNormBackward::forward(x, gamma, beta, eps);
}

LUCID_REGISTER_OP(LayerNormBackward)

}  // namespace lucid
