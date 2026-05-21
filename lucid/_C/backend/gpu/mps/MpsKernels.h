// lucid/_C/backend/gpu/mps/MpsKernels.h
//
// Per-op MPSGraph kernels.  Each kernel takes Lucid Storage inputs (typed
// as GpuStorage internally) and returns a Lucid Storage output wrapping a
// fresh MTLBuffer that MPSGraph allocated.
//
// Public surface is plain C++; the implementation in MpsKernels.mm is
// Obj-C++ for the MPSGraph API.

#pragma once

#include "../../../core/Dtype.h"
#include "../../../core/Shape.h"
#include "../../../core/Storage.h"

namespace lucid::gpu::mps {

// GELU forward — Lucid's tanh-approximation, but implemented as a single
// MPSGraph activation node.  Replaces the 7-op MLX expression in
// GpuBackend::gelu.
Storage gelu_forward(const Storage& x, const Shape& shape, Dtype dt);

// GELU tanh-approx backward — derivative of the tanh formulation:
//   dy/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*dinner
//   where t = tanh(c1 * (x + c2*x^3)), dinner = c1 * (1 + 3*c2*x^2).
Storage gelu_backward(const Storage& x,
                      const Storage& grad,
                      const Shape& shape,
                      Dtype dt);

// SiLU backward — dy/dx = sigmoid(x) * (1 + x*(1 - sigmoid(x))).
Storage silu_backward(const Storage& x,
                      const Storage& grad,
                      const Shape& shape,
                      Dtype dt);

// GELU exact (Gaussian-CDF) forward — replaces the 10-op Python
// _erf_approx composition + the MLX `0.5 * x * (1 + erf(x/√2))` fallback
// with a fused MPSGraph kernel.
Storage gelu_exact_forward(const Storage& x, const Shape& shape, Dtype dt);

// GELU exact backward — dy/dx = cdf(x) + x * pdf(x).  Fused MPSGraph.
Storage gelu_exact_backward(const Storage& x,
                            const Storage& grad,
                            const Shape& shape,
                            Dtype dt);

// LayerNorm backward — single fused MPSGraph executable producing
// (dx, dgamma, dbeta) from (x, gamma, saved_mean, saved_rstd, grad).
// Saved tensors have shape (outer, 1) and gamma/beta (normalized_size,).
struct LayerNormBackwardOut {
    Storage dx;
    Storage dgamma;
    Storage dbeta;
};
LayerNormBackwardOut layer_norm_backward(const Storage& x,
                                         const Storage& gamma,
                                         const Storage& saved_mean,
                                         const Storage& saved_rstd,
                                         const Storage& grad,
                                         std::size_t outer,
                                         std::size_t normalized_size,
                                         const Shape& x_shape,
                                         const Shape& gamma_shape,
                                         const Shape& beta_shape,
                                         Dtype dt);

// BatchNorm train forward — fused MPSGraph producing (y, saved_mean,
// saved_rstd).  ndim is the number of spatial dims (1, 2, or 3); the
// reduction is over axis 0 plus axes 2..2+ndim-1 (Lucid uses NCHW-style).
struct BatchNormForwardOut {
    Storage y;
    Storage mean;
    Storage rstd;
};
BatchNormForwardOut batch_norm_train_forward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& beta,
                                             int channels,
                                             int ndim,
                                             double eps,
                                             const Shape& x_shape,
                                             Dtype dt);

// BatchNorm train backward — fused MPSGraph producing (dx, dgamma, dbeta).
struct BatchNormBackwardOut {
    Storage dx;
    Storage dgamma;
    Storage dbeta;
};
BatchNormBackwardOut batch_norm_train_backward(const Storage& x,
                                               const Storage& gamma,
                                               const Storage& saved_mean,
                                               const Storage& saved_rstd,
                                               const Storage& grad,
                                               int channels,
                                               int ndim,
                                               const Shape& x_shape,
                                               Dtype dt,
                                               double eps);

// Softmax backward — fused MPSGraph chain `z * (grad - sum(z*grad, axis))`.
// `z` is the SAVED softmax output (not the input); axis is the reduction
// axis (already normalised to a positive index by the caller).
Storage softmax_backward(const Storage& z,
                         const Storage& grad,
                         int axis,
                         const Shape& shape,
                         Dtype dt);

}  // namespace lucid::gpu::mps
