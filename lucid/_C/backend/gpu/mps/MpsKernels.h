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

// GELU forward (tanh approximation) via a single MPSGraph activation node.
//
// Replaces the 7-op MLX expression inside :meth:`GpuBackend::gelu` with
// the equivalent fused MPSGraph node, eliminating the intermediate
// memory traffic.
//
// Parameters
// ----------
// x : const Storage&
//     Input activation tensor (GpuStorage).
// shape : const Shape&
//     Logical shape of ``x``.
// dt : Dtype
//     Element dtype.  ``F32`` or ``F16``.
//
// Returns
// -------
// Storage
//     Fresh GPU-resident result of identical shape + dtype.
//
// Math
// ----
// $$y = 0.5 \, x \, \left(1 + \tanh\!\bigl[c_1\,(x + c_2 x^3)\bigr]\right)$$
// where $c_1 = \sqrt{2/\pi}$, $c_2 = 0.044715$.
//
// See Also
// --------
// :func:`gelu_backward` — corresponding backward.
// :func:`gelu_exact_forward` — Gaussian-CDF variant.
Storage gelu_forward(const Storage& x, const Shape& shape, Dtype dt);

// GELU forward via a one-pass custom Metal compute kernel (tanh
// approximation, F32 only).
//
// Empirically the MPSGraph 9-op build above produces no measurable
// speedup vs MLX (both fuse into ~the same multi-op chain on
// M-series).  This kernel matches torch MPS's hand-tuned approach:
// read x once, evaluate the closed-form in registers, write y once.
// Routed when ``should_dispatch_gelu_metal`` returns true.
//
// Parameters
// ----------
// x : const Storage&
//     Input activation tensor (GpuStorage, F32 only).
// shape : const Shape&
//     Logical shape of ``x``.
// dt : Dtype
//     Element dtype.  Non-F32 silently falls back to
//     :func:`gelu_forward` (MPSGraph composite).
//
// Returns
// -------
// Storage
//     Fresh GPU result identical in shape + dtype to ``x``.
//
// See Also
// --------
// :func:`gelu_metal_backward` — corresponding backward.
// :func:`should_dispatch_gelu_metal` — dispatch gate.
Storage gelu_metal_forward(const Storage& x, const Shape& shape, Dtype dt);

// GELU backward via a one-pass custom Metal compute kernel (tanh
// approximation, F32 only).  Same rationale as
// :func:`gelu_metal_forward`.
//
// Parameters
// ----------
// x : const Storage&
//     Original forward input.
// grad : const Storage&
//     Upstream gradient.
// shape : const Shape&
//     Logical shape.
// dt : Dtype
//     Element dtype.
//
// Returns
// -------
// Storage
//     Gradient w.r.t. ``x``.
Storage gelu_metal_backward(const Storage& x,
                            const Storage& grad,
                            const Shape& shape,
                            Dtype dt);

// GELU exact (erf-based) forward via custom Metal kernel.  Matches the
// default ``F.gelu(x)`` path (``approximate="none"``).  Same wrapper
// pattern as :func:`gelu_metal_forward` — uses Metal's intrinsic
// ``erf`` for the closed-form computation.
//
// See Also
// --------
// :func:`gelu_metal_forward` — tanh-approx variant.
Storage gelu_exact_metal_forward(const Storage& x,
                                 const Shape& shape,
                                 Dtype dt);

// GELU exact (erf-based) backward via custom Metal kernel.
//
// Math
// ----
// ``dy/dx = Φ(x) + x · φ(x)`` where ``Φ`` is the standard normal CDF
// and ``φ`` its PDF.
Storage gelu_exact_metal_backward(const Storage& x,
                                  const Storage& grad,
                                  const Shape& shape,
                                  Dtype dt);

// GELU backward (tanh approximation).
//
// Fused MPSGraph evaluation of the derivative of the tanh
// formulation, given the input ``x`` and upstream gradient ``grad``.
//
// Parameters
// ----------
// x : const Storage&
//     Original forward input.
// grad : const Storage&
//     Upstream gradient w.r.t. the GELU output.
// shape : const Shape&
//     Logical shape (shared by ``x`` and ``grad``).
// dt : Dtype
//     Element dtype.
//
// Returns
// -------
// Storage
//     Gradient w.r.t. ``x``.
//
// Math
// ----
// $$\frac{\partial y}{\partial x} = 0.5(1 + t) + 0.5 \, x \, (1 - t^2) \, d_{\text{inner}}$$
// where $t = \tanh\!\bigl[c_1(x + c_2 x^3)\bigr]$ and
// $d_{\text{inner}} = c_1 (1 + 3 c_2 x^2)$.
//
// See Also
// --------
// :func:`gelu_forward` — corresponding forward.
Storage gelu_backward(const Storage& x,
                      const Storage& grad,
                      const Shape& shape,
                      Dtype dt);

// SiLU backward through a fused MPSGraph kernel.
//
// Parameters
// ----------
// x : const Storage&
//     Original forward input.
// grad : const Storage&
//     Upstream gradient w.r.t. the SiLU output.
// shape : const Shape&
//     Logical shape.
// dt : Dtype
//     Element dtype.
//
// Returns
// -------
// Storage
//     Gradient w.r.t. ``x``.
//
// Math
// ----
// $$\frac{\partial y}{\partial x} = \sigma(x) \, \bigl(1 + x \, (1 - \sigma(x))\bigr) \cdot \text{grad}$$
// where $\sigma$ is the logistic sigmoid.
//
// See Also
// --------
// :func:`should_dispatch_silu_backward` — dispatch gate.
Storage silu_backward(const Storage& x,
                      const Storage& grad,
                      const Shape& shape,
                      Dtype dt);

// GELU exact (Gaussian-CDF) forward via fused MPSGraph kernel.
//
// Replaces the 10-op MLX composition of the erf-approximation
// (``0.5 * x * (1 + erf(x/√2))``) with a single fused MPSGraph node.
//
// Parameters
// ----------
// x : const Storage&
//     Input activation.
// shape : const Shape&
//     Logical shape.
// dt : Dtype
//     Element dtype.
//
// Returns
// -------
// Storage
//     Fresh GPU result of identical shape + dtype.
//
// Math
// ----
// $$y = 0.5 \, x \, \left(1 + \operatorname{erf}\!\Bigl(\tfrac{x}{\sqrt{2}}\Bigr)\right)$$
//
// See Also
// --------
// :func:`gelu_exact_backward` — corresponding backward.
// :func:`gelu_forward` — tanh-approx variant.
Storage gelu_exact_forward(const Storage& x, const Shape& shape, Dtype dt);

// GELU exact (Gaussian-CDF) backward via fused MPSGraph kernel.
//
// Parameters
// ----------
// x : const Storage&
//     Original forward input.
// grad : const Storage&
//     Upstream gradient.
// shape : const Shape&
//     Logical shape.
// dt : Dtype
//     Element dtype.
//
// Returns
// -------
// Storage
//     Gradient w.r.t. ``x``.
//
// Math
// ----
// $$\frac{\partial y}{\partial x} = \Phi(x) + x \, \phi(x)$$
// where $\Phi$ is the standard normal CDF and $\phi$ its PDF.
Storage gelu_exact_backward(const Storage& x,
                            const Storage& grad,
                            const Shape& shape,
                            Dtype dt);

// Bundle of LayerNorm backward outputs.
//
// Attributes
// ----------
// dx : Storage
//     Gradient w.r.t. the layer input.
// dgamma : Storage
//     Gradient w.r.t. the scale parameter (shape
//     ``(normalized_size,)``).
// dbeta : Storage
//     Gradient w.r.t. the shift parameter (shape
//     ``(normalized_size,)``).
struct LayerNormBackwardOut {
    Storage dx;
    Storage dgamma;
    Storage dbeta;
};

// LayerNorm backward as a single fused MPSGraph executable.
//
// Computes ``(dx, dgamma, dbeta)`` in one MPSGraph dispatch using
// the saved mean / rstd from the forward.  Faster than the MLX
// composition for large ``normalized_size`` (see
// :func:`should_dispatch_layer_norm_backward`).
//
// Parameters
// ----------
// x : const Storage&
//     Original forward input.
// gamma : const Storage&
//     Scale parameter of shape ``(normalized_size,)``.
// saved_mean : const Storage&
//     Per-row mean from the forward, shape ``(outer, 1)``.
// saved_rstd : const Storage&
//     Per-row reciprocal-stddev from the forward, shape ``(outer, 1)``.
// grad : const Storage&
//     Upstream gradient w.r.t. the layer output.
// outer : std::size_t
//     Product of leading (non-normalised) dimensions.
// normalized_size : std::size_t
//     Size of the trailing normalised dimension.
// x_shape : const Shape&
//     Logical shape of ``x`` (and ``grad``).
// gamma_shape : const Shape&
//     Logical shape of ``gamma``.
// beta_shape : const Shape&
//     Logical shape of ``beta``.  Used for the output ``dbeta``.
// dt : Dtype
//     Element dtype.
//
// Returns
// -------
// LayerNormBackwardOut
//     The three gradients.
//
// Math
// ----
// $$\hat{x} = (x - \mu) \cdot \mathrm{rstd}, \quad y = \gamma \hat{x} + \beta$$
// $$d_\gamma = \sum_{\text{outer}} \mathrm{grad} \cdot \hat{x}, \quad
//   d_\beta  = \sum_{\text{outer}} \mathrm{grad}$$
//
// See Also
// --------
// :func:`should_dispatch_layer_norm_backward` — dispatch gate.
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

// Bundle of BatchNorm train forward outputs.
//
// Attributes
// ----------
// y : Storage
//     Normalised activation (same shape as input).
// mean : Storage
//     Per-channel mean (shape ``(C,)``).  Saved for backward.
// rstd : Storage
//     Per-channel reciprocal-stddev (shape ``(C,)``).  Saved for
//     backward.
struct BatchNormForwardOut {
    Storage y;
    Storage mean;
    Storage rstd;
};

// BatchNorm training-mode forward as one fused MPSGraph executable.
//
// Produces ``(y, saved_mean, saved_rstd)`` in a single dispatch.
// MLX has no fused BN primitive, so this kernel exists primarily to
// close the gap on large activations (see
// :func:`should_dispatch_batch_norm_train`).
//
// Parameters
// ----------
// x : const Storage&
//     Input activation, NCHW-style layout.
// gamma : const Storage&
//     Per-channel scale, shape ``(C,)``.
// beta : const Storage&
//     Per-channel shift, shape ``(C,)``.
// channels : int
//     Number of channels ``C``.
// ndim : int
//     Number of spatial dimensions (1, 2, or 3).
// eps : double
//     Numerical stability epsilon added inside the rsqrt.
// x_shape : const Shape&
//     Logical shape of ``x``.
// dt : Dtype
//     Element dtype.
//
// Returns
// -------
// BatchNormForwardOut
//     Output activation + saved mean + saved rstd.
//
// Math
// ----
// $$\mu_c = \mathbb{E}[x_{n,c,\ldots}], \quad
//   \mathrm{rstd}_c = (\operatorname{Var}[x_{n,c,\ldots}] + \varepsilon)^{-1/2}$$
// $$y_{n,c,\ldots} = \gamma_c \cdot (x_{n,c,\ldots} - \mu_c) \cdot \mathrm{rstd}_c + \beta_c$$
// Reduction axes: $0, 2, 3, \ldots, 2 + \mathrm{ndim} - 1$.
//
// See Also
// --------
// :func:`batch_norm_train_backward` — corresponding backward.
// :func:`should_dispatch_batch_norm_train` — dispatch gate.
BatchNormForwardOut batch_norm_train_forward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& beta,
                                             int channels,
                                             int ndim,
                                             double eps,
                                             const Shape& x_shape,
                                             Dtype dt);

// Bundle of BatchNorm train backward outputs.
//
// Attributes
// ----------
// dx : Storage
//     Gradient w.r.t. ``x`` (same shape as the forward input).
// dgamma : Storage
//     Gradient w.r.t. ``gamma`` (shape ``(C,)``).
// dbeta : Storage
//     Gradient w.r.t. ``beta`` (shape ``(C,)``).
struct BatchNormBackwardOut {
    Storage dx;
    Storage dgamma;
    Storage dbeta;
};

// BatchNorm training-mode backward as one fused MPSGraph executable.
//
// Computes ``(dx, dgamma, dbeta)`` in a single dispatch using the
// saved mean/rstd from the forward.
//
// Parameters
// ----------
// x : const Storage&
//     Original forward input.
// gamma : const Storage&
//     Per-channel scale ``(C,)``.
// saved_mean : const Storage&
//     Per-channel mean from forward ``(C,)``.
// saved_rstd : const Storage&
//     Per-channel rstd from forward ``(C,)``.
// grad : const Storage&
//     Upstream gradient w.r.t. the BN output.
// channels : int
//     Channel count.
// ndim : int
//     Spatial dimension count (1, 2, or 3).
// x_shape : const Shape&
//     Logical shape of ``x``.
// dt : Dtype
//     Element dtype.
// eps : double
//     Numerical-stability epsilon used in the forward.  Needed here
//     for symbolic consistency; the gradient itself uses ``saved_rstd``.
//
// Returns
// -------
// BatchNormBackwardOut
//     The three gradients.
//
// See Also
// --------
// :func:`batch_norm_train_forward` — corresponding forward.
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

// Softmax backward as a fused MPSGraph kernel.
//
// Evaluates the canonical formula ``z * (grad - sum(z*grad, axis))``
// in a single dispatch, where ``z`` is the **saved softmax output**
// (not the input) and ``axis`` has already been normalised to a
// non-negative index by the caller.
//
// Parameters
// ----------
// z : const Storage&
//     Saved softmax output from the forward pass.
// grad : const Storage&
//     Upstream gradient w.r.t. the softmax output.
// axis : int
//     Normalised reduction axis (``0 <= axis < z.ndim``).
// shape : const Shape&
//     Logical shape of ``z`` (and ``grad``).
// dt : Dtype
//     Element dtype.
//
// Returns
// -------
// Storage
//     Gradient w.r.t. the softmax input.
//
// Math
// ----
// $$\frac{\partial L}{\partial x_i} = z_i \left(g_i - \sum_j z_j g_j\right)$$
// reducing along ``axis``.
//
// See Also
// --------
// :func:`should_dispatch_softmax_backward` — dispatch gate.
Storage softmax_backward(const Storage& z,
                         const Storage& grad,
                         int axis,
                         const Shape& shape,
                         Dtype dt);

// Embedding backward — scatter-add gradient rows into the weight grad
// table via a single fused MPSGraph kernel.
//
// Replaces the MLX ``scatter_add_axis`` composition inside
// :meth:`GpuBackend::embedding_backward` with MPSGraph's native
// ``MPSGraphScatterModeAdd`` primitive.  The MPS path is ~28× faster
// on GPT-2-scale inputs (M_total × D ≥ 6M) per
// ``perf-mlx-op-baseline-2026-05.md``; gated behind
// :func:`should_dispatch_embedding_backward`.
//
// Parameters
// ----------
// grad_out : const Storage&
//     Upstream gradient, shape ``indices_shape + (D,)`` flattened to
//     ``(M_total, D)`` internally.
// indices : const Storage&
//     Integer indices into the embedding table, shape
//     ``indices_shape``.  Flattened to ``(M_total,)`` internally.
//     Dtype is int64 (auto-cast inside the kernel if needed).
// N : std::int64_t
//     Vocabulary size — first dim of the weight table.  Defines the
//     output rows of ``dW``.
// D : std::int64_t
//     Embedding dimension — second dim of the weight table.
// M_total : std::int64_t
//     Product of ``indices_shape``.  The number of scatter rows.
// padding_idx : int
//     If ``>= 0``, rows of ``grad_out`` whose corresponding index
//     equals ``padding_idx`` are zeroed before the scatter-add (so
//     the padding row of ``dW`` stays at zero).  ``< 0`` disables
//     the mask.
// dt : Dtype
//     Element dtype of ``grad_out`` and the output ``dW``.
//
// Returns
// -------
// Storage
//     Gradient w.r.t. the embedding weight table, shape ``(N, D)``.
//
// See Also
// --------
// :func:`should_dispatch_embedding_backward` — dispatch gate.
Storage embedding_backward(const Storage& grad_out,
                           const Storage& indices,
                           std::int64_t N,
                           std::int64_t D,
                           std::int64_t M_total,
                           int padding_idx,
                           Dtype dt);

}  // namespace lucid::gpu::mps
