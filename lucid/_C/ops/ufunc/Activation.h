// lucid/_C/ops/ufunc/Activation.h
//
// Autograd backward nodes and entry points for neural-network activation
// functions: relu, sigmoid, silu/swish, gelu, leaky_relu, softplus, elu, selu,
// mish, hard_sigmoid, hard_swish, relu6.
//
// Most activations delegate the analytically complex backward computation to
// the backend dispatcher (IBackend::{activation}_backward) so that CPU can use
// Apple Accelerate and GPU can use MLX.  Simpler activations (relu, sigmoid,
// relu6) implement grad_formula directly using storage primitives for clarity.
//
// Ops with a scalar hyper-parameter (leaky_relu slope, elu alpha) override the
// standard static forward() from UnaryKernel rather than using dispatch(), so
// they can capture and persist the parameter in the backward node.

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Autograd node for the element-wise rectified linear unit $y = \max(0, x)$.
//
// Saves the input so the backward pass can rebuild the positive-input mask
// $\mathbb{1}\{x > 0\}$ and gate the upstream gradient through it.  At the
// non-differentiable point $x = 0$ the subgradient is taken to be $0$, which
// matches the reference framework's convention.
//
// Math
// ----
// $$y = \max(0, x), \qquad \frac{\partial y}{\partial x} = \mathbb{1}\{x > 0\}$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"relu"``, ``AmpPolicy::KeepInput`` (ReLU is well-defined on integer
//     dtypes, so the AMP layer leaves the input dtype untouched).
//
// Notes
// -----
// MLX exposes a native ``maximum(x, 0)`` kernel on the GPU stream; the CPU
// path goes through ``vDSP_vmax`` against a zero buffer.  ``grad_formula_impl``
// (used by ``create_graph=True``) builds the mask as ``sign(relu(x))`` so the
// inverse-mask multiplication stays inside the autograd graph for second-order
// differentiation.
class LUCID_API ReluBackward : public UnaryOp<ReluBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::relu`` to compute $y = \max(0, x)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.relu(a, s, dt);
    }

    // Eager-mode backward: $\partial L/\partial x = \mathbb{1}\{x > 0\} \cdot
    // \partial L/\partial y$, computed via a positive-input mask multiplied
    // element-wise against the upstream gradient.
    //
    // Parameters
    // ----------
    // g : Storage
    //     Upstream gradient ``dL/dy`` matching the forward output shape.
    //
    // Returns
    // -------
    // Storage
    //     ``dL/dx`` at the input shape and dtype.
    Storage grad_formula(const Storage& g);

    // Graph-mode backward used when ``create_graph=True``.  Builds the mask
    // as ``sign(relu(x))`` (zero for $x \le 0$, one for $x > 0$) so the
    // operation is itself differentiable for higher-order gradients.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Autograd node for the logistic sigmoid $y = 1 / (1 + e^{-x})$.
//
// Saves the *output* $y$ rather than the input because the gradient formula
// $y(1 - y)$ depends only on $y$.  This avoids a redundant forward pass and
// keeps the saved tensor at the same dtype as the activation.
//
// Math
// ----
// $$y = \frac{1}{1 + e^{-x}}, \qquad \frac{\partial y}{\partial x} = y\,(1 - y)$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"sigmoid"``, ``AmpPolicy::Promote`` — half-precision inputs are
//     promoted to float32 to keep $e^{-x}$ from underflowing for large $|x|$.
//
// Notes
// -----
// The eager backward builds $(1 - y)$ via ``mul_scalar(-1) + add_scalar(1)``
// to reuse the existing storage primitives rather than introducing a dedicated
// subtract kernel.
class LUCID_API SigmoidBackward : public UnaryOp<SigmoidBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::sigmoid`` to compute $y = 1 / (1 + e^{-x})$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sigmoid(a, s, dt);
    }

    // Eager-mode backward: $\partial L/\partial x = y(1 - y) \cdot
    // \partial L/\partial y$ using the saved output $y$.
    Storage grad_formula(const Storage& g);

    // Graph-mode backward: ``out * (1 - out) * g`` expressed as composable
    // ops so the chain remains differentiable.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr& out);
};

// Autograd node for the SiLU / Swish activation $y = x \cdot \sigma(x)$
// (Hendrycks & Gimpel 2016 / Ramachandran et al. 2017).
//
// Saves the input $x$; the backward delegates to the backend so the fused
// gradient kernel ($\sigma(x)\,(1 + x(1 - \sigma(x)))$) runs as a single pass
// on MLX or a single Accelerate composition on CPU rather than the prior
// seven-op storage-primitive expansion.
//
// Math
// ----
// $$y = x\,\sigma(x), \qquad \frac{\partial y}{\partial x}
//   = \sigma(x)\,\bigl(1 + x\,(1 - \sigma(x))\bigr)$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"silu"``, ``AmpPolicy::Promote``.
//
// References
// ----------
// Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", 2016.
// Ramachandran et al., "Searching for Activation Functions", 2017.
class LUCID_API SiluBackward : public UnaryOp<SiluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::silu`` to compute $y = x \cdot \sigma(x)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.silu(a, s, dt);
    }

    // Eager-mode backward: dispatches to ``IBackend::silu_backward`` with the
    // saved input and the upstream gradient.
    Storage grad_formula(const Storage& g);
};

// Autograd node for the tanh-approximation GeLU
// $y = 0.5\,x\,(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3)))$.
//
// The approximate form (Hendrycks & Gimpel 2016, eq. 3) is used by default
// when callers do not request the exact erf-based variant; it is the form
// implemented by ``F.gelu(x, approximate="tanh")`` in the Python wrapper.
// The backward is delegated to the backend because the analytic derivative
// involves both ``tanh`` and ``sech^2`` over a cubic polynomial of $x$, and
// fusing them in a single kernel is materially faster than chaining storage
// primitives.
//
// Math
// ----
// $$y = \tfrac{x}{2}\,\bigl(1 + \tanh\bigl(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\bigr)\bigr)$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"gelu"``, ``AmpPolicy::ForceFP32`` — the cubic + ``tanh`` chain is
//     not numerically safe in float16.
//
// References
// ----------
// Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415.
class LUCID_API GeluBackward : public UnaryOp<GeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::gelu`` to compute the tanh-approximation
    // GeLU $y = 0.5\,x\,(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3)))$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.gelu(a, s, dt);
    }

    // Eager-mode backward: dispatches to ``IBackend::gelu_backward`` with
    // the saved input and the upstream gradient.
    Storage grad_formula(const Storage& g);
};

// Autograd node for the exact (Gaussian-CDF) GeLU
// $y = 0.5\,x\,(1 + \mathrm{erf}(x/\sqrt{2}))$.
//
// Routed to by ``F.gelu(x, approximate="none")``.  Implementing this as a
// dedicated op (instead of the prior ten-op Python composition built around
// ``_erf_approx``) lets the GPU backend dispatch to a single MPSGraph node
// and keeps the autograd graph compact for memory accounting.
//
// Math
// ----
// $$y = \tfrac{x}{2}\,\Bigl(1 + \mathrm{erf}\bigl(\tfrac{x}{\sqrt{2}}\bigr)\Bigr),
//   \qquad \frac{\partial y}{\partial x}
//   = \tfrac{1}{2}\,\Bigl(1 + \mathrm{erf}\bigl(\tfrac{x}{\sqrt{2}}\bigr)\Bigr)
//     + \tfrac{x}{\sqrt{2\pi}}\,e^{-x^2/2}$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"gelu_exact"``, ``AmpPolicy::ForceFP32`` — matches the tanh-approx
//     variant for consistent AMP semantics across approximations.
class LUCID_API GeluExactBackward : public UnaryOp<GeluExactBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::gelu_exact`` to compute the erf-based exact
    // GeLU $y = 0.5\,x\,(1 + \mathrm{erf}(x/\sqrt{2}))$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.gelu_exact(a, s, dt);
    }

    // Eager-mode backward: dispatches to ``IBackend::gelu_exact_backward``.
    Storage grad_formula(const Storage& g);
};

// Autograd node for Leaky ReLU $y = x$ if $x > 0$ else $\text{slope} \cdot x$.
//
// The slope hyper-parameter is captured on the node so the eager backward
// can rebuild the leaky mask without re-reading it from outside the graph.
// Because the standard ``UnaryKernel::forward`` does not thread the slope
// argument through ``dispatch()``, an explicit ``forward()`` override is
// used; ``cpu_kernel`` is provided alongside it so the per-device dispatcher
// can still pick a CPU path when MLX is unavailable.
//
// Math
// ----
// $$y = \begin{cases} x, & x > 0 \\ s\,x, & x \le 0 \end{cases},
//   \qquad \frac{\partial y}{\partial x}
//   = \begin{cases} 1, & x > 0 \\ s, & x \le 0 \end{cases}$$
//
// where $s$ is the configured slope.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"leaky_relu"``, ``AmpPolicy::KeepInput``.
// slope_ : double
//     Negative-side slope captured at forward time; defaults to ``0.01``.
class LUCID_API LeakyReluBackward : public UnaryOp<LeakyReluBackward> {
public:
    static constexpr bool kSavesInput = true;
    double slope_ = 0.01;
    static const OpSchema schema_v1;

    // Custom forward that captures ``slope`` on the backward node and wires
    // the autograd edges manually (the default ``UnaryKernel::forward``
    // signature does not thread an extra parameter through ``dispatch()``).
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input activation.
    // slope : double
    //     Negative-side slope ``s`` such that $y = s\,x$ for $x \le 0$.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Forward output with autograd edges wired to ``LeakyReluBackward``.
    static TensorImplPtr forward(const TensorImplPtr& a, double slope);

    // CPU kernel used by the dispatcher; mirrors ``forward`` but acts on a
    // pre-resolved CPU storage.
    static CpuStorage
    cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt, double slope);

    // Eager-mode backward: applies the per-element leaky mask
    // (``1`` for $x > 0$, ``slope_`` otherwise) to the upstream gradient.
    Storage grad_formula(const Storage& g);
};

// Autograd node for Softplus $y = \log(1 + e^x)$ — the smooth surrogate of
// ReLU whose derivative is the logistic sigmoid.
//
// Saves the input so ``grad_formula`` can recompute $\sigma(x)$ via the
// storage-level sigmoid primitive (a single MLX or Accelerate call).
//
// Math
// ----
// $$y = \log(1 + e^x), \qquad \frac{\partial y}{\partial x} = \sigma(x)$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"softplus"``, ``AmpPolicy::ForceFP32`` — guards against $e^x$
//     overflow in float16 for $|x| > 11$ or so.
class LUCID_API SoftplusBackward : public UnaryOp<SoftplusBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::softplus`` to compute $y = \log(1 + e^x)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.softplus(a, s, dt);
    }

    // Eager-mode backward: $\partial L/\partial x = \sigma(x) \cdot
    // \partial L/\partial y$, evaluated via the storage-level sigmoid kernel
    // applied to the saved input.
    Storage grad_formula(const Storage& g);
};

// Autograd node for the Exponential Linear Unit
// $y = x$ if $x \ge 0$ else $\alpha\,(e^x - 1)$ (Clevert et al. 2015).
//
// Like Leaky ReLU, the scalar hyper-parameter $\alpha$ is captured on the
// node via a custom ``forward()`` so the backward kernel sees a concrete
// value rather than chasing it through call sites.  The piecewise backward
// itself is delegated to the backend because the conditional branch is not
// expressible through the generic storage primitives.
//
// Math
// ----
// $$y = \begin{cases} x, & x \ge 0 \\ \alpha\,(e^x - 1), & x < 0 \end{cases},
//   \qquad \frac{\partial y}{\partial x}
//   = \begin{cases} 1, & x \ge 0 \\ \alpha\,e^x, & x < 0 \end{cases}$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"elu"``, ``AmpPolicy::ForceFP32``.
// alpha_ : double
//     Negative-saturation scale captured at forward time; defaults to ``1.0``.
//
// References
// ----------
// Clevert, Unterthiner, Hochreiter, "Fast and Accurate Deep Network Learning
// by Exponential Linear Units (ELUs)", arXiv:1511.07289.
class LUCID_API EluBackward : public UnaryOp<EluBackward> {
public:
    static constexpr bool kSavesInput = true;
    double alpha_ = 1.0;
    static const OpSchema schema_v1;

    // Custom forward that captures ``alpha`` on the backward node.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input activation.
    // alpha : double
    //     Negative-saturation scale.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Forward output with autograd edges wired to ``EluBackward``.
    static TensorImplPtr forward(const TensorImplPtr& a, double alpha);

    // CPU kernel mirroring ``forward`` for the dispatcher-driven path.
    static CpuStorage
    cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt, double alpha);

    // Eager-mode backward: dispatches to ``IBackend::elu_backward`` with
    // the captured ``alpha_``.
    Storage grad_formula(const Storage& g);
};

// Autograd node for SELU (Scaled ELU, Klambauer et al. 2017).
//
// Uses the self-normalising constants $\alpha \approx 1.6732632$ and
// $\lambda \approx 1.0507010$ baked into the backend; the Python and C++
// layers do not expose them to the caller.  The backward delegates to the
// backend because the piecewise scaled-ELU formula depends on those fixed
// constants.
//
// Math
// ----
// $$y = \lambda\,\begin{cases} x, & x \ge 0 \\
//                              \alpha\,(e^x - 1), & x < 0 \end{cases}$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"selu"``, ``AmpPolicy::ForceFP32``.
//
// References
// ----------
// Klambauer et al., "Self-Normalizing Neural Networks", arXiv:1706.02515.
class LUCID_API SeluBackward : public UnaryOp<SeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::selu`` to compute the Scaled ELU
    // $y = \lambda\,x$ for $x \ge 0$ else $\lambda\,\alpha\,(e^x - 1)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.selu(a, s, dt);
    }

    // Eager-mode backward: dispatches to ``IBackend::selu_backward``.
    Storage grad_formula(const Storage& g);
};

// Autograd node for Mish $y = x \cdot \tanh(\mathrm{softplus}(x))$
// (Misra 2019).
//
// The composed function's derivative involves both $\tanh$ and its
// derivative through the softplus, so the backward is delegated to the
// backend where it runs as a single fused kernel.
//
// Math
// ----
// $$y = x\,\tanh(\mathrm{softplus}(x)),$$
// $$\frac{\partial y}{\partial x}
//   = \tanh(\mathrm{softplus}(x))
//   + x\,\sigma(x)\,(1 - \tanh^2(\mathrm{softplus}(x)))$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"mish"``, ``AmpPolicy::ForceFP32``.
//
// References
// ----------
// Misra, "Mish: A Self Regularized Non-Monotonic Activation Function",
// arXiv:1908.08681.
class LUCID_API MishBackward : public UnaryOp<MishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::mish`` to compute
    // $y = x \cdot \tanh(\mathrm{softplus}(x))$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.mish(a, s, dt);
    }

    // Eager-mode backward: dispatches to ``IBackend::mish_backward``.
    Storage grad_formula(const Storage& g);
};

// Autograd node for the Hard Sigmoid
// $y = \mathrm{clamp}((x + 3)/6, 0, 1)$ (Howard et al., MobileNetV3).
//
// The derivative is the indicator of the active linear region
// $-3 < x < 3$, scaled by $1/6$.  Backward goes through the backend to
// share the piecewise kernel with the other hard-* activations.
//
// Math
// ----
// $$y = \mathrm{clamp}\!\Bigl(\tfrac{x + 3}{6}, 0, 1\Bigr),
//   \qquad \frac{\partial y}{\partial x}
//   = \begin{cases} 1/6, & -3 < x < 3 \\ 0, & \text{otherwise} \end{cases}$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"hard_sigmoid"``, ``AmpPolicy::KeepInput``.
class LUCID_API HardSigmoidBackward : public UnaryOp<HardSigmoidBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::hard_sigmoid`` to compute
    // $y = \mathrm{clamp}((x + 3)/6, 0, 1)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.hard_sigmoid(a, s, dt);
    }

    // Eager-mode backward: dispatches to ``IBackend::hard_sigmoid_backward``.
    Storage grad_formula(const Storage& g);
};

// Autograd node for Hard Swish $y = x \cdot \mathrm{hard\_sigmoid}(x)$
// (Howard et al., MobileNetV3).
//
// The derivative is the three-region piecewise function
// $0$ for $x \le -3$, $(2x + 3)/6$ for $-3 < x < 3$, and $1$ for $x \ge 3$.
// Backward is delegated to the backend for a single fused pass.
//
// Math
// ----
// $$y = x \cdot \mathrm{hard\_sigmoid}(x),
//   \qquad \frac{\partial y}{\partial x}
//   = \begin{cases} 0, & x \le -3 \\
//                   (2x + 3)/6, & -3 < x < 3 \\
//                   1, & x \ge 3 \end{cases}$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"hard_swish"``, ``AmpPolicy::KeepInput``.
class LUCID_API HardSwishBackward : public UnaryOp<HardSwishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::hard_swish`` to compute
    // $y = x \cdot \mathrm{hard\_sigmoid}(x)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.hard_swish(a, s, dt);
    }

    // Eager-mode backward: dispatches to ``IBackend::hard_swish_backward``.
    Storage grad_formula(const Storage& g);
};

// Autograd node for ReLU6 $y = \mathrm{clamp}(x, 0, 6)$ (Krizhevsky 2010,
// MobileNet family).
//
// Gradient is the indicator of the open interval $(0, 6)$ — built directly
// as an in-range storage mask and multiplied with the upstream gradient,
// avoiding a backend round-trip for this very simple piecewise rule.
//
// Math
// ----
// $$y = \mathrm{clamp}(x, 0, 6),
//   \qquad \frac{\partial y}{\partial x}
//   = \begin{cases} 1, & 0 < x < 6 \\ 0, & \text{otherwise} \end{cases}$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"relu6"``, ``AmpPolicy::KeepInput``.
class LUCID_API Relu6Backward : public UnaryOp<Relu6Backward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::relu6`` to compute $y = \mathrm{clamp}(x, 0, 6)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.relu6(a, s, dt);
    }

    // Eager-mode backward: applies an in-range $(0, 6)$ mask to the upstream
    // gradient.
    Storage grad_formula(const Storage& g);
};

// Element-wise rectified linear unit $y = \max(0, x)$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any real dtype.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape and dtype as ``a``; autograd-aware
//     when ``a.requires_grad`` is true.
//
// See Also
// --------
// ReluBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr relu_op(const TensorImplPtr& a);

// Logistic sigmoid $y = 1 / (1 + e^{-x})$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any real dtype.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with values in $(0, 1)$.
//
// See Also
// --------
// SigmoidBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr sigmoid_op(const TensorImplPtr& a);

// SiLU / Swish activation $y = x \cdot \sigma(x)$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``.
//
// See Also
// --------
// SiluBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr silu_op(const TensorImplPtr& a);

// Tanh-approximation GeLU
// $y = 0.5\,x\,(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3)))$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``.
//
// See Also
// --------
// gelu_exact_op : Exact erf-based variant.
// GeluBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr gelu_op(const TensorImplPtr& a);

// Exact GeLU $y = 0.5\,x\,(1 + \mathrm{erf}(x/\sqrt{2}))$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``.
//
// See Also
// --------
// gelu_op : Tanh-approximation variant (default).
// GeluExactBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr gelu_exact_op(const TensorImplPtr& a);

// Leaky ReLU $y = x$ if $x > 0$ else $\text{slope} \cdot x$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// slope : double
//     Negative-side slope; ``0.01`` is the standard default.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``.
//
// See Also
// --------
// LeakyReluBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr leaky_relu_op(const TensorImplPtr& a, double slope);

// Softplus $y = \log(1 + e^x)$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with values in $(0, \infty)$.
//
// See Also
// --------
// SoftplusBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr softplus_op(const TensorImplPtr& a);

// Exponential Linear Unit
// $y = x$ if $x \ge 0$ else $\alpha\,(e^x - 1)$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// alpha : double
//     Negative-saturation scale.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``.
//
// See Also
// --------
// EluBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr elu_op(const TensorImplPtr& a, double alpha);

// Scaled ELU (SELU) with fixed self-normalising constants
// $\alpha \approx 1.6732632$ and $\lambda \approx 1.0507010$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``.
//
// See Also
// --------
// SeluBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr selu_op(const TensorImplPtr& a);

// Mish activation $y = x \cdot \tanh(\mathrm{softplus}(x))$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``.
//
// See Also
// --------
// MishBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr mish_op(const TensorImplPtr& a);

// Hard sigmoid $y = \mathrm{clamp}((x + 3)/6, 0, 1)$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with values in $[0, 1]$.
//
// See Also
// --------
// HardSigmoidBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr hard_sigmoid_op(const TensorImplPtr& a);

// Hard swish $y = x \cdot \mathrm{hard\_sigmoid}(x)$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``.
//
// See Also
// --------
// HardSwishBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr hard_swish_op(const TensorImplPtr& a);

// ReLU6 $y = \mathrm{clamp}(x, 0, 6)$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with values in $[0, 6]$.
//
// See Also
// --------
// Relu6Backward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr relu6_op(const TensorImplPtr& a);

}  // namespace lucid
