// lucid/_C/ops/ufunc/Hyperbolic.h
//
// Autograd backward nodes and entry points for the hyperbolic family:
// sinh, cosh, tanh.  On CPU the backend routes to vForce (vvsinhf, vvcoshf,
// vvtanhf).  All ops use AmpPolicy::Promote.

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Autograd node for element-wise hyperbolic sine $y = \sinh(x)$.
//
// Saves the input ``x`` so the backward pass can evaluate $\cosh(x)$ and
// scale the upstream gradient by it.  Defined on all of $\mathbb{R}$ with
// the identity $\sinh(x) = (e^x - e^{-x})/2$.
//
// Math
// ----
// $$y = \sinh(x), \qquad
// \frac{\partial y}{\partial x} = \cosh(x), \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \cosh(x)\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"sinh"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvsinhf``/``vvsinh`` (CPU) / MLX ``sinh`` (GPU).
// Output magnitude grows exponentially; large-magnitude inputs can overflow
// to ``+inf`` / ``-inf``.
class LUCID_API SinhBackward : public UnaryOp<SinhBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::sinh`` to compute $y = \sinh x$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sinh(a, s, dt);
    }
    // Backward — $\partial y/\partial x = \cosh x$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
};

// Autograd node for element-wise hyperbolic cosine $y = \cosh(x)$.
//
// Saves the input ``x`` so the backward pass can evaluate $\sinh(x)$.
// Defined on all of $\mathbb{R}$ with the identity
// $\cosh(x) = (e^x + e^{-x})/2 \geq 1$.
//
// Math
// ----
// $$y = \cosh(x), \qquad
// \frac{\partial y}{\partial x} = \sinh(x), \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \sinh(x)\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"cosh"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvcoshf``/``vvcosh`` (CPU) / MLX ``cosh`` (GPU).
// Output magnitude grows exponentially.
class LUCID_API CoshBackward : public UnaryOp<CoshBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::cosh`` to compute $y = \cosh x$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cosh(a, s, dt);
    }
    // Backward — $\partial y/\partial x = \sinh x$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
};

// Autograd node for element-wise hyperbolic tangent $y = \tanh(x)$.
//
// Saves the *output* $y$ rather than the input because the backward
// formula $1 - y^2$ is cheaper than re-evaluating $\tanh(x)$ and avoids a
// second ``vvtanhf`` pass.  ``kSavesInput = false`` opts out of the
// default input-save behaviour and ``kSavesOutput = true`` registers the
// output instead.  Range is $(-1, 1)$ on all of $\mathbb{R}$.
//
// Math
// ----
// $$y = \tanh(x), \qquad
// \frac{\partial y}{\partial x} = 1 - \tanh^2(x) = 1 - y^2, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// (1 - y^2)\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"tanh"`` with ``AmpPolicy::Promote``.
// kSavesInput : bool
//     ``false``.
// kSavesOutput : bool
//     ``true`` — the saved tensor is $y = \tanh(x)$.
//
// Notes
// -----
// Dispatch: Accelerate ``vvtanhf``/``vvtanh`` (CPU) / MLX ``tanh`` (GPU).
class LUCID_API TanhBackward : public UnaryOp<TanhBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::tanh`` to compute $y = \tanh x$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.tanh(a, shape, dt);
    }
    // Backward — $\partial y/\partial x = 1 - y^2$ using the saved output, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\partial x = (1 - \mathrm{out}^2) \cdot g$ where
    // $\mathrm{out} = \tanh x$; kept inside the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr& out);
};

// Compute $y = \sinh(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`SinhBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.  Large-magnitude inputs
//     may overflow.
//
// See Also
// --------
// :class:`SinhBackward` — backward node.
LUCID_API TensorImplPtr sinh_op(const TensorImplPtr& a);

// Compute $y = \cosh(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`CoshBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, values $\geq 1$.
//
// See Also
// --------
// :class:`CoshBackward` — backward node.
LUCID_API TensorImplPtr cosh_op(const TensorImplPtr& a);

// Compute $y = \tanh(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`TanhBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, values in $(-1, 1)$.
//
// See Also
// --------
// :class:`TanhBackward` — backward node.
LUCID_API TensorImplPtr tanh_op(const TensorImplPtr& a);

}  // namespace lucid
