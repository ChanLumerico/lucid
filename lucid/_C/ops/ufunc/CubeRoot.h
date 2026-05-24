// lucid/_C/ops/ufunc/CubeRoot.h
//
// Autograd backward node and entry point for the cube-root operation: y = x^(1/3).

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Autograd node for element-wise cube root: $y = \sqrt[3]{x} = x^{1/3}$.
//
// Unlike ``sqrt``, cube root is real-valued for *all* real inputs —
// negative arguments produce negative outputs and the operation is
// monotone increasing.  The backward formula divides by $3 y^{2}$, which
// is the only undefined point (at $x = y = 0$).
//
// Attributes
// ----------
// kSavesInput : bool
//     ``false`` — the input is not needed for the backward pass.
// kSavesOutput : bool
//     ``true`` — the forward output $y$ is saved so the backward pass
//     can compute $1 / (3 y^{2})$ without re-invoking the cube-root
//     kernel.
// schema_v1 : OpSchema
//     Op name ``"cube_root"`` with ``AmpPolicy::ForceFP32`` because the
//     cube root is numerically unreliable in half precision (loss of
//     accuracy near zero and for very large magnitudes).
//
// Math
// ----
// Forward
//
// $$y_i = \sqrt[3]{x_i} = \operatorname{sgn}(x_i)\, |x_i|^{1/3}$$
//
// Backward
//
// $$\frac{\partial L}{\partial x_i} = \frac{1}{3 y_i^{2}} \cdot
// \frac{\partial L}{\partial y_i}$$
//
// Notes
// -----
// CPU dispatch uses Apple Accelerate's ``vvcbrtf`` (vector cube root in
// float32); GPU dispatch routes through MLX's power primitive applied
// with exponent $1/3$.  The gradient diverges at $x = 0$ — callers
// should guard against zero inputs if they intend to backpropagate.
//
// See Also
// --------
// :func:`cube_root_op`.
class LUCID_API CubeRootBackward : public UnaryOp<CubeRootBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::cube_root`` to compute $y = \sqrt[3]{x}$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cube_root(a, s, dt);
    }
    // Computes $\partial L/\partial x = g / (3 y^{2})$ from the saved
    // output $y$ — squaring $y$ is cheaper than recomputing $\sqrt[3]{x}^{2}$.
    //
    // Parameters
    // ----------
    // g : const Storage&
    //     Upstream gradient $\partial L/\partial y$.
    //
    // Returns
    // -------
    // Storage
    //     Gradient with respect to ``x``.  Undefined at slots where the
    //     saved output is zero (division by zero); callers must avoid
    //     differentiating through $x = 0$.
    Storage grad_formula(const Storage& g);
};

// Element-wise cube root — returns $\sqrt[3]{a}$ with a fully wired
// autograd node.
//
// Defined for all real inputs (positive, zero, and negative); the
// negative branch is the real cube root, not the complex principal value.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor (promoted to FP32 by the op schema).
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``; gradient is
//     $g / (3 y^{2})$.
//
// Math
// ----
// $$y_i = \sqrt[3]{x_i}, \qquad
// \frac{\partial y_i}{\partial x_i} = \frac{1}{3 y_i^{2}}$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Examples
// --------
// ``cube_root_op(-8.0) == -2.0``  (real cube root of a negative number).
// ``cube_root_op(27.0) == 3.0``.
//
// Notes
// -----
// CPU implementation uses Accelerate's ``vvcbrtf``; GPU uses MLX's
// power op with exponent $1/3$.  The gradient is singular at zero —
// avoid backpropagating through inputs that touch the origin.
//
// See Also
// --------
// :class:`CubeRootBackward`.
LUCID_API TensorImplPtr cube_root_op(const TensorImplPtr& a);

}  // namespace lucid
