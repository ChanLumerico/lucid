// lucid/_C/ops/bfunc/Maximum.h
//
// Declares MaximumBackward, the autograd node for element-wise maximum, and the
// public free function maximum_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for elementwise maximum $y = \max(a, b)$ with NumPy
// broadcasting.
//
// Saves both inputs ``a`` and ``b`` (``kSavesInputs = true`` inherited from
// :class:`BinaryKernel`) so the backward pass can reconstruct the indicator
// masks for the comparison.  The larger operand takes the gradient, a tie
// splits it in half, and a NaN on either side gives it to both operands.
//
// Math
// ----
// $$
//   y_i = \max(a_i, b_i)
// $$
// $$
//   \frac{\partial L}{\partial a_i} = w(a_i, b_i) \cdot \frac{\partial L}{\partial y_i},
//   \qquad
//   w(u, v) = \begin{cases} 1 & u > v \text{ or unordered} \\
//                           \tfrac12 & u = v \\
//                           0 & u < v \end{cases}
// $$
// $$
//   \frac{\partial L}{\partial b_i} = w(b_i, a_i) \cdot \frac{\partial L}{\partial y_i}
// $$
//
// Shape
// -----
// Inputs ``a``, ``b`` follow NumPy broadcasting rules; the output takes the
// broadcast shape.  Both gradient branches are sum-reduced back to their
// original input shapes by :func:`sum_to_shape` in the apply trampoline.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"maximum"``, version ``1``, :enum:`AmpPolicy::Promote`, deterministic.
//
// Notes
// -----
// The rule is the reference's.  Sending a tie wholly to ``a`` instead
// doubled the gradient of ``hypot`` and the complex ``abs`` at
// $|a| = |b|$, where both maximum and minimum of the same pair tie.  A
// caller that wants a clamp's rule (a tie to ``a``) uses ``clip`` or
// ``where``, not maximum.
//
// See Also
// --------
// MinimumBackward : Elementwise minimum (mirror operation).
class LUCID_API MaximumBackward : public BinaryOp<MaximumBackward> {
public:
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& a,
                                                              const TensorImplPtr& b);
    // Op registration metadata: name "maximum", schema version 1, dtype
    // promotion, deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's maximum primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.maximum(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

// Compute the elementwise maximum $y = \max(a, b)$ with broadcasting and
// autograd support.
//
// Parameters
// ----------
// a : TensorImplPtr
//     First operand.
// b : TensorImplPtr
//     Second operand.  Broadcast-compatible with ``a``.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``broadcast(a.shape, b.shape)`` holding the elementwise
//     maximum.
//
// Notes
// -----
// At ties ($a_i = b_i$) each operand receives half the gradient.  When
// grad-tracking is on, registers a :class:`MaximumBackward` node which saves
// both inputs.
//
// Examples
// --------
// >>> auto y = maximum_op(a, b);  // y[i] = max(a[i], b[i]) with broadcasting
LUCID_API TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
