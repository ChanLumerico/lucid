// lucid/_C/ops/bfunc/Minimum.h
//
// Declares MinimumBackward, the autograd node for element-wise minimum, and the
// public free function minimum_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for elementwise minimum $y = \min(a, b)$ with NumPy
// broadcasting.
//
// Saves both inputs ``a`` and ``b`` (``kSavesInputs = true`` inherited from
// :class:`BinaryKernel`) so the backward pass can reconstruct the indicator
// masks for the comparison.  Gradient flows only through whichever operand
// "won" the elementwise comparison; ties are broken in favour of ``a``
// (gradient assigned entirely to ``a`` when $a_i = b_i$).
//
// Math
// ----
// $$
//   y_i = \min(a_i, b_i)
// $$
// $$
//   \frac{\partial L}{\partial a_i} = \mathbb{1}\{a_i \leq b_i\} \cdot \frac{\partial L}{\partial y_i}
// $$
// $$
//   \frac{\partial L}{\partial b_i} = \mathbb{1}\{a_i > b_i\} \cdot \frac{\partial L}{\partial y_i}
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
//     ``"minimum"``, version ``1``, :enum:`AmpPolicy::Promote`, deterministic.
//
// Notes
// -----
// Internally the masks are computed with the operand order **reversed**
// relative to :class:`MaximumBackward`: ``ge_mask(b, a)`` and ``lt_mask(b, a)``
// instead of ``ge_mask(a, b)`` / ``lt_mask(a, b)``.  This yields the same tie
// resolution (all gradient to ``a``) while flipping the comparison direction.
// The two masks partition the index set, so their sum is exactly $1$ at every
// position.
//
// See Also
// --------
// MaximumBackward : Elementwise maximum (mirror operation).
class LUCID_API MinimumBackward : public BinaryOp<MinimumBackward> {
public:
    // Op registration metadata: name "minimum", schema version 1, dtype
    // promotion, deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's minimum primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.minimum(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

// Compute the elementwise minimum $y = \min(a, b)$ with broadcasting and
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
//     minimum.
//
// Notes
// -----
// At ties ($a_i = b_i$) the gradient is routed entirely to ``a``.  When
// grad-tracking is on, registers a :class:`MinimumBackward` node which saves
// both inputs.
//
// Examples
// --------
// >>> auto y = minimum_op(a, b);  // y[i] = min(a[i], b[i]) with broadcasting
LUCID_API TensorImplPtr minimum_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
