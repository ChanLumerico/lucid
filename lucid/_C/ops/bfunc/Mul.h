// lucid/_C/ops/bfunc/Mul.h
//
// Declares MulBackward, the autograd node for element-wise tensor
// multiplication, and the public free function mul_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise multiplication $y = a \cdot b$ with NumPy
// broadcasting.
//
// Saves both forward inputs because the gradient w.r.t. each input depends
// on the *other* input's value (product rule).  Broadcast-reduces each
// branch back to the original input shape via :func:`sum_to_shape` inside
// :func:`BinaryKernel::apply`.
//
// Math
// ----
// $$
//   y = a \cdot b, \qquad
//   \frac{\partial L}{\partial a} = b \cdot \frac{\partial L}{\partial y}, \qquad
//   \frac{\partial L}{\partial b} = a \cdot \frac{\partial L}{\partial y}
// $$
//
// Shape
// -----
// Inputs follow NumPy broadcasting; output has the broadcasted shape.  Each
// gradient is then reduced to the original input's shape.
//
// Attributes
// ----------
// kSavesInputs : bool
//     ``true`` (inherited default from BinaryKernel) — both forward inputs
//     are retained for the backward pass.
// schema_v1 : OpSchema
//     ``"mul"``, schema version 1, ``AmpPolicy::Promote``, deterministic.
//
// Notes
// -----
// CPU dispatch uses Accelerate ``vDSP_vmul`` / ``vDSP_vmulD`` for F32 / F64.
// GPU dispatch uses MLX's broadcast-aware ``multiply`` primitive.
//
// See Also
// --------
// AddBackward, SubBackward, DivBackward
class LUCID_API MulBackward : public BinaryOp<MulBackward> {
public:
    // Op registration metadata.
    //
    // Attributes
    // ----------
    // name : const char*
    //     ``"mul"``.
    // version : int
    //     ``1``.
    // amp_policy : AmpPolicy
    //     ``Promote``.
    // deterministic : bool
    //     ``true``.
    static const OpSchema schema_v1;

    // Forward dispatch hook called by BinaryKernel::forward.
    //
    // Parameters
    // ----------
    // be : backend::IBackend&
    //     Active backend (CPU Accelerate or GPU MLX).
    // a, b : const Storage&
    //     Operands already broadcast-expanded to ``shape``.
    // shape : const Shape&
    //     Broadcasted output shape.
    // dt : Dtype
    //     Promoted output dtype.
    //
    // Returns
    // -------
    // Storage
    //     Output storage holding $a \cdot b$.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.mul(a, b, shape, dt);
    }

    // Compute the storage-level gradients for both operands (product rule).
    //
    // Parameters
    // ----------
    // grad_out : const Storage&
    //     Upstream gradient at the broadcasted output shape.
    //
    // Returns
    // -------
    // std::pair<Storage, Storage>
    //     ``(dA, dB)`` where ``dA = grad_out * b_broadcast`` and
    //     ``dB = grad_out * a_broadcast``.  Both still at output shape;
    //     :func:`BinaryKernel::apply` performs the broadcast reduction.
    //
    // Math
    // ----
    // $$
    //   \mathrm{dA} = b \odot \frac{\partial L}{\partial y}, \qquad
    //   \mathrm{dB} = a \odot \frac{\partial L}{\partial y}
    // $$
    //
    // Notes
    // -----
    // Uses ``saved_input_broadcasted(k)`` to expand each saved input to the
    // broadcast output shape before the element-wise multiplication.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);

    // Graph-mode (TensorImpl) variant of the gradient.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     Upstream gradient node.
    // a, b : const TensorImplPtr&
    //     Forward inputs, already broadcast-expanded to ``out_shape_`` by
    //     BinaryKernel.
    //
    // Returns
    // -------
    // std::pair<TensorImplPtr, TensorImplPtr>
    //     ``(grad_out * b, grad_out * a)``.
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& a,
                                                              const TensorImplPtr& b);
};

// Public entry point — element-wise multiplication with autograd support.
//
// Parameters
// ----------
// a, b : const TensorImplPtr&
//     Operands of any rank.  Shapes must be NumPy-broadcast-compatible.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of shape ``broadcast_shapes(a.shape, b.shape)`` and
//     promoted dtype.  Participates in autograd via :class:`MulBackward`.
//
// Math
// ----
// $$ y = a \cdot b $$
//
// Raises
// ------
// LucidError
//     If ``a`` and ``b`` have incompatible shapes under NumPy broadcasting
//     rules, or if their devices differ.
//
// Examples
// --------
// >>> auto c = mul_op(a, b);    // c = a * b (Hadamard / elementwise)
//
// See Also
// --------
// add_op, sub_op, div_op
LUCID_API TensorImplPtr mul_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
