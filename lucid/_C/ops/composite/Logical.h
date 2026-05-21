// lucid/_C/ops/composite/Logical.h
//
// Boolean AND / OR / XOR / NOT.  Inputs are interpreted as truthy when
// non-zero; outputs are bool tensors.  Composes ``not_equal`` (to coerce to
// bool) with the existing ``bitwise_*`` kernels.  Not differentiable.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Elementwise logical AND: $y_i = [a_i \ne 0] \land [b_i \ne 0]$.
//
// Composite over :func:`not_equal_op` (to coerce each operand to a bool
// mask) + :func:`bitwise_and_op`.  Not differentiable — the output is
// integral and ``bitwise_*`` has no backward.
//
// Math
// ----
// $$
//   y_i = \mathbb{1}\!\left[ a_i \ne 0 \right] \cdot
//         \mathbb{1}\!\left[ b_i \ne 0 \right]
// $$
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of broadcastable shapes and arbitrary numeric dtype.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor on the broadcast shape of ``a`` and ``b``.
//
// See Also
// --------
// :func:`bitwise_and_op` — bit-level variant on integer inputs.
LUCID_API TensorImplPtr logical_and_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Elementwise logical OR: $y_i = [a_i \ne 0] \lor [b_i \ne 0]$.
//
// Composite over :func:`not_equal_op` + :func:`bitwise_or_op`.  Not
// differentiable.
//
// Math
// ----
// $$
//   y_i = \mathbb{1}\!\left[ a_i \ne 0 \,\lor\, b_i \ne 0 \right]
// $$
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of broadcastable shapes.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor on the broadcast shape.
//
// See Also
// --------
// :func:`bitwise_or_op` — bit-level variant on integer inputs.
LUCID_API TensorImplPtr logical_or_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Elementwise logical XOR: $y_i = [a_i \ne 0] \oplus [b_i \ne 0]$.
//
// Composite over :func:`not_equal_op` + :func:`bitwise_xor_op`.  Not
// differentiable.
//
// Math
// ----
// $$
//   y_i = \mathbb{1}\!\left[ (a_i \ne 0) \ne (b_i \ne 0) \right]
// $$
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of broadcastable shapes.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor on the broadcast shape.
//
// See Also
// --------
// :func:`bitwise_xor_op` — bit-level variant.
LUCID_API TensorImplPtr logical_xor_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Elementwise logical NOT: $y_i = [a_i = 0]$.
//
// Composite over :func:`equal_op` against zero — strictly cheaper than the
// double-flip ``bitwise_not(not_equal(a, 0))`` pattern.  Not differentiable.
//
// Math
// ----
// $$
//   y_i = \mathbb{1}\!\left[ a_i = 0 \right]
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Operand of arbitrary numeric dtype.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor of the same shape as ``a``.
//
// See Also
// --------
// :func:`logical_and_op`, :func:`logical_or_op`.
LUCID_API TensorImplPtr logical_not_op(const TensorImplPtr& a);

}  // namespace lucid
