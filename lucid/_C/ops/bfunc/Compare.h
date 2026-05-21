// lucid/_C/ops/bfunc/Compare.h
//
// Declares the six element-wise comparison operators: ==, !=, >, >=, <, <=.
// All comparisons require equal-shape operands (no broadcasting) and produce a
// Bool tensor.  None of these operations support autograd because the
// comparison function is not differentiable.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Element-wise equality comparison.
//
// Computes ``out[i] = (a[i] == b[i])`` for every element and returns the
// result as a Bool tensor.  The comparison is exact, so for floating-point
// inputs it follows IEEE 754 equality (``NaN == NaN`` is false).
//
// Non-differentiable: the comparison function has zero gradient almost
// everywhere and undefined gradient on the decision boundary, so no
// autograd node is registered and the output never carries grad.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands.  Must have identical shape, dtype, and device — this op
//     does not broadcast.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor with the same shape as ``a`` and ``b``.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = \mathbf{1}\!\left[a[i] = b[i]\right].
// $$
//
// Notes
// -----
// * Output dtype is always ``Bool`` regardless of input dtype.
// * No broadcasting — callers must align shapes upstream.
//
// Examples
// --------
//     auto a = tensor({1, 2, 3});
//     auto b = tensor({1, 0, 3});
//     auto m = equal_op(a, b);  // Bool[true, false, true]
//
// Raises
// ------
// LucidError
//     If the operand shapes, dtypes, or devices differ.
//
// References
// ----------
// NumPy ``numpy.equal``.
//
// See Also
// --------
// not_equal_op, less_op, greater_op
LUCID_API TensorImplPtr equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise inequality comparison.
//
// Computes ``out[i] = (a[i] != b[i])`` and returns the result as a Bool
// tensor.  Logical complement of ``equal_op``.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of identical shape, dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor with the same shape as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = \mathbf{1}\!\left[a[i] \neq b[i]\right].
// $$
//
// Raises
// ------
// LucidError
//     If the operand shapes, dtypes, or devices differ.
//
// References
// ----------
// NumPy ``numpy.not_equal``.
//
// See Also
// --------
// equal_op
LUCID_API TensorImplPtr not_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise strict greater-than comparison.
//
// Computes ``out[i] = (a[i] > b[i])`` and returns the result as a Bool
// tensor.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of identical shape, dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor with the same shape as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = \mathbf{1}\!\left[a[i] > b[i]\right].
// $$
//
// Raises
// ------
// LucidError
//     If the operand shapes, dtypes, or devices differ.
//
// References
// ----------
// NumPy ``numpy.greater``.
//
// See Also
// --------
// greater_equal_op, less_op
LUCID_API TensorImplPtr greater_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise greater-than-or-equal comparison.
//
// Computes ``out[i] = (a[i] >= b[i])`` and returns the result as a Bool
// tensor.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of identical shape, dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor with the same shape as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = \mathbf{1}\!\left[a[i] \geq b[i]\right].
// $$
//
// Raises
// ------
// LucidError
//     If the operand shapes, dtypes, or devices differ.
//
// References
// ----------
// NumPy ``numpy.greater_equal``.
//
// See Also
// --------
// greater_op, less_equal_op
LUCID_API TensorImplPtr greater_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise strict less-than comparison.
//
// Computes ``out[i] = (a[i] < b[i])`` and returns the result as a Bool
// tensor.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of identical shape, dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor with the same shape as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = \mathbf{1}\!\left[a[i] < b[i]\right].
// $$
//
// Raises
// ------
// LucidError
//     If the operand shapes, dtypes, or devices differ.
//
// References
// ----------
// NumPy ``numpy.less``.
//
// See Also
// --------
// less_equal_op, greater_op
LUCID_API TensorImplPtr less_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise less-than-or-equal comparison.
//
// Computes ``out[i] = (a[i] <= b[i])`` and returns the result as a Bool
// tensor.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of identical shape, dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor with the same shape as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = \mathbf{1}\!\left[a[i] \leq b[i]\right].
// $$
//
// Raises
// ------
// LucidError
//     If the operand shapes, dtypes, or devices differ.
//
// References
// ----------
// NumPy ``numpy.less_equal``.
//
// See Also
// --------
// less_op, greater_equal_op
LUCID_API TensorImplPtr less_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
