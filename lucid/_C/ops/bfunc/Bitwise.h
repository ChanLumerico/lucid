// lucid/_C/ops/bfunc/Bitwise.h
//
// Declares the three element-wise bitwise binary operators: AND, OR, XOR.
// These operations require equal-shape operands with integer or Bool dtype.
// No gradient is defined because bitwise operations are not differentiable.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Element-wise bitwise AND.
//
// Computes ``out[i] = a[i] & b[i]`` for every element.  Defined on
// integer (``I8``/``I16``/``I32``/``I64``) and ``Bool`` dtypes; on Bool
// inputs the result is equivalent to logical AND.
//
// Non-differentiable: bitwise operations have no meaningful gradient on
// integer manifolds, so no autograd node is registered.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of identical shape, integer-or-Bool dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = a[i] \,\wedge_{\text{bit}}\, b[i].
// $$
//
// Notes
// -----
// * No broadcasting — shapes must match exactly.
// * Floating-point inputs are rejected at the dispatch layer.
//
// Examples
// --------
//     auto a = tensor({0b1100, 0b1010}, Dtype::I32);
//     auto b = tensor({0b1010, 0b0110}, Dtype::I32);
//     auto c = bitwise_and_op(a, b);  // {0b1000, 0b0010}
//
// Raises
// ------
// LucidError
//     If shapes/devices differ, or if the input dtype is not integer or
//     Bool.
//
// References
// ----------
// NumPy ``numpy.bitwise_and``.
//
// See Also
// --------
// bitwise_or_op, bitwise_xor_op
LUCID_API TensorImplPtr bitwise_and_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise bitwise OR.
//
// Computes ``out[i] = a[i] | b[i]`` for every element.  Defined on
// integer and ``Bool`` dtypes; on Bool inputs this is equivalent to
// logical OR.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of identical shape, integer-or-Bool dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = a[i] \,\vee_{\text{bit}}\, b[i].
// $$
//
// Raises
// ------
// LucidError
//     If shapes/devices differ, or if the input dtype is not integer or
//     Bool.
//
// References
// ----------
// NumPy ``numpy.bitwise_or``.
//
// See Also
// --------
// bitwise_and_op, bitwise_xor_op
LUCID_API TensorImplPtr bitwise_or_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise bitwise XOR.
//
// Computes ``out[i] = a[i] ^ b[i]`` for every element.  Defined on
// integer and ``Bool`` dtypes; on Bool inputs this is equivalent to
// logical exclusive-or.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands of identical shape, integer-or-Bool dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = a[i] \,\oplus_{\text{bit}}\, b[i].
// $$
//
// Raises
// ------
// LucidError
//     If shapes/devices differ, or if the input dtype is not integer or
//     Bool.
//
// References
// ----------
// NumPy ``numpy.bitwise_xor``.
//
// See Also
// --------
// bitwise_and_op, bitwise_or_op
LUCID_API TensorImplPtr bitwise_xor_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise left shift.
//
// Computes ``out[i] = a[i] << b[i]`` for every element.  Defined only on
// signed integer dtypes (``I8``/``I16``/``I32``/``I64``); ``Bool`` is
// rejected because shifting a bit has no defined meaning.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Value to shift.  Signed integer dtype.
// b : TensorImplPtr
//     Shift amount per element.  Must match ``a`` in shape, dtype, and
//     device.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = a[i] \cdot 2^{b[i]}.
// $$
// Out-of-range shift amounts follow the underlying C++ ``operator<<``
// semantics for signed integers; callers should clamp ``b`` to
// ``[0, bitwidth)`` to stay portable.
//
// Notes
// -----
// * Floating-point and Bool inputs are rejected at the dispatch layer.
// * No broadcasting — shapes must match exactly.
//
// Raises
// ------
// LucidError
//     If shapes/devices differ, or if the input dtype is not one of
//     ``I8``/``I16``/``I32``/``I64``.
//
// References
// ----------
// NumPy ``numpy.left_shift``.
//
// See Also
// --------
// bitwise_right_shift_op
LUCID_API TensorImplPtr bitwise_left_shift_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise right shift.
//
// Computes ``out[i] = a[i] >> b[i]`` for every element.  Defined only on
// signed integer dtypes (``I8``/``I16``/``I32``/``I64``); ``Bool`` is
// rejected.  On signed inputs the shift is arithmetic (sign-extending),
// matching the C++ ``operator>>`` convention used by the backend.
//
// Non-differentiable: no autograd node is registered.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Value to shift.  Signed integer dtype.
// b : TensorImplPtr
//     Shift amount per element.  Must match ``a`` in shape, dtype, and
//     device.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as the inputs.
//
// Math
// ----
// $$
//   \mathrm{out}[i] = \left\lfloor a[i] / 2^{b[i]} \right\rfloor.
// $$
// For negative ``a[i]`` the result is the floor of the true division,
// i.e. the standard arithmetic-shift result.
//
// Notes
// -----
// * Floating-point and Bool inputs are rejected at the dispatch layer.
// * No broadcasting — shapes must match exactly.
//
// Raises
// ------
// LucidError
//     If shapes/devices differ, or if the input dtype is not one of
//     ``I8``/``I16``/``I32``/``I64``.
//
// References
// ----------
// NumPy ``numpy.right_shift``.
//
// See Also
// --------
// bitwise_left_shift_op
LUCID_API TensorImplPtr bitwise_right_shift_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
