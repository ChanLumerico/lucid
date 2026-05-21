// lucid/_C/ops/ufunc/Predicate.h
//
// Floating-point predicate ops (isinf, isnan, isfinite) and nan_to_num.
// Predicates always produce a Bool output tensor; nan_to_num preserves dtype.
// None of these ops are differentiable.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Element-wise test for $\pm\infty$ — returns a boolean tensor that is
// ``True`` at every position whose input is positive or negative infinity.
//
// Not differentiable: this op does not register an autograd node, so the
// returned tensor is detached even if ``a`` has ``requires_grad=True``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any floating-point dtype.  Integer dtypes always
//     produce ``False`` at every position (no value can represent infinity).
//
// Returns
// -------
// TensorImplPtr
//     Boolean tensor with the same shape and device as ``a``.  Dtype is
//     ``Dtype::Bool`` regardless of ``a->dtype()``.
//
// Math
// ----
// $$y_i = \begin{cases} \text{True} & x_i = +\infty \text{ or } x_i = -\infty \\
//                       \text{False} & \text{otherwise} \end{cases}$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Notes
// -----
// CPU dispatch uses Accelerate's vector isinf primitive; GPU dispatch
// uses MLX's ``isinf``.  NaN inputs yield ``False`` (NaN is neither
// finite nor infinite).
//
// See Also
// --------
// :func:`isnan_op`, :func:`isfinite_op`, :func:`nan_to_num_op`.
LUCID_API TensorImplPtr isinf_op(const TensorImplPtr& a);

// Element-wise test for ``NaN`` values — returns a boolean tensor that is
// ``True`` at every position whose input is ``NaN``.
//
// Not differentiable: returns a detached boolean tensor with no autograd
// node, regardless of ``a->requires_grad()``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any floating-point dtype.  Integer inputs always
//     produce ``False`` since no integer encodes NaN.
//
// Returns
// -------
// TensorImplPtr
//     Boolean tensor with the same shape and device as ``a``; dtype is
//     ``Dtype::Bool``.
//
// Math
// ----
// $$y_i = (x_i \ne x_i)$$
//
// The IEEE-754 property "NaN does not equal itself" defines the predicate.
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Notes
// -----
// Useful as a guard inside composite reductions (``nanmean``, ``nansum``)
// where NaN-tainted slots must be replaced before accumulation.
//
// See Also
// --------
// :func:`isinf_op`, :func:`isfinite_op`, :func:`nan_to_num_op`.
LUCID_API TensorImplPtr isnan_op(const TensorImplPtr& a);

// Element-wise test for finite values — returns a boolean tensor that is
// ``True`` at every position whose input is neither ``NaN`` nor $\pm\infty$.
//
// Not differentiable: no autograd node is wired and the result is always
// detached.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any floating-point dtype.  Integer inputs always
//     produce ``True`` since every integer value is finite.
//
// Returns
// -------
// TensorImplPtr
//     Boolean tensor with the same shape and device as ``a``; dtype is
//     ``Dtype::Bool``.
//
// Math
// ----
// $$y_i = \neg \operatorname{isnan}(x_i) \wedge \neg \operatorname{isinf}(x_i)$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Notes
// -----
// Equivalent to ``~(isnan(a) | isinf(a))`` but dispatched as a single
// backend kernel for efficiency.
//
// See Also
// --------
// :func:`isnan_op`, :func:`isinf_op`.
LUCID_API TensorImplPtr isfinite_op(const TensorImplPtr& a);

// Replace ``NaN`` and $\pm\infty$ values with user-specified finite
// substitutes, preserving dtype and shape.
//
// Not differentiable: returns a detached tensor with no backward node.
// Use only in inference / data-cleaning paths.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any floating-point dtype.  Integer inputs are
//     passed through unchanged (no non-finite values to replace).
// nan_val : double, optional
//     Replacement for ``NaN`` positions.  Default ``0.0``.
// posinf_val : double, optional
//     Replacement for $+\infty$ positions.  Default is the maximum
//     finite IEEE-754 ``float32`` value (~``3.4028e+38``).
// neginf_val : double, optional
//     Replacement for $-\infty$ positions.  Default is the minimum
//     finite IEEE-754 ``float32`` value (~``-3.4028e+38``).
//
// Returns
// -------
// TensorImplPtr
//     New tensor with the same shape, dtype, and device as ``a``;
//     non-finite slots are replaced according to the three scalars.
//
// Math
// ----
// $$y_i = \begin{cases}
//   \text{nan\_val}    & x_i \text{ is NaN} \\
//   \text{posinf\_val} & x_i = +\infty \\
//   \text{neginf\_val} & x_i = -\infty \\
//   x_i                & \text{otherwise}
// \end{cases}$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Notes
// -----
// The default ``posinf_val`` / ``neginf_val`` clamp to ``float32`` range
// even when ``a`` is ``float64``.  Pass ``std::numeric_limits<double>::max``
// explicitly to preserve double-precision extremes.
//
// See Also
// --------
// :func:`isnan_op`, :func:`isinf_op`, :func:`isfinite_op`.
LUCID_API TensorImplPtr nan_to_num_op(const TensorImplPtr& a,
                                      double nan_val = 0.0,
                                      double posinf_val = 3.4028234663852886e+38,
                                      double neginf_val = -3.4028234663852886e+38);

// Full-tensor boolean OR-reduction — returns a scalar Bool tensor that is
// ``True`` iff at least one element of ``a`` is non-zero (or ``True`` for
// boolean inputs).
//
// Not differentiable: predicate output is detached.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any dtype.  Floating-point ``0.0`` counts as
//     ``False``; ``NaN`` counts as ``True`` (non-zero).
//
// Returns
// -------
// TensorImplPtr
//     Zero-dimensional ``Dtype::Bool`` tensor.  Shape is ``{}``.
//
// Math
// ----
// $$y = \bigvee_i (x_i \ne 0)$$
//
// Shape
// -----
// Output is a 0-D scalar; input may have any shape.
//
// Notes
// -----
// Reduction is over the *flattened* tensor; there is no ``axis`` argument
// at this level.  Use the Python ``any(dim=...)`` wrapper for axis-wise
// reductions.
//
// See Also
// --------
// :func:`all_op`.
LUCID_API TensorImplPtr any_op(const TensorImplPtr& a);

// Full-tensor boolean AND-reduction — returns a scalar Bool tensor that is
// ``True`` iff every element of ``a`` is non-zero.
//
// Not differentiable: predicate output is detached.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any dtype.  Floating-point ``0.0`` counts as
//     ``False``; ``NaN`` counts as ``True``.
//
// Returns
// -------
// TensorImplPtr
//     Zero-dimensional ``Dtype::Bool`` tensor.  Shape is ``{}``.
//
// Math
// ----
// $$y = \bigwedge_i (x_i \ne 0)$$
//
// Shape
// -----
// Output is a 0-D scalar; input may have any shape.
//
// Notes
// -----
// Reduction is over the *flattened* tensor.  An empty tensor returns
// ``True`` (vacuous truth) — call sites should special-case zero numel
// if a different convention is needed.
//
// See Also
// --------
// :func:`any_op`.
LUCID_API TensorImplPtr all_op(const TensorImplPtr& a);

}  // namespace lucid
