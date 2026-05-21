// lucid/_C/ops/utils/Sort.h
//
// Declares sorting, index-finding, and uniqueness operations.  sort_op is
// differentiable for floating-point inputs; the remaining ops (argsort,
// argmax, argmin, nonzero, unique, topk) return integer indices or
// non-differentiable results.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Sort a tensor along a single axis in ascending order.
//
// Returns the sorted values; the permutation indices are computed
// internally and consumed by the backward pass but are not exposed here.
// See :func:`argsort_op` for the indices alone, or :func:`topk_op` for a
// (values, indices) pair.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axis : int
//     Axis along which to sort.
//
// Returns
// -------
// TensorImplPtr
//     Sorted tensor of the same shape and dtype as ``a``.
//
// Math
// ----
// Let $\sigma$ be the permutation that sorts ``a`` along ``axis``.  Then
// $$y_i = a_{\sigma(i)}, \qquad
// \frac{\partial L}{\partial a_i} =
//     \left(\frac{\partial L}{\partial y}\right)_{\sigma^{-1}(i)}.$$
//
// Notes
// -----
// Differentiable for F32 / F64 inputs only via ``IndexScatterBackward``,
// which scatters the incoming gradient back to the original positions.
// Integer dtypes return a detached output.
LUCID_API TensorImplPtr sort_op(const TensorImplPtr& a, int axis);

// Return the indices that would sort ``a`` along ``axis``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axis : int
//     Axis along which to compute the sort permutation.
//
// Returns
// -------
// TensorImplPtr
//     Integer tensor of the same shape as ``a`` and dtype I32, such that
//     ``a.take_along_axis(out, axis)`` equals ``sort_op(a, axis)``.
//
// Notes
// -----
// Non-differentiable (indices are discontinuous in the input values).
LUCID_API TensorImplPtr argsort_op(const TensorImplPtr& a, int axis);

// Index of the maximum value along an axis.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axis : int
//     Axis to reduce.
// keepdims : bool
//     If true, retain the reduced axis as size 1 in the output shape;
//     otherwise remove it.
//
// Returns
// -------
// TensorImplPtr
//     Integer tensor of dtype I64 containing the argmax indices.
//
// Notes
// -----
// Ties are broken by returning the lowest index.  Non-differentiable.
LUCID_API TensorImplPtr argmax_op(const TensorImplPtr& a, int axis, bool keepdims);

// Index of the minimum value along an axis.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axis : int
//     Axis to reduce.
// keepdims : bool
//     If true, retain the reduced axis as size 1; otherwise remove it.
//
// Returns
// -------
// TensorImplPtr
//     Integer tensor of dtype I64 containing the argmin indices.
//
// Notes
// -----
// Ties are broken by returning the lowest index.  Non-differentiable.
//
// See Also
// --------
// argmax_op : Maximum-index counterpart.
LUCID_API TensorImplPtr argmin_op(const TensorImplPtr& a, int axis, bool keepdims);

// Coordinates of all non-zero elements.
//
// Returns a 2-D tensor of shape ``(count, ndim)`` listing the multi-axis
// coordinates of every non-zero element of ``a`` in row-major order.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any rank and dtype.
//
// Returns
// -------
// TensorImplPtr
//     Integer tensor of shape ``(count, a.ndim)`` and dtype I64.
//     ``count`` is the number of non-zero entries.
//
// Notes
// -----
// The output shape depends on the input values, so this op always
// materialises the result on the CPU stream and copies it to the
// requested device on demand.  Non-differentiable.
LUCID_API TensorImplPtr nonzero_op(const TensorImplPtr& a);

// Unique sorted elements of a tensor.
//
// Flattens ``a``, removes duplicates, and returns the result sorted in
// ascending order as a 1-D tensor.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  Supported dtypes: F32, F64, I32, I64.
//
// Returns
// -------
// TensorImplPtr
//     1-D tensor of the same dtype as ``a`` containing the unique
//     elements in sorted order.  Always resides on ``Device::CPU``
//     because the output length is data-dependent.
//
// Notes
// -----
// Non-differentiable.
LUCID_API TensorImplPtr unique_op(const TensorImplPtr& a);

// Top-k values and their indices along an axis.
//
// Selects the ``k`` largest elements along ``axis`` and returns them
// together with their positions in the original tensor.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// k : int64
//     Number of top elements to return.  Must satisfy
//     ``0 < k <= a.shape[axis]``.
// axis : int
//     Axis along which to select.
//
// Returns
// -------
// vector<TensorImplPtr>
//     ``[values, indices]`` where both have shape equal to ``a.shape``
//     with ``axis`` reduced to ``k``.  ``values`` has the dtype of ``a``;
//     ``indices`` has dtype I32.
//
// Notes
// -----
// ``values`` is differentiable for F32 / F64 inputs via
// ``IndexScatterBackward``, which scatters the incoming gradient back to
// the selected positions in the original tensor.  ``indices`` is never
// differentiable.
//
// See Also
// --------
// sort_op : Full sort (no truncation).
// argsort_op : Sort indices only.
LUCID_API std::vector<TensorImplPtr> topk_op(const TensorImplPtr& a, std::int64_t k, int axis);

}  // namespace lucid
