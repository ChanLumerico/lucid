// lucid/_C/ops/composite/Indexing.h
//
// Indexing convenience ops layered on top of ``gather``, ``scatter_add``,
// ``split_at``, and ``sort``.  Each entry function below is a thin shape
// shim — the underlying primitives carry the gradient.
//
//   take(a, indices)            — gather over a flattened ``a``
//   index_select(a, dim, idx)   — gather with a 1-D index broadcast to ``a``'s rank
//   narrow(a, dim, start, len)  — slice a contiguous window via ``split_at``
//   scatter(base, dim, idx, src)— overwrite-semantics scatter via ``scatter_add``
//   kthvalue(a, k, dim, keepdim)— sort + gather to pluck the k-th element

#pragma once

#include <cstdint>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Index ``a`` with a flat index list: $y_j = \mathrm{flatten}(a)_{i_j}$.
//
// Composite over :func:`reshape_op` + :func:`gather_op`.  The input is
// flattened to a 1-D view first so the index dimension is unambiguous;
// gradient flows back through ``GatherBackward`` (scatter-add) and
// ``ReshapeBackward``.
//
// Math
// ----
// $$
//   y_j = a_{\mathrm{unravel}(i_j)}
// $$
// where $\mathrm{unravel}$ maps the linear index into ``a``'s row-major
// coordinates.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor of arbitrary shape.
// indices : TensorImplPtr
//     Integer tensor (``int32`` or ``int64``) of arbitrary shape; each
//     element is a flat-index into ``a``.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape as ``indices`` and dtype of ``a``.
//
// Raises
// ------
// Failure
//     If either input is null, or if ``indices`` is not an integer dtype.
//
// Notes
// -----
// Out-of-bounds indices are forwarded verbatim to ``gather_op`` — behaviour
// matches the underlying primitive's bounds policy.
//
// See Also
// --------
// :func:`index_select_op` — keeps the original rank instead of flattening.
LUCID_API TensorImplPtr take_op(const TensorImplPtr& a, const TensorImplPtr& indices);

// Pick ``indices.size`` slices along ``dim`` of ``a``.
//
// Composite over :func:`reshape_op` + :func:`expand_op` + :func:`gather_op`.
// The 1-D index list is reshaped to rank ``a.ndim`` (size $k$ along ``dim``,
// 1 elsewhere) and expanded to the source shape so ``gather_op``'s
// same-rank contract holds.
//
// Math
// ----
// For each output coordinate $(\ldots, j, \ldots)$ along ``dim``:
// $$
//   y_{\ldots, j, \ldots} = a_{\ldots, \mathrm{idx}_j, \ldots}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor of rank $\ge 1$.
// dim : int
//     Axis to index along.  Negative values wrap modulo ``a.ndim``.
// indices : TensorImplPtr
//     1-D integer tensor (``int32`` or ``int64``) of length $k$.
//
// Returns
// -------
// TensorImplPtr
//     Tensor with the same shape as ``a`` except size $k$ along ``dim``.
//
// Raises
// ------
// IndexError
//     If ``dim`` is out of range.
// Failure
//     If ``indices`` is not 1-D or not an integer dtype.
//
// See Also
// --------
// :func:`take_op` — flat indexing; :func:`narrow_op` — contiguous window.
LUCID_API TensorImplPtr index_select_op(const TensorImplPtr& a,
                                        int dim,
                                        const TensorImplPtr& indices);

// Slice a contiguous window ``[start, start + length)`` along ``dim``.
//
// Composite over :func:`split_at_op`.  Returns the input unchanged when
// the window covers the full axis (zero-cost fast path).  Gradient flows
// through ``SplitSliceBackward``.
//
// Math
// ----
// $$
//   y_{\ldots, j, \ldots} = a_{\ldots, j + \mathrm{start}, \ldots},
//   \qquad 0 \le j < \mathrm{length}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor of rank $\ge 1$.
// dim : int
//     Axis to slice along.  Negative values wrap modulo ``a.ndim``.
// start : int64
//     Inclusive start index along ``dim``.  Must be non-negative.
// length : int64
//     Window length.  Must satisfy ``start + length <= a.shape[dim]``.
//
// Returns
// -------
// TensorImplPtr
//     Tensor with the same shape as ``a`` except size ``length`` along
//     ``dim``.  When the window covers the full axis, the input is
//     returned by identity.
//
// Raises
// ------
// IndexError
//     If ``dim`` is out of range or the window falls outside ``a``.
//
// See Also
// --------
// :func:`split_at_op` — underlying primitive.
LUCID_API TensorImplPtr narrow_op(const TensorImplPtr& a,
                                  int dim,
                                  std::int64_t start,
                                  std::int64_t length);

// Overwrite-semantics scatter: $\mathrm{out}[..., \mathrm{idx}, ...] = \mathrm{src}$.
//
// Composite over :func:`gather_op` + :func:`sub_op` + :func:`scatter_add_op`.
// The overwrite is encoded as a scatter-add of the *delta*
// $\mathrm{src} - \mathrm{base}[\mathrm{idx}]$, so the result is identical
// to assignment but reuses the existing additive scatter kernel.  Gradient
// flows through ``ScatterAddBackward``, ``SubBackward``, and
// ``GatherBackward``.
//
// Math
// ----
// $$
//   y_i = \begin{cases}
//     \mathrm{src}_j & \text{if } i = \mathrm{idx}_j \text{ for some } j \\
//     \mathrm{base}_i & \text{otherwise}
//   \end{cases}
// $$
//
// Parameters
// ----------
// base : TensorImplPtr
//     Tensor to scatter into.  Provides the output shape and dtype.
// dim : int
//     Axis along which ``indices`` selects.
// indices : TensorImplPtr
//     Integer tensor (``int32`` or ``int64``) with the same shape as
//     ``src``.
// src : TensorImplPtr
//     Source values written into ``base`` at the positions given by
//     ``indices``.
//
// Returns
// -------
// TensorImplPtr
//     A new tensor with the same shape and dtype as ``base``.
//
// Raises
// ------
// Failure
//     If any input is null or ``indices`` is not an integer dtype.
// IndexError
//     If ``dim`` is out of range.
//
// Notes
// -----
// Duplicate indices yield undefined ordering — the *last write wins*
// guarantee of true scatter is not preserved here because the underlying
// primitive is additive.  Use disjoint ``indices`` for deterministic
// behaviour.
//
// See Also
// --------
// :func:`scatter_add_op` — additive variant; canonical primitive.
LUCID_API TensorImplPtr scatter_op(const TensorImplPtr& base,
                                   int dim,
                                   const TensorImplPtr& indices,
                                   const TensorImplPtr& src);

// Pluck the $k$-th smallest element along ``dim``.
//
// Composite over :func:`sort_op` + :func:`gather_op` (+ optional
// :func:`squeeze_op`).  After sorting ``a`` ascending along ``dim``, we
// gather the slice at position ``k - 1`` to obtain the $k$-th smallest
// value.  Not differentiable — the integer-valued index breaks the chain;
// gradient is conventionally treated as zero.
//
// Math
// ----
// Let $a^{\uparrow}$ be ``a`` sorted ascending along ``dim``.  Then
// $$
//   y_{\ldots} = a^{\uparrow}_{\ldots, k - 1, \ldots}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor of rank $\ge 1$.
// k : int64
//     1-based rank of the value to return.  Must satisfy
//     ``1 <= k <= a.shape[dim]``.
// dim : int
//     Axis to reduce along.  Negative values wrap modulo ``a.ndim``.
// keepdim : bool
//     If ``true``, retain a size-1 axis at ``dim``; otherwise the axis is
//     squeezed off.
//
// Returns
// -------
// TensorImplPtr
//     Tensor with the same shape as ``a`` except size 1 (or removed) at
//     ``dim``.  Dtype matches ``a``.
//
// Raises
// ------
// Failure
//     If ``a`` is null or ``k`` is outside ``[1, a.shape[dim]]``.
// IndexError
//     If ``dim`` is out of range.
//
// Notes
// -----
// Unlike the reference framework's API, this entry point returns only the
// value — the matching index can be recovered via :func:`argsort_op` if
// required.
//
// See Also
// --------
// :func:`sort_op` — produces the full sorted view.
LUCID_API TensorImplPtr kthvalue_op(const TensorImplPtr& a, std::int64_t k, int dim, bool keepdim);

}  // namespace lucid
