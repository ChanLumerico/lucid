// lucid/_C/ops/bfunc/Inplace.h
//
// Public entry points for the in-place (``_``-suffixed) forms of every
// elementwise binary op in the bfunc subsystem — ``add_``, ``sub_``,
// ``mul_``, ``div_``, ``pow_``, ``maximum_``, ``minimum_``.
//
// Each function runs the corresponding out-of-place op
// (e.g. :func:`add_op`, :func:`mul_op`), splices the resulting
// :class:`Storage` back into the *left* input tensor ``a``, updates
// ``a``'s dtype / device fields to match the result, and bumps
// ``a``'s version counter so autograd can detect illegal mutations
// of tensors that are still saved for backward.
//
// Notes
// -----
// **Why in-place binary ops exist.**  Like their unary siblings, the
// primary benefit is peak-memory reduction: writing ``a + b`` back over
// ``a``'s buffer avoids allocating a fresh output, which is significant
// for residual / accumulator patterns inside training loops and for
// optimizer state updates (``param.grad_.mul_(scale)``).
//
// **Asymmetric mutation semantics.**  The first operand ``a`` is the
// in-place target; ``b`` is read-only.  The returned
// :class:`TensorImplPtr` is the *same* pointer as ``a``, now holding the
// new storage.  Aliases of ``a`` (other ``TensorImplPtr`` handles to
// the same impl) observe the mutation; aliases that hold an independent
// view but share storage do not — Lucid views are not refcounted on
// storage, only on impl, which is why ``storage_is_shared()`` rejects
// the op rather than risk a silently broken view.
//
// **Broadcasting.**  The out-of-place binary op may broadcast ``b``
// up to ``a``'s shape, but ``a`` itself cannot be smaller than the
// broadcast output — there is no place to write the extra elements.
// The shape-equality post-check in ``inplace_apply`` enforces this:
// if broadcasting would have grown ``a``, the op raises
// :exc:`ShapeMismatch` instead of silently allocating a new buffer.
//
// **Autograd safety.**  These ops are **not** autograd-safe when
// ``a->requires_grad()`` is ``true`` *and* ``a`` is still reachable
// from a saved tensor on a live backward node.  The version-counter
// bump issued by ``a->bump_version()`` lets autograd raise during
// backward if a saved snapshot is later mutated, but it is the
// **caller's responsibility** to avoid the situation; callers must
// check before invoking these entry points (typically through the
// Python-side ``_no_grad_check`` guard).
//
// **Invariants enforced.**
//   - Neither ``a`` nor ``b`` may be null.
//   - ``a`` must not share storage with a view tensor
//     (``storage_is_shared()`` must be ``false``) — otherwise the op
//     would corrupt the view; the error tells the caller to ``.clone()``
//     first or operate on the base tensor.
//   - The out-of-place result must have the same :class:`Shape` as
//     ``a``; otherwise :exc:`ShapeMismatch` is raised.
//   - Dtype and device, on the other hand, are *overwritten* on
//     ``a``'s impl to match the out-of-place result (the AMP autocast
//     layer may have promoted the dtype during dispatch).
//
// See Also
// --------
// :file:`Inplace.cpp` — defines the shared ``inplace_apply<Fn>`` helper
//     that every entry point below delegates to.
// :file:`../ufunc/Inplace.h` — the unary in-place analog.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// In-place elementwise addition: writes $a + b$ back into ``a``.
//
// Equivalent to ``a = add_op(a, b)`` but reuses ``a``'s storage and
// bumps its version counter.  ``b`` may broadcast up to ``a``'s shape
// but the broadcast output shape must equal ``a->shape()``.
//
// Math
// ----
// $$a_i \leftarrow a_i + b_i$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand and in-place target.  Must be non-null and must not
//     share storage with a view.
// b : TensorImplPtr
//     Right operand.  Must be non-null and have a shape broadcastable
//     to ``a->shape()`` such that the broadcast result equals
//     ``a->shape()``.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``, now holding the summed storage.
//
// Raises
// ------
// ShapeMismatch
//     If the broadcast output of ``add_op(a, b)`` does not equal
//     ``a->shape()`` (i.e. ``a`` would have to grow to fit the result).
// RuntimeError
//     If ``a`` shares storage with a view (call ``.clone()`` first).
//
// See Also
// --------
// :func:`add_op` — out-of-place counterpart.
LUCID_API TensorImplPtr add_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// In-place elementwise subtraction: writes $a - b$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow a_i - b_i$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand and in-place target.  Must be non-null and must not
//     share storage with a view.
// b : TensorImplPtr
//     Right operand.  Must be non-null and broadcastable into
//     ``a->shape()``.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``, now holding the difference storage.
//
// Raises
// ------
// ShapeMismatch
//     If the broadcast output would change ``a``'s shape.
// RuntimeError
//     If ``a`` shares storage with a view.
//
// See Also
// --------
// :func:`sub_op` — out-of-place counterpart.
LUCID_API TensorImplPtr sub_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// In-place elementwise multiplication: writes $a \cdot b$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow a_i \cdot b_i$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand and in-place target.  Must be non-null and must not
//     share storage with a view.
// b : TensorImplPtr
//     Right operand.  Must be non-null and broadcastable into
//     ``a->shape()``.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``, now holding the product storage.
//
// Raises
// ------
// ShapeMismatch
//     If the broadcast output would change ``a``'s shape.
// RuntimeError
//     If ``a`` shares storage with a view.
//
// Notes
// -----
// Common usage: scaling gradients (``grad_.mul_(scale)``) or applying
// element-wise masks (``out_.mul_(mask)``) without allocating a new
// buffer per step.
//
// See Also
// --------
// :func:`mul_op` — out-of-place counterpart.
LUCID_API TensorImplPtr mul_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// In-place elementwise division: writes $a / b$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \frac{a_i}{b_i}$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand and in-place target.  Must be non-null and must not
//     share storage with a view.
// b : TensorImplPtr
//     Right operand.  Must be non-null and broadcastable into
//     ``a->shape()``.  Division by zero follows the backend's IEEE-754
//     convention (typically $\pm\infty$ or NaN).
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``, now holding the quotient storage.
//
// Raises
// ------
// ShapeMismatch
//     If the broadcast output would change ``a``'s shape.
// RuntimeError
//     If ``a`` shares storage with a view.
//
// See Also
// --------
// :func:`div_op` — out-of-place counterpart.
LUCID_API TensorImplPtr div_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// In-place elementwise exponentiation: writes $a^b$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow a_i^{b_i}$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Base / in-place target.  Must be non-null and must not share
//     storage with a view.
// b : TensorImplPtr
//     Exponent.  Must be non-null and broadcastable into
//     ``a->shape()``.  Domain restrictions (e.g. $0^0$, negative
//     base with fractional exponent) follow the backend's convention.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``, now holding the powered storage.
//
// Raises
// ------
// ShapeMismatch
//     If the broadcast output would change ``a``'s shape.
// RuntimeError
//     If ``a`` shares storage with a view.
//
// See Also
// --------
// :func:`pow_op` — out-of-place counterpart.
LUCID_API TensorImplPtr pow_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// In-place elementwise maximum: writes $\max(a, b)$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \max(a_i, b_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand and in-place target.  Must be non-null and must not
//     share storage with a view.
// b : TensorImplPtr
//     Right operand.  Must be non-null and broadcastable into
//     ``a->shape()``.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``, now holding the elementwise maxima.
//
// Raises
// ------
// ShapeMismatch
//     If the broadcast output would change ``a``'s shape.
// RuntimeError
//     If ``a`` shares storage with a view.
//
// See Also
// --------
// :func:`maximum_op` — out-of-place counterpart.
// :func:`minimum_inplace_op` — elementwise minimum sibling.
LUCID_API TensorImplPtr maximum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

// In-place elementwise minimum: writes $\min(a, b)$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \min(a_i, b_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand and in-place target.  Must be non-null and must not
//     share storage with a view.
// b : TensorImplPtr
//     Right operand.  Must be non-null and broadcastable into
//     ``a->shape()``.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``, now holding the elementwise minima.
//
// Raises
// ------
// ShapeMismatch
//     If the broadcast output would change ``a``'s shape.
// RuntimeError
//     If ``a`` shares storage with a view.
//
// See Also
// --------
// :func:`minimum_op` — out-of-place counterpart.
// :func:`maximum_inplace_op` — elementwise maximum sibling.
LUCID_API TensorImplPtr minimum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
