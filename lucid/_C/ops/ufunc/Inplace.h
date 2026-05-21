// lucid/_C/ops/ufunc/Inplace.h
//
// Public entry points for the in-place (``_``-suffixed) forms of every
// elementwise unary op in the ufunc subsystem.
//
// Each function runs the corresponding out-of-place op
// (e.g. :func:`neg_op`, :func:`exp_op`), copies the resulting
// :class:`Storage` back into the source tensor, updates the dtype / device
// fields, and bumps the input :class:`TensorImpl`'s version counter so
// autograd can detect illegal mutations of tensors that are still saved
// for backward.
//
// Notes
// -----
// **Why in-place ops exist.**  The primary benefit is peak-memory
// reduction during inference and training of large models: writing the
// activation back over its own buffer halves the working set of an
// element-wise step.  All Lucid in-place ops are pure wrappers around
// the out-of-place op + a storage swap — they do not bypass dispatch or
// AMP, so behaviour is identical to the non-mutating form.
//
// **Mutation semantics.**  Each function returns the *same*
// :class:`TensorImplPtr` it received, now holding the new storage.
// Aliases of the original tensor (other ``TensorImplPtr`` handles to
// the same impl) observe the mutation; aliases that hold an independent
// view but share storage do not — Lucid views are not refcounted on
// storage, only on impl.
//
// **Autograd safety.**  These ops are **not** autograd-safe when
// ``a->requires_grad()`` is ``true`` *and* ``a`` is still reachable
// from a saved tensor on a live backward node.  The version-counter
// bump issued by ``a->bump_version()`` lets autograd raise during
// backward if a saved snapshot is later mutated, but it is the
// **caller's responsibility** to avoid the situation; callers must
// check before invoking these entry points, typically by routing
// through the Python-side ``_no_grad_check`` guard.
//
// **Invariants enforced.**  Every function asserts that the output of
// the underlying out-of-place op has the same :class:`Shape` as the
// input — any shape change raises :exc:`ShapeMismatch`.  Dtype and
// device, on the other hand, are *overwritten* on the input impl to
// match the out-of-place result (the AMP autocast layer may have
// promoted the dtype during dispatch).
//
// See Also
// --------
// :file:`Inplace.cpp` — defines the shared ``inplace_unary<Fn>`` helper
//     that all entry points below delegate to.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// In-place element-wise negation: writes $-x$ back into ``a``.
//
// Equivalent to ``a = neg_op(a)`` but reuses ``a``'s storage and bumps
// its version counter.  Useful for negating large activations without
// allocating a second buffer.
//
// Math
// ----
// $$a_i \leftarrow -a_i$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to negate in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``, now holding the negated storage.
//
// Raises
// ------
// ShapeMismatch
//     If :func:`neg_op` returns a tensor whose shape differs from
//     ``a->shape()`` (should never happen under the standard schema).
//
// See Also
// --------
// :func:`neg_op` — out-of-place counterpart.
LUCID_API TensorImplPtr neg_inplace_op(const TensorImplPtr& a);

// In-place absolute value: writes $|x|$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow |a_i|$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to take absolute value of in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`abs_op` — out-of-place counterpart.
LUCID_API TensorImplPtr abs_inplace_op(const TensorImplPtr& a);

// In-place sign function: writes $\mathrm{sign}(x) \in \{-1, 0, +1\}$
// back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \mathrm{sign}(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to apply sign to in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`sign_op` — out-of-place counterpart.
LUCID_API TensorImplPtr sign_inplace_op(const TensorImplPtr& a);

// In-place reciprocal: writes $1/x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \frac{1}{a_i}$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to invert in place.  Must be non-null.  Behaviour at
//     $a_i = 0$ follows the dispatching backend (typically $\pm\infty$).
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`reciprocal_op` — out-of-place counterpart.
LUCID_API TensorImplPtr reciprocal_inplace_op(const TensorImplPtr& a);

// In-place square: writes $x^2$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow a_i^2$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to square in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`square_op` — out-of-place counterpart.
LUCID_API TensorImplPtr square_inplace_op(const TensorImplPtr& a);

// In-place cube: writes $x^3$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow a_i^3$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to cube in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`cube_op` — out-of-place counterpart.
LUCID_API TensorImplPtr cube_inplace_op(const TensorImplPtr& a);

// In-place natural exponential: writes $e^x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \exp(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to exponentiate in place.  Must be non-null.  AMP may
//     promote integer inputs to float before dispatch; the promoted
//     dtype is written back to ``a``.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`exp_op` — out-of-place counterpart.
LUCID_API TensorImplPtr exp_inplace_op(const TensorImplPtr& a);

// In-place natural logarithm: writes $\ln(x)$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \ln(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to log in place.  Must be non-null.  Inputs with
//     $a_i \le 0$ follow the backend's convention (typically NaN /
//     $-\infty$).
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`log_op` — out-of-place counterpart.
LUCID_API TensorImplPtr log_inplace_op(const TensorImplPtr& a);

// In-place base-2 logarithm: writes $\log_2 x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \log_2 a_i$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to log in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`log2_op` — out-of-place counterpart.
LUCID_API TensorImplPtr log2_inplace_op(const TensorImplPtr& a);

// In-place square root: writes $\sqrt{x}$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \sqrt{a_i}$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to take square root of in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`sqrt_op` — out-of-place counterpart.
LUCID_API TensorImplPtr sqrt_inplace_op(const TensorImplPtr& a);

// In-place sine: writes $\sin x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \sin(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor (radians) to sine in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`sin_op` — out-of-place counterpart.
LUCID_API TensorImplPtr sin_inplace_op(const TensorImplPtr& a);

// In-place cosine: writes $\cos x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \cos(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor (radians) to cosine in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`cos_op` — out-of-place counterpart.
LUCID_API TensorImplPtr cos_inplace_op(const TensorImplPtr& a);

// In-place tangent: writes $\tan x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \tan(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor (radians) to tangent in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`tan_op` — out-of-place counterpart.
LUCID_API TensorImplPtr tan_inplace_op(const TensorImplPtr& a);

// In-place inverse sine: writes $\arcsin x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \arcsin(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to arcsin in place.  Must be non-null.  Values outside
//     $[-1, 1]$ follow the backend's NaN convention.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`arcsin_op` — out-of-place counterpart.
LUCID_API TensorImplPtr arcsin_inplace_op(const TensorImplPtr& a);

// In-place inverse cosine: writes $\arccos x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \arccos(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to arccos in place.  Must be non-null.  Values outside
//     $[-1, 1]$ follow the backend's NaN convention.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`arccos_op` — out-of-place counterpart.
LUCID_API TensorImplPtr arccos_inplace_op(const TensorImplPtr& a);

// In-place inverse tangent: writes $\arctan x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \arctan(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to arctan in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`arctan_op` — out-of-place counterpart.
LUCID_API TensorImplPtr arctan_inplace_op(const TensorImplPtr& a);

// In-place hyperbolic sine: writes $\sinh x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \sinh(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to sinh in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`sinh_op` — out-of-place counterpart.
LUCID_API TensorImplPtr sinh_inplace_op(const TensorImplPtr& a);

// In-place hyperbolic cosine: writes $\cosh x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \cosh(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to cosh in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`cosh_op` — out-of-place counterpart.
LUCID_API TensorImplPtr cosh_inplace_op(const TensorImplPtr& a);

// In-place hyperbolic tangent: writes $\tanh x$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \tanh(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to tanh in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`tanh_op` — out-of-place counterpart.
LUCID_API TensorImplPtr tanh_inplace_op(const TensorImplPtr& a);

// In-place banker's rounding: writes $\mathrm{round}(x)$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \mathrm{round}(a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to round in place.  Must be non-null.  Half-to-even
//     ("banker's") rounding follows the backend default.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`round_op` — out-of-place counterpart.
LUCID_API TensorImplPtr round_inplace_op(const TensorImplPtr& a);

// In-place floor: writes $\lfloor x \rfloor$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \lfloor a_i \rfloor$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to floor in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`floor_op` — out-of-place counterpart.
LUCID_API TensorImplPtr floor_inplace_op(const TensorImplPtr& a);

// In-place ceiling: writes $\lceil x \rceil$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \lceil a_i \rceil$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to ceil in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// See Also
// --------
// :func:`ceil_op` — out-of-place counterpart.
LUCID_API TensorImplPtr ceil_inplace_op(const TensorImplPtr& a);

// In-place logistic sigmoid: writes $\sigma(x) = 1/(1 + e^{-x})$ back
// into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \sigma(a_i) = \frac{1}{1 + e^{-a_i}}$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to apply sigmoid to in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// Notes
// -----
// Primarily useful for inference paths where the pre-activation buffer
// can be safely reused.  Avoid in training when ``a`` is saved for the
// backward of a downstream op.
//
// See Also
// --------
// :func:`sigmoid_op` — out-of-place counterpart.
LUCID_API TensorImplPtr sigmoid_inplace_op(const TensorImplPtr& a);

// In-place rectified linear unit: writes $\max(0, x)$ back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \max(0, a_i)$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to apply ReLU to in place.  Must be non-null.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// Notes
// -----
// The most common in-place op in practice: ResNet-style residual
// blocks save a per-layer activation by writing ReLU back over the
// convolution output.  Safe under autograd as long as the convolution
// was the only saved-for-backward dependency.
//
// See Also
// --------
// :func:`relu_op` — out-of-place counterpart.
LUCID_API TensorImplPtr relu_inplace_op(const TensorImplPtr& a);

// In-place clip / clamp: writes $\mathrm{clip}(x, \mathit{lo}, \mathit{hi})$
// back into ``a``.
//
// Math
// ----
// $$a_i \leftarrow \min(\mathit{hi}, \max(\mathit{lo}, a_i))$$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to clip in place.  Must be non-null.
// lo : double
//     Lower bound; values below ``lo`` are set to ``lo``.
// hi : double
//     Upper bound; values above ``hi`` are set to ``hi``.
//
// Returns
// -------
// TensorImplPtr
//     The same pointer as ``a``.
//
// Notes
// -----
// Implemented directly (without going through the generic
// ``inplace_unary`` helper) because the two scalar parameters
// ``lo`` / ``hi`` cannot be threaded through the zero-argument
// function-pointer template the other entry points use.
//
// See Also
// --------
// :func:`clip_op` — out-of-place counterpart.
LUCID_API TensorImplPtr clip_inplace_op(const TensorImplPtr& a, double lo, double hi);

}  // namespace lucid
