// lucid/_C/ops/complex/Real.h
//
// Real-part extraction $\Re(z) = a$ for $z = a + b\,i$.
//
// Complex (C64) input yields an F32 output of the same shape.  Each backend
// implements the projection natively: CPU walks the interleaved
// ``[re, im, re, im, ...]`` storage with stride-2 reads from the real offset,
// GPU dispatches to ``mlx::core::real``.  Real-dtype inputs are rejected by
// ``complex_detail::require_complex``.
//
// Forward only — the Python autograd layer embeds the incoming real-valued
// gradient as the real part of a fresh complex gradient with zero imaginary
// part: ``d real(z) / d z = complex(grad, 0)``.  This matches the Wirtinger
// calculus convention used by complex autograd in the reference framework.
//
// Math
// ----
// $$
//   y = \Re(z), \qquad
//   \frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} + 0\,i
// $$
//
// References
// ----------
// Hirose, "Complex-Valued Neural Networks: Theories and Applications"
// (2003), §3.4 (Wirtinger derivatives).

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Extract the real part of a complex tensor as a real tensor.
//
// The result dtype is the corresponding real dtype (``C64`` → ``F32``); the
// shape and device are unchanged.
//
// Math
// ----
// $$
//   y_k = \Re(z_k)
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Complex-dtype input tensor (currently ``C64``).
//
// Returns
// -------
// TensorImplPtr
//     Real-dtype output (``F32``) of the same shape and device as ``a``.
//
// Raises
// ------
// DtypeMismatch
//     If ``a`` is not a complex dtype.
//
// See Also
// --------
// imag_op, complex_op, conj_op
LUCID_API TensorImplPtr real_op(const TensorImplPtr& a);

}  // namespace lucid
