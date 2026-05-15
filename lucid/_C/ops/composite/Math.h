// lucid/_C/ops/composite/Math.h
//
// Elementwise math operations whose forward pass is built from existing
// primitive ops in ``ufunc/`` and ``bfunc/``.  No new backward classes —
// gradients flow through the underlying primitives' autograd nodes.
//
// Membership criterion for this header: a single mathematical formula whose
// implementation requires composing two or more primitive ``*_op`` calls.
// Standalone primitives belong in ``ufunc/`` or ``bfunc/``.
//
// Listing:
//   log10  = log(x) / log(10)
//   log1p  = log(1 + x)
//   exp2   = exp(x · log(2))
//   trunc  = where(x ≥ 0, floor(x), ceil(x))           [zero gradient]
//   frac   = x − trunc(x)
//   atan2  = quadrant-aware arctangent (where + arctan + ±π corrections)
//   fmod   = a − trunc(a / b) · b                       [C-style modulo]
//   remainder = a − floor(a / b) · b                    [Python-style modulo]
//   hypot  = sqrt(a² + b²)
//   logaddexp = m + log(exp(a − m) + exp(b − m))        [stable; m = max(a,b)]

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// ── unary compositions ──────────────────────────────────────────────────────
LUCID_API TensorImplPtr log10_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr log1p_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr exp2_op(const TensorImplPtr& a);

// ``trunc`` and ``frac`` round toward zero / extract the fractional part.
// ``trunc`` itself is piecewise constant, so its contribution to autograd is
// zero — but ``frac = x − trunc(x)`` still propagates through the subtract.
LUCID_API TensorImplPtr trunc_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr frac_op(const TensorImplPtr& a);

// ── binary compositions ─────────────────────────────────────────────────────
LUCID_API TensorImplPtr atan2_op(const TensorImplPtr& y, const TensorImplPtr& x);
LUCID_API TensorImplPtr fmod_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr remainder_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr hypot_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr logaddexp_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Elementwise tolerance check: ``|a − b| ≤ atol + rtol·|b|``.  Returns a
// bool tensor.  Composes ``sub``, ``abs``, scalar ``mul``, ``add``, and
// ``less_equal`` so no dedicated kernel is required.
LUCID_API TensorImplPtr isclose_op(const TensorImplPtr& a,
                                   const TensorImplPtr& b,
                                   double rtol,
                                   double atol);

}  // namespace lucid
