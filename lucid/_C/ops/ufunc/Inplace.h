// lucid/_C/ops/ufunc/Inplace.h
//
// In-place unary operation entry points.  Each function runs the corresponding
// out-of-place op, writes the result back into the input tensor's storage, and
// bumps its version counter.  The version counter is used by autograd to detect
// illegal in-place modifications on tensors that still have downstream users.
//
// Invariants:
//   - The output shape must equal the input shape; any change raises ShapeMismatch.
//   - The dtype and device of `a` are updated to match the op output (AmpPolicy
//     may have promoted them).
//   - These functions are not autograd-safe when `a->requires_grad()` is true
//     and `a` has downstream users; callers are responsible for checking.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Arithmetic in-place ops.
LUCID_API TensorImplPtr neg_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr abs_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr sign_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr reciprocal_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr square_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr cube_inplace_op(const TensorImplPtr& a);

// Exponential / logarithm in-place ops.
LUCID_API TensorImplPtr exp_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr log_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr log2_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr sqrt_inplace_op(const TensorImplPtr& a);

// Trigonometric in-place ops.
LUCID_API TensorImplPtr sin_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr cos_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr tan_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr arcsin_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr arccos_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr arctan_inplace_op(const TensorImplPtr& a);

// Hyperbolic in-place ops.
LUCID_API TensorImplPtr sinh_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr cosh_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr tanh_inplace_op(const TensorImplPtr& a);

// Rounding in-place ops.
LUCID_API TensorImplPtr round_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr floor_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr ceil_inplace_op(const TensorImplPtr& a);

// Activation in-place ops.
LUCID_API TensorImplPtr sigmoid_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr relu_inplace_op(const TensorImplPtr& a);

// Clip in-place: clamp(a, lo, hi) written back into a.
LUCID_API TensorImplPtr clip_inplace_op(const TensorImplPtr& a, double lo, double hi);

}  // namespace lucid
