#pragma once

// =====================================================================
// In-place unary ops. Same semantics as bfunc/Inplace.h: compute the
// non-in-place counterpart, overwrite `a`'s storage, bump version.
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Element-wise arithmetic
LUCID_API TensorImplPtr neg_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr abs_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr sign_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr reciprocal_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr square_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr cube_inplace_op(const TensorImplPtr& a);

// Exponential / log
LUCID_API TensorImplPtr exp_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr log_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr log2_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr sqrt_inplace_op(const TensorImplPtr& a);

// Trig
LUCID_API TensorImplPtr sin_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr cos_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr tan_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr arcsin_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr arccos_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr arctan_inplace_op(const TensorImplPtr& a);

// Hyperbolic
LUCID_API TensorImplPtr sinh_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr cosh_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr tanh_inplace_op(const TensorImplPtr& a);

// Discrete
LUCID_API TensorImplPtr round_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr floor_inplace_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr ceil_inplace_op(const TensorImplPtr& a);

// Scalar-parameterized
LUCID_API TensorImplPtr clip_inplace_op(const TensorImplPtr& a, double lo, double hi);

}  // namespace lucid
