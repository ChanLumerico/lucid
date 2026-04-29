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
/// Neg inplace.
LUCID_API TensorImplPtr neg_inplace_op(const TensorImplPtr& a);
/// Abs inplace.
LUCID_API TensorImplPtr abs_inplace_op(const TensorImplPtr& a);
/// Sign inplace.
LUCID_API TensorImplPtr sign_inplace_op(const TensorImplPtr& a);
/// Reciprocal inplace.
LUCID_API TensorImplPtr reciprocal_inplace_op(const TensorImplPtr& a);
/// Square inplace.
LUCID_API TensorImplPtr square_inplace_op(const TensorImplPtr& a);
/// Cube inplace.
LUCID_API TensorImplPtr cube_inplace_op(const TensorImplPtr& a);

// Exponential / log
/// Exp inplace.
LUCID_API TensorImplPtr exp_inplace_op(const TensorImplPtr& a);
/// Log inplace.
LUCID_API TensorImplPtr log_inplace_op(const TensorImplPtr& a);
/// Log2 inplace.
LUCID_API TensorImplPtr log2_inplace_op(const TensorImplPtr& a);
/// Sqrt inplace.
LUCID_API TensorImplPtr sqrt_inplace_op(const TensorImplPtr& a);

// Trig
/// Sin inplace.
LUCID_API TensorImplPtr sin_inplace_op(const TensorImplPtr& a);
/// Cos inplace.
LUCID_API TensorImplPtr cos_inplace_op(const TensorImplPtr& a);
/// Tan inplace.
LUCID_API TensorImplPtr tan_inplace_op(const TensorImplPtr& a);
/// Arcsin inplace.
LUCID_API TensorImplPtr arcsin_inplace_op(const TensorImplPtr& a);
/// Arccos inplace.
LUCID_API TensorImplPtr arccos_inplace_op(const TensorImplPtr& a);
/// Arctan inplace.
LUCID_API TensorImplPtr arctan_inplace_op(const TensorImplPtr& a);

// Hyperbolic
/// Sinh inplace.
LUCID_API TensorImplPtr sinh_inplace_op(const TensorImplPtr& a);
/// Cosh inplace.
LUCID_API TensorImplPtr cosh_inplace_op(const TensorImplPtr& a);
/// Tanh inplace.
LUCID_API TensorImplPtr tanh_inplace_op(const TensorImplPtr& a);

// Discrete
/// Round inplace.
LUCID_API TensorImplPtr round_inplace_op(const TensorImplPtr& a);
/// Floor inplace.
LUCID_API TensorImplPtr floor_inplace_op(const TensorImplPtr& a);
/// Ceil inplace.
LUCID_API TensorImplPtr ceil_inplace_op(const TensorImplPtr& a);

// Scalar-parameterized
/// Clip inplace.
LUCID_API TensorImplPtr clip_inplace_op(const TensorImplPtr& a, double lo, double hi);

}  // namespace lucid
