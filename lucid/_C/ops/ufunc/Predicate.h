// lucid/_C/ops/ufunc/Predicate.h
//
// Floating-point predicate ops (isinf, isnan, isfinite) and nan_to_num.
// Predicates always produce a Bool output tensor; nan_to_num preserves dtype.
// None of these ops are differentiable.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Returns a Bool tensor: true where elements are ±Inf.
LUCID_API TensorImplPtr isinf_op(const TensorImplPtr& a);

// Returns a Bool tensor: true where elements are NaN.
LUCID_API TensorImplPtr isnan_op(const TensorImplPtr& a);

// Returns a Bool tensor: true where elements are finite (not NaN and not Inf).
LUCID_API TensorImplPtr isfinite_op(const TensorImplPtr& a);

// Replace NaN/Inf values with finite substitutes.
// nan_val    : replacement for NaN     (default 0.0)
// posinf_val : replacement for +Inf    (default max finite value)
// neginf_val : replacement for -Inf    (default min finite value)
LUCID_API TensorImplPtr nan_to_num_op(
    const TensorImplPtr& a,
    double nan_val    = 0.0,
    double posinf_val = 3.4028234663852886e+38,
    double neginf_val = -3.4028234663852886e+38);

// Full-tensor boolean reductions.  Output is a scalar Bool tensor.
// No gradient is attached.
LUCID_API TensorImplPtr any_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr all_op(const TensorImplPtr& a);

}  // namespace lucid
