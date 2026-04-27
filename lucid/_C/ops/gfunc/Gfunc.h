#pragma once

// =====================================================================
// Lucid C++ engine — tensor generation ops (Phase 4c, mirrors gfunc).
// =====================================================================
//
// Direct C++ counterpart to `lucid/_func/gfunc.py`: zeros, ones, full, eye,
// arange, linspace, empty, diag (the `_like` family is in the binding layer
// since it just forwards to the shape-aware variants).
//
// All creation ops produce a fresh leaf tensor with `requires_grad` configurable
// from the call site. They have no autograd hookup — there is nothing to
// differentiate through.

#include <cstdint>

#include "../../api.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr zeros_op(const Shape& shape, Dtype dt, Device device,
                                 bool requires_grad = false);

LUCID_API TensorImplPtr ones_op(const Shape& shape, Dtype dt, Device device,
                                bool requires_grad = false);

LUCID_API TensorImplPtr full_op(const Shape& shape, double fill_value,
                                Dtype dt, Device device,
                                bool requires_grad = false);

LUCID_API TensorImplPtr empty_op(const Shape& shape, Dtype dt, Device device,
                                 bool requires_grad = false);

LUCID_API TensorImplPtr eye_op(std::int64_t N, std::int64_t M, std::int64_t k,
                               Dtype dt, Device device,
                               bool requires_grad = false);

LUCID_API TensorImplPtr arange_op(double start, double stop, double step,
                                  Dtype dt, Device device,
                                  bool requires_grad = false);

LUCID_API TensorImplPtr linspace_op(double start, double stop, std::int64_t num,
                                    Dtype dt, Device device,
                                    bool requires_grad = false);

/// `diag(v, k)`:
///   • If `v` is 1-D: returns a 2-D matrix with `v` on the k-th diagonal.
///   • If `v` is 2-D: returns the k-th diagonal as a 1-D vector.
LUCID_API TensorImplPtr diag_op(const TensorImplPtr& v, std::int64_t k = 0);

// ----- `_like` family — copy shape/dtype/device of an existing tensor -------
LUCID_API TensorImplPtr zeros_like_op(const TensorImplPtr& a,
                                       bool requires_grad = false);
LUCID_API TensorImplPtr ones_like_op(const TensorImplPtr& a,
                                      bool requires_grad = false);
LUCID_API TensorImplPtr empty_like_op(const TensorImplPtr& a,
                                       bool requires_grad = false);
LUCID_API TensorImplPtr full_like_op(const TensorImplPtr& a, double fill_value,
                                      bool requires_grad = false);

}  // namespace lucid
