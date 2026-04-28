#pragma once

// =====================================================================
// Lucid C++ engine — native einops (rearrange / reduce / repeat / einsum).
// =====================================================================
//
// All four ops are implemented as compositions of primitive autograd-aware
// engine ops (reshape, permute, broadcast_to, multiply, sum, ...). The
// pattern parser lives in C++ so dispatch/parsing has no Python overhead;
// gradients flow through the primitive ops' existing backward Nodes.
//
//   einops_rearrange_op(x, pattern, axes_lengths)
//     Pure shape rearrangement; equivalent to einops.rearrange.
//
//   einops_reduce_op(x, pattern, reduction, axes_lengths)
//     reduction codes: 1=mean, 2=sum, 3=max, 4=min, 5=prod.
//
//   einops_repeat_op(x, pattern, axes_lengths)
//     Insert+expand new axes per pattern.
//
//   einsum_op(pattern, operands)
//     Einstein summation. Multi-operand pairwise reduce.
//
// Determinism: deterministic.
// AMP policy: inherits from underlying primitives.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr
einops_rearrange_op(const TensorImplPtr& a,
                    const std::string& pattern,
                    const std::map<std::string, std::int64_t>& axes_lengths);

LUCID_API TensorImplPtr einops_reduce_op(const TensorImplPtr& a,
                                         const std::string& pattern,
                                         int reduction,
                                         const std::map<std::string, std::int64_t>& axes_lengths);

LUCID_API TensorImplPtr einops_repeat_op(const TensorImplPtr& a,
                                         const std::string& pattern,
                                         const std::map<std::string, std::int64_t>& axes_lengths);

LUCID_API TensorImplPtr einsum_op(const std::string& pattern,
                                  const std::vector<TensorImplPtr>& operands);

}  // namespace lucid
