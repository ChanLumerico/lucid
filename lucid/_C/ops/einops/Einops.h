// lucid/_C/ops/einops/Einops.h
//
// Public entry-point declarations for all einops operations implemented in
// the Lucid engine.  The four ops — rearrange, reduce, repeat, and einsum —
// together cover the full einops API surface.
//
// Each function follows the naming convention <name>_op used throughout the
// Lucid codebase.  They take TensorImplPtr inputs and return TensorImplPtr
// outputs; the Python bindings call these directly.
//
// Pattern string format:
//   All ops (except einsum) use the same "lhs -> rhs" pattern syntax parsed
//   by _Pattern.h.  Axis names are alphanumeric identifiers.  Parentheses
//   group multiple axes into a single merged dimension.  Integer literals
//   assert that a dimension has a specific fixed size.
//
//   Example: "b c (h w) -> b (c h) w"
//     - "(h w)" on the lhs splits a single input dim into h and w.
//     - "(c h)" on the rhs merges the c and h axes into one output dim.
//     - Rearrange / Reduce / Repeat all use this notation.
//
// Reduction codes used by einops_reduce_op:
//   1 = mean, 2 = sum, 3 = max, 4 = min, 5 = prod
//   These must stay in sync with the Python-side enum in the bindings layer.
//
// Implementation details (pattern parsing helpers, axis-size resolution) live
// in _Pattern.h and the per-op .cpp files.
//
// Autograd:
//   Rearrange, Reduce, and Repeat compose existing differentiable primitives
//   (reshape_op, permute_op, sum_op, broadcast_to_op, etc.) so autograd flows
//   automatically through the chain without dedicated backward nodes.
//   Einsum similarly composes mul_op and sum_op.  No separate *Backward
//   classes are needed for any of the four einops ops.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Rearrange (permute and/or reshape) a tensor according to an einops pattern.
//
// The pattern must have the form "lhs -> rhs".  Both sides must contain
// exactly the same named axes; rhs may group or split them with parentheses.
// axes_lengths provides sizes for any axis that cannot be inferred from the
// input shape.  No reduction is performed; element count is preserved.
LUCID_API TensorImplPtr
einops_rearrange_op(const TensorImplPtr& a,
                    const std::string& pattern,
                    const std::map<std::string, std::int64_t>& axes_lengths);

// Reduce a tensor over one or more axes as specified by an einops pattern.
//
// Axes present on the lhs but absent from the rhs are reduced using the
// given reduction code (1=mean, 2=sum, 3=max, 4=min, 5=prod).
// axes_lengths is forwarded to the internal rearrange steps.
LUCID_API TensorImplPtr einops_reduce_op(const TensorImplPtr& a,
                                         const std::string& pattern,
                                         int reduction,
                                         const std::map<std::string, std::int64_t>& axes_lengths);

// Tile/broadcast a tensor to introduce new axes as specified by an einops pattern.
//
// Axes present in rhs but absent from lhs are new; their sizes must be
// provided in axes_lengths.  Existing axes may be reordered and merged.
// Internally uses unsqueeze + broadcast_to to avoid data copies when possible.
LUCID_API TensorImplPtr einops_repeat_op(const TensorImplPtr& a,
                                         const std::string& pattern,
                                         const std::map<std::string, std::int64_t>& axes_lengths);

// General tensor contraction via an Einstein summation equation.
//
// The equation uses the same syntax as numpy.einsum: single-character labels
// separated by commas for the inputs, followed by "->" and the output labels.
// If the "->" is omitted, the output is the sorted set of labels that appear
// exactly once across all inputs (implicit form).
// Ellipsis (...) is not yet supported.
// Operands must all be on the same device and share a dtype.
LUCID_API TensorImplPtr einsum_op(const std::string& pattern,
                                  const std::vector<TensorImplPtr>& operands);

}  // namespace lucid
