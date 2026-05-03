// lucid/_C/ops/einops/Rearrange.cpp
//
// Implementation of einops_rearrange_op: pure permute + reshape, no reduction.
//
// Algorithm:
//   1. Parse "lhs -> rhs" with _Pattern.h helpers.
//   2. resolve_lhs_sizes: match each lhs token against the input shape to
//      determine the size of every named axis.  For group tokens where one
//      axis size is unknown, infer it by division.
//   3. Flatten lhs and rhs to ordered axis name sequences.
//   4. Reshape the input to the flat lhs shape (splitting any merged dims).
//   5. Permute axes if the flat rhs ordering differs from flat lhs.
//   6. Reshape to the final rhs shape (merging any grouped dims).
//
// Autograd: the composed reshape/permute/reshape ops each carry their own
// autograd nodes, so gradient flow works automatically.

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../ufunc/Transpose.h"
#include "../utils/View.h"
#include "Einops.h"
#include "_Pattern.h"

namespace lucid {

using einops_detail::flat_axes;
using einops_detail::parse_side;
using einops_detail::split_arrow;
using einops_detail::Token;

namespace {

// Walk the lhs token sequence and determine the size of every named axis by
// matching against in_shape.  Sizes provided in kwargs take precedence over
// those inferred from the input shape, allowing the caller to split a merged
// dimension by specifying one or more of its component sizes explicitly.
//
// Token processing by kind:
//   is_name()    — the axis size equals in_shape[i] directly.
//   is_literal() — in_shape[i] must equal the literal; raises if not.
//   is_group()   — the sizes of all child axes must multiply to in_shape[i].
//     If exactly one child axis size is unknown, it is inferred by division.
//     If all child axis sizes are known, the product is validated.
//     If more than one child axis is unknown, an error is raised because the
//     system is under-constrained (there are infinitely many solutions).
//
// This function is the single source of truth for axis sizes during a
// rearrange; all subsequent steps (permutation, reshape) read from the
// returned map.
std::map<std::string, std::int64_t>
resolve_lhs_sizes(const std::vector<Token>& lhs,
                  const Shape& in_shape,
                  const std::map<std::string, std::int64_t>& kwargs,
                  const char* op_name) {
    if (lhs.size() != in_shape.size())
        ErrorBuilder(op_name).fail("lhs token count != input ndim");
    std::map<std::string, std::int64_t> sz;
    // Pre-populate with caller-provided sizes (these override inferred values).
    for (auto& [k, v] : kwargs)
        sz[k] = v;

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const auto& tk = lhs[i];
        const std::int64_t dim = in_shape[i];
        if (tk.is_name()) {
            // Simple axis: size comes directly from the input dimension.
            sz[tk.name()] = dim;
        } else if (tk.is_group()) {
            const auto& inner = tk.group();
            std::vector<std::string> unknown;
            std::int64_t known_prod = 1;
            for (auto& sub : inner) {
                if (!sub.is_name())
                    ErrorBuilder(op_name).fail("nested groups/literals not allowed");
                const auto& nm = sub.name();
                auto it = sz.find(nm);
                if (it == sz.end())
                    unknown.push_back(nm);
                else
                    known_prod *= it->second;
            }
            if (unknown.size() == 1) {
                // Exactly one unknown: infer by dividing out the known product.
                if (known_prod == 0)
                    ErrorBuilder(op_name).fail("cannot infer axis from zero-product group");
                if (dim % known_prod != 0)
                    ErrorBuilder(op_name).fail("group does not divide input dim");
                sz[unknown[0]] = dim / known_prod;
            } else if (unknown.size() > 1) {
                // Under-constrained: caller must supply more sizes via kwargs.
                ErrorBuilder(op_name).fail("multiple unknown axes in group; pass via kwargs");
            } else if (known_prod != dim) {
                // Over-constrained and inconsistent: the sizes don't add up.
                ErrorBuilder(op_name).fail("group product mismatches input dim");
            }
        } else {
            // Literal token: validate that the input dimension matches exactly.
            if (tk.literal() != dim)
                ErrorBuilder(op_name).fail("literal mismatches input dim");
        }
    }
    return sz;
}

// Compute the target shape for the rhs by multiplying out group products and
// substituting named axis sizes from the resolved size map sz.
//
// This produces the shape to pass to reshape_op at the final rhs reshape step.
// Groups are represented as a single merged dimension (their product), which
// is exactly what reshape needs to merge multiple axes into one.
std::vector<std::int64_t> group_shape_merged(const std::vector<Token>& tokens,
                                             const std::map<std::string, std::int64_t>& sz) {
    std::vector<std::int64_t> out;
    for (auto& tk : tokens) {
        if (tk.is_name())
            out.push_back(sz.at(tk.name()));
        else if (tk.is_group()) {
            // A group on the rhs means merge: its size is the product of members.
            std::int64_t p = 1;
            for (auto& sub : tk.group())
                p *= sz.at(sub.name());
            out.push_back(p);
        } else {
            // Literal: use the integer value as the merged dimension size.
            out.push_back(tk.literal());
        }
    }
    return out;
}

}  // namespace

// Rearrange tensor a according to the einops pattern.
//
// The three-step pipeline (reshape to flat lhs, permute, reshape to rhs) is
// applied only when each step is actually needed — identity reshapes and
// identity permutations are detected and skipped to avoid unnecessary op
// nodes in the autograd graph (which would add overhead at backward time).
//
// Full example: pattern = "b (c h) w -> b c (h w)", a shape = [2, 12, 8]
//   flat_lhs = ["b","c","h","w"],  flat_lhs_shape = [2,3,4,8]
//   Step 2: reshape [2,12,8] -> [2,3,4,8]  (split the "(c h)" merged dim)
//   perm for flat_rhs ["b","c","h","w"] -> ["b","c","h","w"] = [0,1,2,3] (identity)
//   Step 3: no-op
//   rhs_shape from group_shape_merged = [2, 3, 32]  ("(h w)" = 4*8 = 32)
//   Step 4: reshape [2,3,4,8] -> [2,3,32]
TensorImplPtr einops_rearrange_op(const TensorImplPtr& a,
                                  const std::string& pattern,
                                  const std::map<std::string, std::int64_t>& axes_lengths) {
    Validator::input(a, "rearrange.a").non_null();
    OpScopeFull scope{"einops_rearrange", a->device(), a->dtype(), a->shape()};

    auto [lhs_str, rhs_str] = split_arrow(pattern);
    auto lhs = parse_side(lhs_str);
    auto rhs = parse_side(rhs_str);

    // Step 1: resolve every axis size from the input shape + caller kwargs.
    auto sz = resolve_lhs_sizes(lhs, a->shape(), axes_lengths, "rearrange");

    // Ensure all rhs axis names are also in sz.  Normally every rhs name also
    // appears on the lhs so sz already covers them; the fallback handles
    // unusual patterns where an rhs-only name was provided via axes_lengths.
    for (const auto& n : flat_axes(rhs)) {
        if (sz.find(n) == sz.end()) {
            auto it = axes_lengths.find(n);
            if (it != axes_lengths.end())
                sz[n] = it->second;
            else
                ErrorBuilder("rearrange").fail("unknown axis '" + n + "' on rhs");
        }
    }

    // Build the flat (fully split) lhs shape from the resolved axis sizes.
    auto flat_lhs = flat_axes(lhs);
    std::vector<std::int64_t> flat_lhs_shape(flat_lhs.size());
    for (std::size_t i = 0; i < flat_lhs.size(); ++i)
        flat_lhs_shape[i] = sz.at(flat_lhs[i]);

    auto cur = a;
    // Step 2: reshape to the flat lhs shape to split any merged input dims.
    // Skip if the input is already in flat form (common when no groups appear
    // on the lhs).
    if (Shape(flat_lhs_shape.begin(), flat_lhs_shape.end()) != a->shape())
        cur = reshape_op(cur, flat_lhs_shape);

    auto flat_rhs = flat_axes(rhs);
    // Step 3: permute axes if the rhs ordering differs from the lhs ordering.
    // Build a permutation vector mapping each rhs position to the corresponding
    // lhs position.
    if (flat_lhs != flat_rhs) {
        std::vector<int> perm(flat_rhs.size());
        for (std::size_t i = 0; i < flat_rhs.size(); ++i) {
            auto it = std::find(flat_lhs.begin(), flat_lhs.end(), flat_rhs[i]);
            if (it == flat_lhs.end())
                ErrorBuilder("rearrange").fail("rhs axis '" + flat_rhs[i] + "' not in lhs");
            perm[i] = static_cast<int>(it - flat_lhs.begin());
        }
        // Skip the permute call if the permutation is the identity.
        bool identity = true;
        for (std::size_t i = 0; i < perm.size(); ++i)
            if (perm[i] != static_cast<int>(i)) {
                identity = false;
                break;
            }
        if (!identity)
            cur = permute_op(cur, perm);
    }

    // Step 4: reshape to the final rhs shape to merge any grouped rhs dims.
    // Skip if already in the target shape (common when rhs has no groups).
    auto rhs_shape = group_shape_merged(rhs, sz);
    if (Shape(rhs_shape.begin(), rhs_shape.end()) != cur->shape())
        cur = reshape_op(cur, rhs_shape);

    return cur;
}

}  // namespace lucid
