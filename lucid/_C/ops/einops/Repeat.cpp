// lucid/_C/ops/einops/Repeat.cpp
//
// Implementation of einops_repeat_op: introduce new axes and tile the tensor.
//
// Algorithm:
//   1. Parse "lhs -> rhs" with _Pattern.h helpers.
//   2. Use einops_rearrange_op to bring the input into the flat lhs order.
//   3. Build a size map for all lhs axes from the flattened input shape; look
//      up rhs-only axes (the new ones) from axes_lengths.
//   4. For each axis in flat_rhs that is not yet in the tensor, append it via
//      unsqueeze at the trailing position.
//   5. Permute to match the flat_rhs order (if different from the interim order).
//   6. broadcast_to the desired sizes (tiling new unit-size axes).
//   7. Reshape to the final rhs shape (handles grouping).
//
// Autograd: the composed rearrange/unsqueeze/permute/broadcast/reshape ops
// each carry their own autograd nodes, so gradient flow works automatically.

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../ufunc/Transpose.h"
#include "../utils/Layout.h"
#include "../utils/View.h"
#include "Einops.h"
#include "_Pattern.h"

namespace lucid {

using einops_detail::flat_axes;
using einops_detail::parse_side;
using einops_detail::split_arrow;

// Repeat (tile/broadcast) tensor a by introducing new axes defined in pattern.
//
// New axes must have their sizes supplied in axes_lengths.  Existing axes may
// be reordered and merged on the output side.  The implementation uses only
// unsqueeze_op + broadcast_to_op to avoid materialising full copies of the
// data when possible — the Metal/MLX GPU path and the CPU path can both
// broadcast in-place without allocating n * sizeof(element) extra memory.
//
// Example: pattern = "b c -> b 1 c n",  axes_lengths = {{"n", 4}}
//   flat_lhs = ["b","c"],  flat_rhs = ["b","1","c","n"]
//   Step 1: rearrange to flat lhs (no-op here, already flat).
//   Step 2: insert "n" as a trailing size-1 dim → shape [B, C, 1].
//   Step 3: permute ["b","c","n"] → ["b","1","c","n"] ...
//   Step 4: broadcast to [B, 1, C, N] — tiling the n axis from 1 to N.
//   Step 5: reshape to the final rhs shape (if rhs has groups; no-op here).
TensorImplPtr einops_repeat_op(const TensorImplPtr& a,
                               const std::string& pattern,
                               const std::map<std::string, std::int64_t>& axes_lengths) {
    Validator::input(a, "repeat.a").non_null();
    OpScopeFull scope{"einops_repeat", a->device(), a->dtype(), a->shape()};

    auto [lhs_str, rhs_str] = split_arrow(pattern);
    auto lhs = parse_side(lhs_str);
    auto rhs = parse_side(rhs_str);

    auto flat_lhs = flat_axes(lhs);
    auto flat_rhs = flat_axes(rhs);

    // Build a flat intermediate pattern string used as the rhs of the first
    // rearrange, which normalises any grouped lhs dims into separate axes.
    std::string flat_lhs_str;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i) {
        if (i)
            flat_lhs_str.push_back(' ');
        flat_lhs_str += flat_lhs[i];
    }
    // Step 1: rearrange input to flat lhs (split any grouped dims).
    auto cur = einops_rearrange_op(a, lhs_str + " -> " + flat_lhs_str, axes_lengths);

    // Build a size map: lhs axes come from the post-rearrange shape; new rhs
    // axes must be supplied by the caller via axes_lengths.
    std::map<std::string, std::int64_t> sz;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i)
        sz[flat_lhs[i]] = cur->shape()[i];
    for (auto& n : flat_rhs) {
        if (sz.find(n) == sz.end()) {
            auto it = axes_lengths.find(n);
            if (it == axes_lengths.end())
                ErrorBuilder("repeat").fail("new axis '" + n + "' requires kwargs size");
            sz[n] = it->second;
        }
    }

    // Step 2: insert each new rhs-only axis as a trailing size-1 dimension.
    // The unsqueeze inserts a broadcastable unit axis that broadcast_to will
    // later tile to the requested size.
    std::vector<std::string> interim = flat_lhs;
    std::set<std::string> lhs_set(flat_lhs.begin(), flat_lhs.end());
    for (auto& n : flat_rhs) {
        if (lhs_set.find(n) == lhs_set.end()) {
            // Append at the end; the permute in step 3 will re-order them.
            cur = unsqueeze_op(cur, static_cast<int>(cur->shape().size()));
            interim.push_back(n);
            lhs_set.insert(n);
        }
    }

    // Step 3: permute interim axis order to match the flat_rhs order.
    if (interim != flat_rhs) {
        std::vector<int> perm(flat_rhs.size());
        for (std::size_t i = 0; i < flat_rhs.size(); ++i) {
            auto it = std::find(interim.begin(), interim.end(), flat_rhs[i]);
            if (it == interim.end())
                ErrorBuilder("repeat").fail("rhs axis '" + flat_rhs[i] +
                                            "' missing after insertion");
            perm[i] = static_cast<int>(it - interim.begin());
        }
        // Skip if the permutation happens to be the identity.
        bool identity = true;
        for (std::size_t i = 0; i < perm.size(); ++i)
            if (perm[i] != static_cast<int>(i)) {
                identity = false;
                break;
            }
        if (!identity)
            cur = permute_op(cur, perm);
    }

    // Step 4: broadcast to the target per-axis sizes.  The size-1 dims
    // inserted in step 2 are tiled to their requested sizes here.
    Shape target_shape;
    for (auto& n : flat_rhs)
        target_shape.push_back(sz.at(n));
    if (cur->shape() != target_shape)
        cur = broadcast_to_op(cur, target_shape);

    // Step 5: reshape to the final rhs shape, merging any groups on the output.
    // Most patterns don't have rhs groups, so this is a no-op in practice.
    std::vector<std::int64_t> rhs_shape;
    for (auto& tk : rhs) {
        if (tk.is_name())
            rhs_shape.push_back(sz.at(tk.name()));
        else if (tk.is_group()) {
            // Group on rhs: multiply member sizes to get the merged dim size.
            std::int64_t p = 1;
            for (auto& sub : tk.group())
                p *= sz.at(sub.name());
            rhs_shape.push_back(p);
        } else {
            rhs_shape.push_back(tk.literal());
        }
    }
    if (Shape(rhs_shape.begin(), rhs_shape.end()) != cur->shape())
        cur = reshape_op(cur, rhs_shape);
    return cur;
}

}  // namespace lucid
