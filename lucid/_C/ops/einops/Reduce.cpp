// lucid/_C/ops/einops/Reduce.cpp
//
// Implementation of einops_reduce_op: permute + reshape + reduce.
//
// Algorithm:
//   1. Parse "lhs -> rhs" with _Pattern.h helpers.
//   2. Flatten both sides to axis-name sequences.
//   3. Use einops_rearrange_op to bring the input to the flat lhs order.
//   4. Identify which flat-lhs axes do not appear in the rhs; reduce over those.
//   5. Use einops_rearrange_op again to bring the surviving axes into the rhs
//      shape (handles grouping on the output side).
//
// Supported reduction codes (match Python-side enum):
//   1 = mean, 2 = sum, 3 = max, 4 = min, 5 = prod
//
// Autograd: the composed rearrange/reduce/rearrange ops each carry their own
// autograd nodes, so gradient flow works automatically.

#include <set>
#include <string>
#include <vector>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../ufunc/Reductions.h"
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

// Dispatch to the appropriate reduction primitive based on the reduction code.
//
// axes is the sorted list of axis indices (into the current tensor layout) to
// reduce over.  keepdims is always false here because einops_reduce_op handles
// shape management via the surviving-axis rearrange in the calling function.
//
// Returns the input unchanged when axes is empty (all lhs axes appear in the
// rhs, so no reduction is needed — effectively an identity rearrange).
//
// Reduction codes: 1=mean, 2=sum, 3=max, 4=min, 5=prod.  These must stay
// synchronised with the Python-side enum in the bindings layer.
TensorImplPtr dispatch_reduce(const TensorImplPtr& x, const std::vector<int>& axes, int reduction) {
    if (axes.empty())
        return x;
    switch (reduction) {
    case 1:
        return mean_op(x, axes, false);
    case 2:
        return sum_op(x, axes, false);
    case 3:
        return max_op(x, axes, false);
    case 4:
        return min_op(x, axes, false);
    case 5:
        return prod_op(x, axes, false);
    default:
        ErrorBuilder("reduce").fail("unknown reduction code " + std::to_string(reduction));
    }
}

}  // namespace

// Reduce tensor a as specified by pattern and reduction code.
//
// The two-rearrange strategy avoids re-implementing axis tracking from scratch:
//   1. First rearrange brings any grouped lhs dims into flat form so that every
//      axis occupies its own tensor dimension.  After this step the tensor rank
//      equals flat_lhs.size() and the axis order matches flat_lhs.
//   2. dispatch_reduce collapses axes that appear in flat_lhs but not flat_rhs.
//      The surviving axes retain their relative order within the flat_lhs
//      sequence (i.e. their positions in the post-reduction tensor are
//      determined by their original lhs order, not by the rhs order).
//   3. Second rearrange converts the surviving-axis layout into the rhs shape,
//      applying any grouping or reordering specified in rhs.
//
// Example: "b c h w -> b (h w)", reduction=2 (sum)
//   flat_lhs = ["b","c","h","w"],  flat_rhs = {"b","h","w"}
//   reduce_axes = [1]  (c is absent from flat_rhs)
//   surviving = ["b","h","w"]
//   second rearrange: "b h w -> b (h w)"
TensorImplPtr einops_reduce_op(const TensorImplPtr& a,
                               const std::string& pattern,
                               int reduction,
                               const std::map<std::string, std::int64_t>& axes_lengths) {
    Validator::input(a, "reduce.a").non_null();
    OpScopeFull scope{"einops_reduce", a->device(), a->dtype(), a->shape()};

    auto [lhs_str, rhs_str] = split_arrow(pattern);
    auto lhs = parse_side(lhs_str);
    auto rhs = parse_side(rhs_str);

    auto flat_lhs = flat_axes(lhs);
    auto flat_rhs_v = flat_axes(rhs);
    // Use a set for O(1) membership queries when determining reduction axes.
    std::set<std::string> flat_rhs(flat_rhs_v.begin(), flat_rhs_v.end());

    // Build a whitespace-separated flat-lhs pattern string used as the rhs of
    // the first rearrange (splitting any merged lhs dims into individual axes).
    std::string flat_lhs_str;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i) {
        if (i)
            flat_lhs_str.push_back(' ');
        flat_lhs_str += flat_lhs[i];
    }
    // First rearrange: transform the (possibly grouped) lhs to flat axis order.
    auto cur = einops_rearrange_op(a, lhs_str + " -> " + flat_lhs_str, axes_lengths);

    // Any flat-lhs axis missing from flat-rhs must be reduced away.
    std::vector<int> reduce_axes;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i)
        if (flat_rhs.find(flat_lhs[i]) == flat_rhs.end())
            reduce_axes.push_back(static_cast<int>(i));
    cur = dispatch_reduce(cur, reduce_axes, reduction);

    // Collect the axes that survived the reduction in their original lhs order.
    std::vector<std::string> surviving;
    for (auto& n : flat_lhs)
        if (flat_rhs.find(n) != flat_rhs.end())
            surviving.push_back(n);

    // Build the whitespace-separated surviving-axis pattern string.
    std::string surviving_str;
    for (std::size_t i = 0; i < surviving.size(); ++i) {
        if (i)
            surviving_str.push_back(' ');
        surviving_str += surviving[i];
    }
    // Sanity check: after reduction the tensor rank must equal the number of
    // surviving axes.  A mismatch would indicate a bug in reduce_axes logic.
    if (cur->shape().size() != surviving.size())
        ErrorBuilder("reduce").fail("internal axis bookkeeping mismatch");
    // Second rearrange: go from flat surviving axes to the requested rhs shape,
    // applying any merges or reorderings specified in the rhs pattern.
    return einops_rearrange_op(cur, surviving_str + " -> " + rhs_str, axes_lengths);
}

}  // namespace lucid
