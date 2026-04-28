#include "Einops.h"

#include <set>
#include <string>
#include <vector>

#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "../ufunc/Reductions.h"
#include "../ufunc/Transpose.h"
#include "../utils/View.h"
#include "_Pattern.h"

namespace lucid {

using einops_detail::Token;
using einops_detail::parse_side;
using einops_detail::flat_axes;
using einops_detail::split_arrow;

namespace {

TensorImplPtr dispatch_reduce(const TensorImplPtr& x,
                              const std::vector<int>& axes,
                              const std::string& reduction) {
    if (axes.empty()) return x;
    if (reduction == "sum")  return sum_op(x, axes, /*keepdims=*/false);
    if (reduction == "mean") return mean_op(x, axes, /*keepdims=*/false);
    if (reduction == "max")  return max_op(x, axes, /*keepdims=*/false);
    if (reduction == "min")  return min_op(x, axes, /*keepdims=*/false);
    if (reduction == "prod") return prod_op(x, axes, /*keepdims=*/false);
    throw LucidError("reduce: unknown reduction '" + reduction + "'");
}

}  // namespace

TensorImplPtr einops_reduce_op(
    const TensorImplPtr& a, const std::string& pattern,
    const std::string& reduction,
    const std::map<std::string, std::int64_t>& axes_lengths) {
    if (!a) throw LucidError("reduce: null input");
    OpScope scope{"einops_reduce", a->device_, a->dtype_, a->shape_};

    auto [lhs_str, rhs_str] = split_arrow(pattern);
    auto lhs = parse_side(lhs_str);
    auto rhs = parse_side(rhs_str);

    auto flat_lhs = flat_axes(lhs);
    auto flat_rhs_v = flat_axes(rhs);
    std::set<std::string> flat_rhs(flat_rhs_v.begin(), flat_rhs_v.end());

    // Step 1: rearrange lhs → flat-lhs canonical order via einops_rearrange_op.
    std::string flat_lhs_str;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i) {
        if (i) flat_lhs_str.push_back(' ');
        flat_lhs_str += flat_lhs[i];
    }
    auto cur = einops_rearrange_op(a, lhs_str + " -> " + flat_lhs_str,
                                    axes_lengths);

    // Step 2: reduce away axes that don't appear in rhs.
    std::vector<int> reduce_axes;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i)
        if (flat_rhs.find(flat_lhs[i]) == flat_rhs.end())
            reduce_axes.push_back(static_cast<int>(i));
    cur = dispatch_reduce(cur, reduce_axes, reduction);

    // Step 3: rearrange surviving flat axes → grouped rhs.
    std::vector<std::string> surviving;
    for (auto& n : flat_lhs)
        if (flat_rhs.find(n) != flat_rhs.end()) surviving.push_back(n);
    std::string surviving_str;
    for (std::size_t i = 0; i < surviving.size(); ++i) {
        if (i) surviving_str.push_back(' ');
        surviving_str += surviving[i];
    }
    if (cur->shape_.size() != surviving.size())
        throw LucidError("reduce: internal axis bookkeeping mismatch");
    return einops_rearrange_op(cur, surviving_str + " -> " + rhs_str,
                                axes_lengths);
}

}  // namespace lucid
