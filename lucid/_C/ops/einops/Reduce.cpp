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
    std::set<std::string> flat_rhs(flat_rhs_v.begin(), flat_rhs_v.end());

    std::string flat_lhs_str;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i) {
        if (i)
            flat_lhs_str.push_back(' ');
        flat_lhs_str += flat_lhs[i];
    }
    auto cur = einops_rearrange_op(a, lhs_str + " -> " + flat_lhs_str, axes_lengths);

    std::vector<int> reduce_axes;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i)
        if (flat_rhs.find(flat_lhs[i]) == flat_rhs.end())
            reduce_axes.push_back(static_cast<int>(i));
    cur = dispatch_reduce(cur, reduce_axes, reduction);

    std::vector<std::string> surviving;
    for (auto& n : flat_lhs)
        if (flat_rhs.find(n) != flat_rhs.end())
            surviving.push_back(n);
    std::string surviving_str;
    for (std::size_t i = 0; i < surviving.size(); ++i) {
        if (i)
            surviving_str.push_back(' ');
        surviving_str += surviving[i];
    }
    if (cur->shape().size() != surviving.size())
        ErrorBuilder("reduce").fail("internal axis bookkeeping mismatch");
    return einops_rearrange_op(cur, surviving_str + " -> " + rhs_str, axes_lengths);
}

}  // namespace lucid
