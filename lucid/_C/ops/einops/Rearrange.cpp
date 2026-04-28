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

// Resolve every axis name on the LHS into its concrete size, using
// (a) the input shape, (b) supplied kwargs.
std::map<std::string, std::int64_t> resolve_lhs_sizes(
    const std::vector<Token>& lhs,
    const Shape& in_shape,
    const std::map<std::string, std::int64_t>& kwargs,
    const char* op_name) {
    if (lhs.size() != in_shape.size())
        ErrorBuilder(op_name).fail("lhs token count != input ndim");
    std::map<std::string, std::int64_t> sz;
    for (auto& [k, v] : kwargs)
        sz[k] = v;

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const auto& tk = lhs[i];
        const std::int64_t dim = in_shape[i];
        if (tk.is_name()) {
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
                if (known_prod == 0)
                    ErrorBuilder(op_name).fail("cannot infer axis from zero-product group");
                if (dim % known_prod != 0)
                    ErrorBuilder(op_name).fail("group does not divide input dim");
                sz[unknown[0]] = dim / known_prod;
            } else if (unknown.size() > 1) {
                ErrorBuilder(op_name).fail("multiple unknown axes in group; pass via kwargs");
            } else if (known_prod != dim) {
                ErrorBuilder(op_name).fail("group product mismatches input dim");
            }
        } else {
            // literal
            if (tk.literal() != dim)
                ErrorBuilder(op_name).fail("literal mismatches input dim");
        }
    }
    return sz;
}

// Compute the merged group-shape that the rhs token tree describes.
std::vector<std::int64_t> group_shape_merged(const std::vector<Token>& tokens,
                                             const std::map<std::string, std::int64_t>& sz) {
    std::vector<std::int64_t> out;
    for (auto& tk : tokens) {
        if (tk.is_name())
            out.push_back(sz.at(tk.name()));
        else if (tk.is_group()) {
            std::int64_t p = 1;
            for (auto& sub : tk.group())
                p *= sz.at(sub.name());
            out.push_back(p);
        } else {
            out.push_back(tk.literal());
        }
    }
    return out;
}

}  // namespace

TensorImplPtr einops_rearrange_op(const TensorImplPtr& a,
                                  const std::string& pattern,
                                  const std::map<std::string, std::int64_t>& axes_lengths) {
    Validator::input(a, "rearrange.a").non_null();
    OpScopeFull scope{"einops_rearrange", a->device_, a->dtype_, a->shape_};

    auto [lhs_str, rhs_str] = split_arrow(pattern);
    auto lhs = parse_side(lhs_str);
    auto rhs = parse_side(rhs_str);

    auto sz = resolve_lhs_sizes(lhs, a->shape_, axes_lengths, "rearrange");
    // Every name on rhs must resolve.
    for (const auto& n : flat_axes(rhs)) {
        if (sz.find(n) == sz.end()) {
            auto it = axes_lengths.find(n);
            if (it != axes_lengths.end())
                sz[n] = it->second;
            else
                ErrorBuilder("rearrange").fail("unknown axis '" + n + "' on rhs");
        }
    }

    // 1. reshape input → flat-lhs.
    auto flat_lhs = flat_axes(lhs);
    std::vector<std::int64_t> flat_lhs_shape(flat_lhs.size());
    for (std::size_t i = 0; i < flat_lhs.size(); ++i)
        flat_lhs_shape[i] = sz.at(flat_lhs[i]);

    auto cur = a;
    if (Shape(flat_lhs_shape.begin(), flat_lhs_shape.end()) != a->shape_)
        cur = reshape_op(cur, flat_lhs_shape);

    // 2. permute to flat-rhs order.
    auto flat_rhs = flat_axes(rhs);
    if (flat_lhs != flat_rhs) {
        std::vector<int> perm(flat_rhs.size());
        for (std::size_t i = 0; i < flat_rhs.size(); ++i) {
            auto it = std::find(flat_lhs.begin(), flat_lhs.end(), flat_rhs[i]);
            if (it == flat_lhs.end())
                ErrorBuilder("rearrange").fail("rhs axis '" + flat_rhs[i] + "' not in lhs");
            perm[i] = static_cast<int>(it - flat_lhs.begin());
        }
        bool identity = true;
        for (std::size_t i = 0; i < perm.size(); ++i)
            if (perm[i] != static_cast<int>(i)) {
                identity = false;
                break;
            }
        if (!identity)
            cur = permute_op(cur, perm);
    }

    // 3. reshape into rhs grouped/literal shape.
    auto rhs_shape = group_shape_merged(rhs, sz);
    if (Shape(rhs_shape.begin(), rhs_shape.end()) != cur->shape_)
        cur = reshape_op(cur, rhs_shape);

    return cur;
}

}  // namespace lucid
