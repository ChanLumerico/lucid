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

    std::string flat_lhs_str;
    for (std::size_t i = 0; i < flat_lhs.size(); ++i) {
        if (i)
            flat_lhs_str.push_back(' ');
        flat_lhs_str += flat_lhs[i];
    }
    auto cur = einops_rearrange_op(a, lhs_str + " -> " + flat_lhs_str, axes_lengths);

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

    std::vector<std::string> interim = flat_lhs;
    std::set<std::string> lhs_set(flat_lhs.begin(), flat_lhs.end());
    for (auto& n : flat_rhs) {
        if (lhs_set.find(n) == lhs_set.end()) {
            cur = unsqueeze_op(cur, static_cast<int>(cur->shape().size()));
            interim.push_back(n);
            lhs_set.insert(n);
        }
    }

    if (interim != flat_rhs) {
        std::vector<int> perm(flat_rhs.size());
        for (std::size_t i = 0; i < flat_rhs.size(); ++i) {
            auto it = std::find(interim.begin(), interim.end(), flat_rhs[i]);
            if (it == interim.end())
                ErrorBuilder("repeat").fail("rhs axis '" + flat_rhs[i] +
                                            "' missing after insertion");
            perm[i] = static_cast<int>(it - interim.begin());
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

    Shape target_shape;
    for (auto& n : flat_rhs)
        target_shape.push_back(sz.at(n));
    if (cur->shape() != target_shape)
        cur = broadcast_to_op(cur, target_shape);

    std::vector<std::int64_t> rhs_shape;
    for (auto& tk : rhs) {
        if (tk.is_name())
            rhs_shape.push_back(sz.at(tk.name()));
        else if (tk.is_group()) {
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
