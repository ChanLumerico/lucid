#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/Mul.h"
#include "../ufunc/Reductions.h"
#include "../ufunc/Transpose.h"
#include "../utils/Layout.h"
#include "../utils/View.h"
#include "Einops.h"

namespace lucid {

namespace {

// Strip all whitespace from a string.
std::string strip_ws(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (c != ' ' && c != '\t')
            out.push_back(c);
    return out;
}

// Split by ','.
std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

// Move/insert axes of `t` so the resulting tensor has labels `target_labels`,
// with size 1 in any axis not present in t's source labels (will be
// broadcast next). Returns the realigned tensor.
TensorImplPtr align_to_labels(const TensorImplPtr& t,
                              std::vector<std::string> src_labels,
                              const std::vector<std::string>& target_labels,
                              const std::map<std::string, std::int64_t>& sizes) {
    auto cur = t;
    // Insert trailing size-1 axes for any target label not in src.
    for (const auto& c : target_labels) {
        if (std::find(src_labels.begin(), src_labels.end(), c) == src_labels.end()) {
            cur = unsqueeze_op(cur, static_cast<int>(cur->shape_.size()));
            src_labels.push_back(c);
        }
    }
    // Build perm: for each target label, find its index in src_labels.
    std::vector<int> perm(target_labels.size());
    for (std::size_t i = 0; i < target_labels.size(); ++i) {
        auto it = std::find(src_labels.begin(), src_labels.end(), target_labels[i]);
        perm[i] = static_cast<int>(it - src_labels.begin());
    }
    bool identity = true;
    for (std::size_t i = 0; i < perm.size(); ++i)
        if (perm[i] != static_cast<int>(i)) {
            identity = false;
            break;
        }
    if (!identity)
        cur = permute_op(cur, perm);

    // Broadcast singleton axes to actual sizes.
    Shape target_shape;
    for (const auto& c : target_labels)
        target_shape.push_back(sizes.at(c));
    if (cur->shape_ != target_shape)
        cur = broadcast_to_op(cur, target_shape);
    return cur;
}

}  // namespace

TensorImplPtr einsum_op(const std::string& pattern, const std::vector<TensorImplPtr>& operands) {
    if (operands.empty())
        throw LucidError("einsum: at least one operand required");
    if (pattern.find("...") != std::string::npos)
        throw NotImplementedError(
            "einsum: ellipsis (...) patterns not yet supported in C++ engine");

    OpScope scope{"einsum", operands[0]->device_, operands[0]->dtype_, operands[0]->shape_};

    auto eq = strip_ws(pattern);
    std::string lhs, rhs;
    auto arrow = eq.find("->");
    if (arrow != std::string::npos) {
        lhs = eq.substr(0, arrow);
        rhs = eq.substr(arrow + 2);
    } else {
        // Implicit output: every label appearing exactly once across all
        // input specs, sorted alphabetically.
        lhs = eq;
        std::map<char, int> counts;
        for (char c : lhs) {
            if (c == ',')
                continue;
            counts[c]++;
        }
        std::vector<char> singles;
        for (auto& [c, v] : counts)
            if (v == 1)
                singles.push_back(c);
        std::sort(singles.begin(), singles.end());
        rhs.assign(singles.begin(), singles.end());
    }

    auto in_specs = split_commas(lhs);
    if (in_specs.size() != operands.size())
        throw LucidError("einsum: pattern operand count != operands.size()");

    // Build label → size map.
    std::map<std::string, std::int64_t> sizes;
    for (std::size_t k = 0; k < operands.size(); ++k) {
        const auto& spec = in_specs[k];
        const auto& t = operands[k];
        if (spec.size() != t->shape_.size())
            throw LucidError("einsum: operand " + std::to_string(k) + " has " +
                             std::to_string(t->shape_.size()) + " axes but spec '" + spec +
                             "' has " + std::to_string(spec.size()) + " labels");
        for (std::size_t i = 0; i < spec.size(); ++i) {
            std::string c(1, spec[i]);
            std::int64_t n = t->shape_[i];
            auto it = sizes.find(c);
            if (it == sizes.end())
                sizes[c] = n;
            else if (it->second != n)
                throw LucidError("einsum: label '" + c + "' has conflicting sizes " +
                                 std::to_string(it->second) + " vs " + std::to_string(n));
        }
    }
    for (char c : rhs) {
        std::string s(1, c);
        if (sizes.find(s) == sizes.end())
            throw LucidError("einsum: output label '" + s + "' not in inputs");
    }

    // Single-operand case: drop labels not in rhs (sum), then permute.
    if (operands.size() == 1) {
        std::vector<std::string> cur_labels;
        for (char c : in_specs[0])
            cur_labels.push_back(std::string(1, c));
        auto cur = operands[0];

        std::set<std::string> rhs_set;
        for (char c : rhs)
            rhs_set.insert(std::string(1, c));

        std::vector<int> kill;
        for (std::size_t i = 0; i < cur_labels.size(); ++i)
            if (rhs_set.find(cur_labels[i]) == rhs_set.end())
                kill.push_back(static_cast<int>(i));
        if (!kill.empty()) {
            cur = sum_op(cur, kill, /*keepdims=*/false);
            std::vector<std::string> new_labels;
            for (std::size_t i = 0; i < cur_labels.size(); ++i)
                if (rhs_set.find(cur_labels[i]) != rhs_set.end())
                    new_labels.push_back(cur_labels[i]);
            cur_labels = std::move(new_labels);
        }
        // Permute to rhs order.
        std::vector<std::string> rhs_labels;
        for (char c : rhs)
            rhs_labels.push_back(std::string(1, c));
        if (cur_labels != rhs_labels) {
            std::vector<int> perm(rhs_labels.size());
            for (std::size_t i = 0; i < rhs_labels.size(); ++i) {
                auto it = std::find(cur_labels.begin(), cur_labels.end(), rhs_labels[i]);
                perm[i] = static_cast<int>(it - cur_labels.begin());
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
        return cur;
    }

    // Multi-operand: pairwise reduce. cur_labels tracks the running result.
    auto cur = operands[0];
    std::vector<std::string> cur_labels;
    for (char c : in_specs[0])
        cur_labels.push_back(std::string(1, c));

    for (std::size_t k = 1; k < operands.size(); ++k) {
        const auto& right = operands[k];
        std::vector<std::string> right_labels;
        for (char c : in_specs[k])
            right_labels.push_back(std::string(1, c));

        // Future labels: anything still needed (rhs OR appearing in any
        // later operand).
        std::set<std::string> future_labels;
        for (char c : rhs)
            future_labels.insert(std::string(1, c));
        for (std::size_t j = k + 1; j < operands.size(); ++j)
            for (char c : in_specs[j])
                future_labels.insert(std::string(1, c));

        std::set<std::string> cur_set(cur_labels.begin(), cur_labels.end());
        std::set<std::string> right_set(right_labels.begin(), right_labels.end());
        std::set<std::string> union_set;
        union_set.insert(cur_set.begin(), cur_set.end());
        union_set.insert(right_set.begin(), right_set.end());

        // Keep = (cur ∪ right) ∩ future.
        // Contract = (cur ∩ right) − future.
        std::vector<std::string> keep, contract;
        for (auto& c : union_set) {
            bool inter = (cur_set.count(c) && right_set.count(c));
            if (future_labels.count(c))
                keep.push_back(c);
            else if (inter)
                contract.push_back(c);
            // else: label is in only one operand AND not in future — that's
            // a free label of just this one operand we should sum out.
            // It will get summed below as part of `kill`.
        }
        std::sort(keep.begin(), keep.end());
        std::sort(contract.begin(), contract.end());

        // Order axes: keep first, then contract (we sum the trailing axes).
        std::vector<std::string> order;
        order.insert(order.end(), keep.begin(), keep.end());
        order.insert(order.end(), contract.begin(), contract.end());

        // Drop "loner" labels (in only one operand AND not in future) by
        // pre-summing the operand they belong to.
        auto drop_loners = [&](TensorImplPtr t, std::vector<std::string>& labels,
                               const std::set<std::string>& other_set) {
            std::vector<int> kill;
            std::vector<std::string> new_labels;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (!future_labels.count(labels[i]) && !other_set.count(labels[i])) {
                    kill.push_back(static_cast<int>(i));
                } else {
                    new_labels.push_back(labels[i]);
                }
            }
            if (!kill.empty())
                t = sum_op(t, kill, /*keepdims=*/false);
            labels = std::move(new_labels);
            return t;
        };
        cur = drop_loners(cur, cur_labels, right_set);
        auto right_pre = drop_loners(right, right_labels, cur_set);

        cur = align_to_labels(cur, cur_labels, order, sizes);
        right_pre = align_to_labels(right_pre, right_labels, order, sizes);

        cur = mul_op(cur, right_pre);

        // Sum over trailing contract axes.
        if (!contract.empty()) {
            std::vector<int> sum_axes;
            for (std::size_t i = keep.size(); i < order.size(); ++i)
                sum_axes.push_back(static_cast<int>(i));
            cur = sum_op(cur, sum_axes, /*keepdims=*/false);
        }
        cur_labels.assign(order.begin(), order.begin() + keep.size());
    }

    // Final: drop any remaining label not in rhs (rare, e.g. a free label
    // that survived because it was in `keep` but not actually in rhs).
    std::set<std::string> rhs_set;
    for (char c : rhs)
        rhs_set.insert(std::string(1, c));
    std::vector<int> kill;
    std::vector<std::string> new_cur_labels;
    for (std::size_t i = 0; i < cur_labels.size(); ++i) {
        if (rhs_set.find(cur_labels[i]) == rhs_set.end())
            kill.push_back(static_cast<int>(i));
        else
            new_cur_labels.push_back(cur_labels[i]);
    }
    if (!kill.empty()) {
        cur = sum_op(cur, kill, /*keepdims=*/false);
        cur_labels = std::move(new_cur_labels);
    }

    // Permute to rhs order.
    std::vector<std::string> rhs_labels;
    for (char c : rhs)
        rhs_labels.push_back(std::string(1, c));
    if (cur_labels != rhs_labels) {
        std::vector<int> perm(rhs_labels.size());
        for (std::size_t i = 0; i < rhs_labels.size(); ++i) {
            auto it = std::find(cur_labels.begin(), cur_labels.end(), rhs_labels[i]);
            if (it == cur_labels.end())
                throw LucidError("einsum: rhs label '" + rhs_labels[i] +
                                 "' missing from cur_labels");
            perm[i] = static_cast<int>(it - cur_labels.begin());
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
    return cur;
}

}  // namespace lucid
