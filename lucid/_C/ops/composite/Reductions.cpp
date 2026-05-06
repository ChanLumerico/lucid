// lucid/_C/ops/composite/Reductions.cpp
//
// ``logsumexp`` composes max + sub + exp + sum + log + add to recover the
// stable formula without registering a new schema.  The reduce axes are
// collapsed via repeated ``squeeze_op`` calls (in descending order so each
// remaining index stays valid) when ``keepdims`` is false.

#include "Reductions.h"

#include <algorithm>
#include <functional>

#include "../bfunc/Add.h"
#include "../bfunc/Sub.h"
#include "../ufunc/Exponential.h"
#include "../ufunc/Reductions.h"
#include "../utils/View.h"

namespace lucid {

TensorImplPtr logsumexp_op(const TensorImplPtr& a,
                           const std::vector<int>& axes,
                           bool keepdims) {
    // Reduce with keepdims=true so the subtraction broadcasts naturally.
    auto m_keep = max_op(a, axes, true);
    auto shifted = sub_op(a, m_keep);
    auto exp_shifted = exp_op(shifted);
    auto summed = sum_op(exp_shifted, axes, true);
    auto out_keepdim = add_op(log_op(summed), m_keep);

    if (keepdims)
        return out_keepdim;

    // Drop the reduced axes one at a time.  Sorting descending keeps each
    // squeeze operating on a still-valid index after earlier removals.
    std::vector<int> drop = axes;
    std::sort(drop.begin(), drop.end(), std::greater<int>());
    auto out = out_keepdim;
    for (int axis : drop)
        out = squeeze_op(out, axis);
    return out;
}

}  // namespace lucid
