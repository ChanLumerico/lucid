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
#include "../ufunc/Astype.h"
#include "../ufunc/Exponential.h"
#include "../ufunc/Reductions.h"
#include "../utils/View.h"

namespace lucid {

TensorImplPtr logsumexp_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    // A float16 / bfloat16 input is evaluated in float32 and the answer
    // rounded back.  The sum of exp(x - max) is up to the length of the
    // reduced axis, which float16 cannot hold past 65504: logsumexp of
    // 70000 zeros came back inf where the reference gives log(70000).
    if (a->dtype() == Dtype::F16 || a->dtype() == Dtype::BF16)
        return astype_op(logsumexp_op(astype_op(a, Dtype::F32), axes, keepdims), a->dtype());

    // An empty reduced axis has no maximum to shift by — ``max`` refuses it,
    // having no identity — and needs none: the sum of nothing is 0 and its
    // log is -inf, which is logsumexp's identity and the reference's answer.
    const int ndim = static_cast<int>(a->shape().size());
    bool empty_axis = axes.empty() && a->numel() == 0;
    for (int ax : axes) {
        const int w = ax < 0 ? ax + ndim : ax;
        if (w >= 0 && w < ndim && a->shape()[static_cast<std::size_t>(w)] == 0)
            empty_axis = true;
    }
    if (empty_axis)
        return log_op(sum_op(exp_op(a), axes, keepdims));

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
