// lucid/_C/ops/composite/Reductions.cpp
//
// ``logsumexp`` composes max + sub + exp + sum + log + add to recover the
// stable formula without registering a new schema.  The reduction runs with
// keepdims and the reduced axes are dropped by one reshape when ``keepdims``
// is false.

#include "Reductions.h"

#include "../../autograd/Helpers.h"
#include "../bfunc/Add.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Exponential.h"
#include "../ufunc/Predicate.h"
#include "../ufunc/Reductions.h"
#include "../utils/Select.h"
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
    //
    // The axes go through the reductions' one normaliser: sorting the raw
    // signed axes and squeezing in that order dropped a wrapped negative
    // axis after a larger positive one (``dim=(-3, 2)`` on 3-D: "squeeze:
    // axis out of range").
    const auto norm = normalize_axes(axes, static_cast<int>(a->shape().size()), "logsumexp");
    bool empty_axis = false;
    for (int ax : norm)
        empty_axis = empty_axis || a->shape()[static_cast<std::size_t>(ax)] == 0;
    if (empty_axis)
        return log_op(sum_op(exp_op(a), norm, keepdims));

    // Reduce with keepdims=true so the subtraction broadcasts naturally.  An
    // infinite max shifts by 0 instead: ``inf - inf`` is NaN, so a row holding
    // +inf summed to NaN (the reference: +inf) and an all -inf row too
    // (the reference: -inf).
    auto m_raw = max_op(a, norm, true);
    auto m_keep = where_op(isinf_op(m_raw), zeros_like_op(m_raw), m_raw);
    auto shifted = sub_op(a, m_keep);
    auto exp_shifted = exp_op(shifted);
    auto summed = sum_op(exp_shifted, norm, true);
    auto out_keepdim = add_op(log_op(summed), m_keep);

    if (keepdims)
        return out_keepdim;
    return reshape_op(out_keepdim, reduce_output_shape(a->shape(), norm, /*keepdims=*/false));
}

}  // namespace lucid
