#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class LUCID_API ScaledDotProductAttentionBackward
    : public FuncOp<ScaledDotProductAttentionBackward, 3> {
public:
    static const OpSchema schema_v1;
    double scale_ = 1.0;
    Shape orig_q_shape_;
    Shape orig_k_shape_;
    Shape orig_v_shape_;

    Storage saved_weights_;

    static TensorImplPtr forward(const TensorImplPtr& q,
                                 const TensorImplPtr& k,
                                 const TensorImplPtr& v,
                                 const TensorImplPtr& attn_mask_or_null,
                                 double scale,
                                 bool is_causal);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr scaled_dot_product_attention_op(const TensorImplPtr& q,
                                                        const TensorImplPtr& k,
                                                        const TensorImplPtr& v,
                                                        const TensorImplPtr& attn_mask_or_null,
                                                        double scale,
                                                        bool is_causal);

LUCID_API std::vector<TensorImplPtr>
scaled_dot_product_attention_with_weights_op(const TensorImplPtr& q,
                                             const TensorImplPtr& k,
                                             const TensorImplPtr& v,
                                             const TensorImplPtr& attn_mask_or_null,
                                             double scale,
                                             bool is_causal);

}  // namespace lucid
