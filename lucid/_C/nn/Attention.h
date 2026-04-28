#pragma once

// =====================================================================
// Lucid C++ engine — fused scaled-dot-product attention.
// =====================================================================
//
//   scores  = (Q @ K^T) * scale                                  [B, L_q, L_k]
//   scores  = scores + attn_mask         (additive float mask)
//   scores  = scores.where(attn_mask, -inf)  (boolean mask)
//   scores  = mask_upper(scores, diag=1) (is_causal)
//   weights = softmax(scores, axis=-1)
//   output  = weights @ V                                        [B, L_q, d_v]
//
// Q, K, V can have any number of leading batch dims; the last two are
// (L, d). All leading dims are flattened to a single batch axis B internally.
// Saves `weights` for backward (softmax output is the cheapest checkpoint
// for the chain rule through softmax).
//
// Backward chain:
//   dV       = weights^T @ dout
//   dweights = dout @ V^T
//   dscores  = weights * (dweights - rowsum(weights * dweights))
//   dQ       = (dscores @ K) * scale
//   dK       = (dscores^T @ Q) * scale
//
// AMP policy: ForceFP32 (softmax precision-sensitive).

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
    // [B, L_q, L_k] — softmax output, the only large tensor we need to save
    // for backward (cheaper than re-running mask + softmax).
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

// Returns {output, weights}. Weights are detached (no grad_fn) — they are an
// intermediate of `output`'s backward chain, not an independent differentiable
// tensor.
LUCID_API std::vector<TensorImplPtr> scaled_dot_product_attention_with_weights_op(
    const TensorImplPtr& q,
    const TensorImplPtr& k,
    const TensorImplPtr& v,
    const TensorImplPtr& attn_mask_or_null,
    double scale,
    bool is_causal);

}  // namespace lucid
