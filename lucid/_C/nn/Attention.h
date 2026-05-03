// lucid/_C/nn/Attention.h
//
// Autograd-aware Scaled Dot-Product Attention (SDPA).
//
// Computes:  out = softmax(Q @ K^T * scale + mask) @ V
//
// Backend dispatch:
//   GPU – calls mlx::core::fast::scaled_dot_product_attention, which returns
//         a dummy {1}-shape weights tensor.  The backward detects ndim==1 and
//         recomputes W = softmax(Q @ K^T * scale) before computing gradients.
//   CPU – rolls the full softmax(Q @ K^T / sqrt(d)) @ V computation by hand.
//
// Two public entry points are provided:
//   scaled_dot_product_attention_op       – returns output only.
//   scaled_dot_product_attention_with_weights_op – returns {output, weights}.

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for Scaled Dot-Product Attention.
//
// saved_weights_ stores the attention weight matrix W = softmax(Q @ K^T * scale).
// On the GPU path the backend may return a dummy {1}-shape tensor instead; the
// backward recomputes W in that case using the saved Q, K, and scale_.
// Saved edges: {Q, K, V} (3 inputs).
class LUCID_API ScaledDotProductAttentionBackward
    : public FuncOp<ScaledDotProductAttentionBackward, 3> {
public:
    static const OpSchema schema_v1;
    double scale_ = 1.0;     // Dot-product scale factor (usually 1/sqrt(d_k)).
    Shape orig_q_shape_;
    Shape orig_k_shape_;
    Shape orig_v_shape_;

    // Attention weight matrix; may be a {1} placeholder on the GPU path.
    Storage saved_weights_;

    // Run the forward pass.
    // q, k, v     – at least 2-D tensors of shape (..., L, D).
    //               All leading "batch" dims must be equal.
    //               Q.D must equal K.D; K.L must equal V.L.
    // attn_mask   – optional additive mask (nullptr = none); broadcastable to
    //               (..., Lq, Lk).
    // scale       – multiplied by Q @ K^T before softmax.
    // is_causal   – if true, a causal lower-triangular mask is applied.
    // Returns output of shape (..., Lq, Dv).
    static TensorImplPtr forward(const TensorImplPtr& q,
                                 const TensorImplPtr& k,
                                 const TensorImplPtr& v,
                                 const TensorImplPtr& attn_mask_or_null,
                                 double scale,
                                 bool is_causal);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point that returns only the attention output tensor.
LUCID_API TensorImplPtr scaled_dot_product_attention_op(const TensorImplPtr& q,
                                                        const TensorImplPtr& k,
                                                        const TensorImplPtr& v,
                                                        const TensorImplPtr& attn_mask_or_null,
                                                        double scale,
                                                        bool is_causal);

// Public entry point that returns {output, weights}.
// When autograd is required a second ScaledDotProductAttentionBackward node is
// created for the output tensor so both return values can be used independently.
LUCID_API std::vector<TensorImplPtr>
scaled_dot_product_attention_with_weights_op(const TensorImplPtr& q,
                                             const TensorImplPtr& k,
                                             const TensorImplPtr& v,
                                             const TensorImplPtr& attn_mask_or_null,
                                             double scale,
                                             bool is_causal);

}  // namespace lucid
