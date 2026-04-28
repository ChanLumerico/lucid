#pragma once

// =====================================================================
// Lucid C++ engine — embedding family.
// =====================================================================
//
//   embedding(weight, indices, padding_idx)
//     Index-gather rows of a weight table. weight: [N, D], indices any
//     shape, output: [...indices, D]. padding_idx (-1 = none) zeroes the
//     output rows where index == padding_idx, and gradient does not flow
//     into those rows of weight.
//
//   sinusoidal_pos_embedding(seq_len, embed_dim, dtype, device)
//     Pure forward — emits the standard sinusoidal table without grad:
//       table[i, 2k]   = sin(i · exp(-2k · log(10000)/D))
//       table[i, 2k+1] = cos(i · ...)
//
//   rotary_pos_embedding(x, position_ids, interleaved)
//     Apply RoPE to the last 2 dims (L, D) of x. theta_k = exp(-2k·log(10000)/D),
//     freq[i, k] = pos[i] · theta_k. Final:
//       interleaved : (out[i, 2k], out[i, 2k+1]) = R(freq) · (x[i, 2k], x[i, 2k+1])
//       split-half  : (out[i, k],  out[i, k+H])  = R(freq) · (x[i, k],  x[i, k+H])
//     where R(f) = [[cos f, -sin f], [sin f, cos f]]. Saves cos/sin tables
//     for backward.

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class LUCID_API EmbeddingBackward : public FuncOp<EmbeddingBackward, 1> {
public:
    static const OpSchema schema_v1;
    int padding_idx_ = -1;
    Shape weight_shape_;
    Storage saved_indices_;
    Shape saved_indices_shape_;
    Dtype saved_indices_dtype_ = Dtype::I64;
    static TensorImplPtr forward(const TensorImplPtr& weight,
                                 const TensorImplPtr& indices,
                                 int padding_idx);
    std::vector<Storage> apply(Storage grad_out) override;
};

class LUCID_API RotaryPosEmbeddingBackward : public FuncOp<RotaryPosEmbeddingBackward, 1> {
public:
    static const OpSchema schema_v1;
    bool interleaved_ = true;
    Shape orig_shape_;
    // [L, D/2] tables saved in the input dtype.
    Storage saved_cos_;
    Storage saved_sin_;
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& position_ids_or_null,
                                 bool interleaved);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr embedding_op(const TensorImplPtr& weight,
                                     const TensorImplPtr& indices,
                                     int padding_idx);

LUCID_API TensorImplPtr sinusoidal_pos_embedding_op(std::int64_t seq_len,
                                                    std::int64_t embed_dim,
                                                    Dtype dtype,
                                                    Device device);

LUCID_API TensorImplPtr rotary_pos_embedding_op(const TensorImplPtr& input,
                                                const TensorImplPtr& position_ids_or_null,
                                                bool interleaved);

}  // namespace lucid
