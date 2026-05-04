// lucid/_C/nn/Embedding.h
//
// Autograd-aware integer-index embedding lookup and positional encoding helpers.
//
// EmbeddingBackward: looks up rows of a (vocab_size, embed_dim) weight matrix
//   by integer indices.  Backward is a sparse scatter-add of grad_out into
//   the weight gradient.  Rows at padding_idx are zeroed in the forward output
//   and receive no gradient in backward.
//
// RotaryPosEmbeddingBackward: applies Rotary Position Embedding (RoPE) to an
//   input tensor by rotating pairs of features using precomputed cos/sin tables.
//   The backward rotates the incoming gradient using the saved tables.
//
// sinusoidal_pos_embedding_op: generates a fixed (no grad) sinusoidal position
//   encoding matrix of shape (seq_len, embed_dim); does not create a backward
//   node.

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the embedding lookup.
//
// Only the weight matrix is a differentiable input; indices are integer-typed
// and saved directly in saved_indices_ rather than via saved_inputs_.
// Backward: IBackend::embedding_backward performs scatter-add of grad_out
// into the weight gradient at positions given by the saved indices.
class LUCID_API EmbeddingBackward : public FuncOp<EmbeddingBackward, 1> {
public:
    static const OpSchema schema_v1;
    int padding_idx_ = -1;         // Rows at this index are zeroed and skipped.
    Shape weight_shape_;           // (num_embeddings, embed_dim).
    Storage saved_indices_;        // Integer index tensor from forward.
    Shape saved_indices_shape_;
    Dtype saved_indices_dtype_ = Dtype::I64;

    // weight – (num_embeddings, embed_dim); indices – integer tensor of any shape.
    // Returns a tensor of shape (*indices.shape, embed_dim).
    static TensorImplPtr
    forward(const TensorImplPtr& weight, const TensorImplPtr& indices, int padding_idx);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for Rotary Position Embedding (RoPE).
//
// Computes cos/sin tables during the forward pass (returned by the backend as
// rope_out[1] and rope_out[2]) and saves them for the backward.
// interleaved_ controls the pair layout: true = adjacent pairs (d=0,1, 2,3 ...),
// false = split-half layout (first D/2 paired with second D/2).
class LUCID_API RotaryPosEmbeddingBackward : public FuncOp<RotaryPosEmbeddingBackward, 1> {
public:
    static const OpSchema schema_v1;
    bool interleaved_ = true;  // Pair layout; must match how the model was trained.
    Shape orig_shape_;

    Storage saved_cos_;  // Cosine table produced during forward.
    Storage saved_sin_;  // Sine table produced during forward.

    // input – (..., L, D) with D even.  position_ids_or_null – integer (L,) or null.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& position_ids_or_null,
                                 bool interleaved);
    std::vector<Storage> apply(Storage grad_out) override;
};

// Integer-index embedding lookup.
LUCID_API TensorImplPtr embedding_op(const TensorImplPtr& weight,
                                     const TensorImplPtr& indices,
                                     int padding_idx);

// Generate a fixed sinusoidal position encoding matrix (no grad).
// Returns shape (seq_len, embed_dim).
LUCID_API TensorImplPtr sinusoidal_pos_embedding_op(std::int64_t seq_len,
                                                    std::int64_t embed_dim,
                                                    Dtype dtype,
                                                    Device device);

// Apply Rotary Position Embedding to input.
LUCID_API TensorImplPtr rotary_pos_embedding_op(const TensorImplPtr& input,
                                                const TensorImplPtr& position_ids_or_null,
                                                bool interleaved);

// Pooled embedding lookup (bag mode).
// Combines multiple embeddings per bag via sum (mode=0), mean (mode=1), or
// max (mode=2).  offsets marks the start of each bag in the flat indices
// tensor.  padding_idx < 0 means no padding.
LUCID_API TensorImplPtr embedding_bag_op(const TensorImplPtr& weight,
                                          const TensorImplPtr& indices,
                                          const TensorImplPtr& offsets,
                                          int mode,
                                          int padding_idx,
                                          bool include_last_offset);

}  // namespace lucid
