#pragma once

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
    static TensorImplPtr
    forward(const TensorImplPtr& weight, const TensorImplPtr& indices, int padding_idx);
    std::vector<Storage> apply(Storage grad_out) override;
};

class LUCID_API RotaryPosEmbeddingBackward : public FuncOp<RotaryPosEmbeddingBackward, 1> {
public:
    static const OpSchema schema_v1;
    bool interleaved_ = true;
    Shape orig_shape_;

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
