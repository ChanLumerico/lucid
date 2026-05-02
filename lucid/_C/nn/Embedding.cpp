#include "Embedding.h"

#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {


// =====================================================================
// Embedding
// =====================================================================

const OpSchema EmbeddingBackward::schema_v1{"embedding", 1, AmpPolicy::Promote, true};

TensorImplPtr EmbeddingBackward::forward(const TensorImplPtr& weight,
                                         const TensorImplPtr& indices,
                                         int padding_idx) {
    if (!weight || !indices)
        ErrorBuilder("embedding").fail("null input");
    if (weight->device() != indices->device())
        throw DeviceMismatch(std::string(device_name(weight->device())),
                             std::string(device_name(indices->device())),
                             "embedding: weight/indices");
    if (weight->shape().size() != 2)
        throw ShapeMismatch(weight->shape(), Shape{},
                            "embedding: weight must be 2-D (num_embeddings, dim)");

    const std::int64_t D = weight->shape()[1];

    Shape out_shape = indices->shape();
    out_shape.push_back(D);
    OpScopeFull scope{schema_v1.name, weight->device(), weight->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(weight->device());
    Storage out_storage = be.embedding_forward(
        weight->storage(), indices->storage(),
        weight->shape(), indices->shape(), out_shape,
        padding_idx, weight->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, weight->dtype(),
                                            weight->device(), false);

    auto bwd = std::make_shared<EmbeddingBackward>();
    bwd->saved_indices_ = indices->storage();
    bwd->saved_indices_shape_ = indices->shape();
    bwd->saved_indices_dtype_ = indices->dtype();
    bwd->padding_idx_ = padding_idx;
    bwd->weight_shape_ = weight->shape();
    kernel::NaryKernel<EmbeddingBackward, 1>::wire_autograd(std::move(bwd), {weight}, out,
                                                            /*save_ins=*/false);
    return out;
}

std::vector<Storage> EmbeddingBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.embedding_backward(
        grad_out, saved_indices_,
        weight_shape_, saved_indices_shape_,
        padding_idx_, dtype_)};
}

TensorImplPtr embedding_op(const TensorImplPtr& weight,
                           const TensorImplPtr& indices,
                           int padding_idx) {
    return EmbeddingBackward::forward(weight, indices, padding_idx);
}
LUCID_REGISTER_OP(EmbeddingBackward)

// =====================================================================
// Sinusoidal positional embedding — pure forward, no grad.
// =====================================================================

TensorImplPtr sinusoidal_pos_embedding_op(std::int64_t seq_len,
                                          std::int64_t embed_dim,
                                          Dtype dtype,
                                          Device device) {
    if (seq_len < 0)
        ErrorBuilder("sinusoidal_pos_embedding").fail("seq_len < 0");
    if (embed_dim <= 0)
        ErrorBuilder("sinusoidal_pos_embedding").fail("embed_dim must be > 0");

    Shape out_shape{seq_len, embed_dim};
    OpScopeFull scope{"sinusoidal_pos_embedding", device, dtype, out_shape};

    auto& be = backend::Dispatcher::for_device(device);
    Storage out_s = be.sinusoidal_pos_embedding(seq_len, embed_dim, dtype);
    return std::make_shared<TensorImpl>(std::move(out_s), out_shape, dtype, device, false);
}

// =====================================================================
// Rotary positional embedding (RoPE).
// =====================================================================

const OpSchema RotaryPosEmbeddingBackward::schema_v1{"rotary_pos_embedding", 1,
                                                     AmpPolicy::ForceFP32, true};

TensorImplPtr RotaryPosEmbeddingBackward::forward(const TensorImplPtr& input,
                                                  const TensorImplPtr& position_ids_or_null,
                                                  bool interleaved) {
    Validator::input(input, "rotary_pos_embedding.input").non_null();
    if (position_ids_or_null && position_ids_or_null->device() != input->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(position_ids_or_null->device())),
                             "rotary_pos_embedding: input/position_ids");
    if (input->shape().size() < 2)
        ErrorBuilder("rotary_pos_embedding").fail("input must be at least 2-D ([..., L, D])");

    const std::size_t ndim = input->shape().size();
    const std::size_t D = static_cast<std::size_t>(input->shape()[ndim - 1]);
    if (D % 2 != 0)
        ErrorBuilder("rotary_pos_embedding").fail("embed_dim must be even");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), input->shape()};

    const Storage* pos_storage = position_ids_or_null ? &position_ids_or_null->storage() : nullptr;
    const Dtype pos_dt = position_ids_or_null ? position_ids_or_null->dtype() : Dtype::I64;
    auto& be = backend::Dispatcher::for_device(input->device());
    auto rope_out = be.rope_forward(
        input->storage(), pos_storage, input->shape(), interleaved, pos_dt, input->dtype());
    // rope_out = {out, saved_cos, saved_sin}
    Storage out_storage = std::move(rope_out[0]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), input->shape(),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<RotaryPosEmbeddingBackward>();
        bwd->saved_cos_ = std::move(rope_out[1]);
        bwd->saved_sin_ = std::move(rope_out[2]);
        bwd->interleaved_ = interleaved;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<RotaryPosEmbeddingBackward, 1>::wire_autograd(std::move(bwd), {input},
                                                                         out, /*save_ins=*/false);
    }
    return out;
}

std::vector<Storage> RotaryPosEmbeddingBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.rope_backward(
        grad_out, saved_cos_, saved_sin_, orig_shape_, interleaved_, dtype_)};
}

TensorImplPtr rotary_pos_embedding_op(const TensorImplPtr& input,
                                      const TensorImplPtr& position_ids_or_null,
                                      bool interleaved) {
    return RotaryPosEmbeddingBackward::forward(input, position_ids_or_null, interleaved);
}
LUCID_REGISTER_OP(RotaryPosEmbeddingBackward)

}  // namespace lucid
