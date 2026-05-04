// lucid/_C/nn/EmbeddingBag.cpp
#include "EmbeddingBag.h"
#include "../backend/Dispatcher.h"
#include "../core/ErrorBuilder.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
namespace lucid {

TensorImplPtr embedding_bag_op(const TensorImplPtr& weight,
                                 const TensorImplPtr& indices,
                                 const TensorImplPtr& offsets,
                                 int mode,
                                 int padding_idx,
                                 bool include_last_offset) {
    Validator::input(weight,  "embedding_bag.weight").non_null();
    Validator::input(indices, "embedding_bag.indices").non_null();
    Validator::input(offsets, "embedding_bag.offsets").non_null();

    const int B = static_cast<int>(offsets->shape()[0]);
    const int D = static_cast<int>(weight->shape()[1]);
    Shape out_shape = {static_cast<std::int64_t>(B), static_cast<std::int64_t>(D)};
    OpScopeFull scope{"embedding_bag", weight->device(), weight->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(weight->device());
    Storage out = be.embedding_bag_forward(
        weight->storage(), indices->storage(), offsets->storage(),
        weight->shape(), indices->shape(),
        mode, padding_idx, include_last_offset, weight->dtype());

    return std::make_shared<TensorImpl>(
        std::move(out), out_shape, weight->dtype(), weight->device(), false);
}

}  // namespace lucid
