// lucid/_C/nn/EmbeddingBag.h
#pragma once
#include "../api.h"
#include "../core/fwd.h"
namespace lucid {
// mode: 0=sum, 1=mean, 2=max
LUCID_API TensorImplPtr embedding_bag_op(const TensorImplPtr& weight,
                                           const TensorImplPtr& indices,
                                           const TensorImplPtr& offsets,
                                           int mode,
                                           int padding_idx,
                                           bool include_last_offset);
}  // namespace lucid
