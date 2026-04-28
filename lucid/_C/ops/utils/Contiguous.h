#pragma once

// =====================================================================
// Lucid C++ engine — `.contiguous()` materialization op.
// =====================================================================
//
// Forces a tensor's storage to a contiguous layout. Currently a no-op (just
// clones) since Phase 3.4 ops always materialize. Becomes load-bearing when
// future zero-copy view ops produce non-contiguous TensorImpls.
//
// Backward: identity (clone). The gradient flows through unchanged because
// `contiguous` doesn't reshape — the layout change is byte-level only and
// gradient layout matches the output's anyway.

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

class LUCID_API ContiguousBackward : public FuncOp<ContiguousBackward, 1> {
public:
    static const OpSchema schema_v1;
    static TensorImplPtr forward(const TensorImplPtr& a);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr contiguous_op(const TensorImplPtr& a);

}  // namespace lucid
