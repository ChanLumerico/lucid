#pragma once

// =====================================================================
// Lucid C++ engine — view-shape ops (reshape / squeeze / unsqueeze).
// =====================================================================
//
// All three are *metadata-only* in the contiguous-tensor world: the
// underlying buffer is unchanged, just `shape_` and `stride_` are recomputed.
// We materialize a copy anyway in Phase 3.4 v1 to keep the engine's
// contiguous-input contract trivial; the data is already laid out correctly,
// the copy is straight memcpy.
//
//   reshape(t, new_shape)  total numel must match
//   squeeze(t, dim)        remove a size-1 dim (or all size-1 dims if dim=-1)
//   unsqueeze(t, dim)      insert a size-1 dim at `dim`
//
// Backward: each op inverted (reshape→reshape, squeeze→unsqueeze, etc.).

#include <vector>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "../../autograd/FuncOp.h"

namespace lucid {

// One backward node serves all three ops since the implementation is a
// straight memcpy: backward = reshape grad to input_shape (memcpy back).
class LUCID_API ViewBackward : public FuncOp<ViewBackward, 1> {
public:
    static const OpSchema schema_v1;
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr reshape_op(const TensorImplPtr& a,
                                   const std::vector<std::int64_t>& new_shape);
LUCID_API TensorImplPtr squeeze_op(const TensorImplPtr& a, int dim);
LUCID_API TensorImplPtr squeeze_all_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr unsqueeze_op(const TensorImplPtr& a, int dim);

}  // namespace lucid
