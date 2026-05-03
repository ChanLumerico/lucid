// lucid/_C/ops/utils/View.h
//
// Declares the autograd node and public free functions for shape-reinterpretation
// ops: reshape, squeeze, and unsqueeze.  All three produce a new TensorImpl that
// shares the underlying Storage with the input rather than copying data; they are
// therefore only valid when the input tensor is contiguous (verified by the
// backend dispatcher's reshape implementation).

#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for reshape (and the squeeze/unsqueeze ops that are built on
// top of it).
//
// Forward:  reinterpret the flat element buffer under a new shape.  The
//           backend dispatcher's reshape creates a new Storage alias over the
//           same physical buffer; no data movement occurs.
// Backward: reshape the incoming gradient back to the original input shape
//           recorded in input_shapes_[0].  Because reshape is its own inverse
//           (element ordering in a contiguous buffer is unchanged), the
//           backward pass is simply another reshape call targeting the saved
//           input shape.
//
// Invariants:
//   input_shapes_[0] — the shape of the input tensor before the reshape.
//   out_shape_       — the shape after the reshape (inherited from FuncOp).
class LUCID_API ViewBackward : public FuncOp<ViewBackward, 1> {
public:
    static const OpSchema schema_v1;
    // Reshape grad_out from out_shape_ back to input_shapes_[0].
    std::vector<Storage> apply(Storage grad_out) override;
};

// Reshape tensor `a` to `new_shape`.  Exactly one element of `new_shape` may
// be -1, in which case that dimension is inferred so that the total element
// count is preserved.  Raises ShapeMismatch if the total element count does
// not match after inference.  The input must be contiguous; a non-contiguous
// tensor should be passed through contiguous_op first.
LUCID_API TensorImplPtr reshape_op(const TensorImplPtr& a,
                                   const std::vector<std::int64_t>& new_shape);

// Remove the dimension at `dim`, which must have size 1.  Raises an index
// error if `dim` is out of range and a ValueError if the targeted dimension
// is not size 1.  Negative `dim` wraps relative to the input rank.
LUCID_API TensorImplPtr squeeze_op(const TensorImplPtr& a, int dim);

// Remove all size-1 dimensions from `a`.  If no size-1 dimensions exist, the
// output is a view of the input with the same shape.
LUCID_API TensorImplPtr squeeze_all_op(const TensorImplPtr& a);

// Insert a new size-1 dimension before position `dim` in the output.
// Negative values wrap relative to the output rank (ndim + 1), so dim=-1
// inserts before the last existing dimension.
LUCID_API TensorImplPtr unsqueeze_op(const TensorImplPtr& a, int dim);

}  // namespace lucid
