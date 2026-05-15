// lucid/_C/ops/bfunc/Add.cpp
//
// Implements AddBackward::grad_formula and the add_op free function.  The
// forward pass is handled entirely by BinaryKernel<AddBackward>::forward.

#include "Add.h"

#include <utility>

#include <mlx/ops.h>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../../core/Shape.h"

namespace lucid {

// AmpPolicy::Promote: if one operand is F32 and the other is F64, both are
// upcast to F64 before the kernel runs.
const OpSchema AddBackward::schema_v1{"add", 1, AmpPolicy::Promote, true};

// Gradient of element-wise addition.
//
// Because addition distributes the gradient equally to both inputs, both dA
// and dB start as full copies of grad_out at the broadcast output shape.
// reduce_grad_to_shape (called by BinaryKernel::apply, which wraps this
// function) then sums over any dimensions that were broadcast-expanded to
// recover tensors shaped like the original inputs.
//
// Note: both reduce_grad_to_shape calls receive (out_shape_, out_shape_, …)
// rather than the original input shapes; the reduction to the actual input
// shape is performed one level up in BinaryKernel<Derived>::apply.
std::pair<Storage, Storage> AddBackward::grad_formula(const Storage& grad_out) {
    return {
        reduce_grad_to_shape(grad_out, out_shape_, out_shape_, dtype_, device_),
        reduce_grad_to_shape(grad_out, out_shape_, out_shape_, dtype_, device_),
    };
}

// Thin wrapper: delegates to AddBackward::forward which is provided by
// BinaryKernel<AddBackward>.
TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return AddBackward::forward(a, b);
}

LUCID_REGISTER_OP(AddBackward)

}  // namespace lucid
