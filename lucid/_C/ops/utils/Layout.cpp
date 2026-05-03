// lucid/_C/ops/utils/Layout.cpp
//
// Implements flatten, broadcast_to, and expand.
//
// flatten delegates entirely to reshape_op: it computes the product of
// shape[start_axis..end_axis], builds a new shape vector with that product
// replacing the flattened range, and forwards to reshape_op.
//
// broadcast_to validates that `a` can be broadcast to the requested shape by
// left-padding with 1s and checking that each dimension is either equal or
// equal to 1 in the padded input.  A non-contiguous input is materialised
// before broadcasting.  BroadcastBackward reduces the gradient back by
// summing over all axes that were broadcast via Dispatcher::reduce_broadcast.

#include "Layout.h"

#include <algorithm>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "Contiguous.h"
#include "View.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;
using utils_detail::numel;

}  // namespace

// Collapse dimensions [start_axis, end_axis] (inclusive, after negative-index
// normalisation) into a single dimension whose size is their product.  The
// resulting shape has the dimensions before `s` unchanged, the single flat
// dimension, and then the dimensions after `e`.  Delegates to reshape_op so
// that the autograd node (ViewBackward) is correctly wired.
TensorImplPtr flatten_op(const TensorImplPtr& a, int start_axis, int end_axis) {
    Validator::input(a, "flatten.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    int s = start_axis < 0 ? start_axis + ndim : start_axis;
    int e = end_axis < 0 ? end_axis + ndim : end_axis;
    if (s < 0 || e >= ndim || s > e)
        ErrorBuilder("flatten").fail("invalid axis range");

    std::vector<std::int64_t> new_shape;
    for (int d = 0; d < s; ++d)
        new_shape.push_back(a->shape()[d]);
    std::int64_t flat = 1;
    for (int d = s; d <= e; ++d)
        flat *= a->shape()[d];
    new_shape.push_back(flat);
    for (int d = e + 1; d < ndim; ++d)
        new_shape.push_back(a->shape()[d]);
    return reshape_op(a, new_shape);
}

// Schema for BroadcastBackward.  The trailing true marks this op as a potential
// view: the backend may return an aliased (stride-zero) buffer for the forward
// broadcast rather than allocating a physically expanded copy.
const OpSchema BroadcastBackward::schema_v1{"broadcast_to", 1, AmpPolicy::KeepInput, true};

namespace {

// Thin dispatch helper: asks the backend to reduce the broadcast gradient
// over every dimension that was expanded during the forward broadcast.
// `input_shape` and `output_shape` together determine which axes were
// broadcast (any axis where padded input was 1 but output was > 1, and any
// leading axis that was prepended).
Storage reduce_broadcast(const Storage& grad,
                         const Shape& input_shape,
                         const Shape& output_shape,
                         Dtype dt,
                         Device device) {
    return backend::Dispatcher::for_device(device).reduce_broadcast(grad, input_shape, output_shape,
                                                                    dt);
}

}  // namespace

// Sum the output gradient back over every broadcast dimension so its shape
// matches the original input shape stored in input_shape_.  The reduction
// set is computed by the backend from input_shape_ vs. output_shape_.
std::vector<Storage> BroadcastBackward::apply(Storage grad_out) {
    return {reduce_broadcast(grad_out, input_shape_, output_shape_, dtype_, device_)};
}

LUCID_REGISTER_OP(BroadcastBackward)

// Ensure the input is contiguous (materialise if needed), left-pad its shape
// with 1s to match the target rank, verify that each dimension is either
// equal or 1 in the padded input, then call Dispatcher::broadcast to produce
// the expanded storage.  Attaches BroadcastBackward so that the gradient can
// be summed back over the broadcast dimensions.
//
// Raises ShapeMismatch if:
//   - the input rank is greater than the target rank (cannot broadcast down), or
//   - any dimension is neither equal nor 1 in the padded input shape.
TensorImplPtr broadcast_to_op(const TensorImplPtr& a, const Shape& shape) {
    Validator::input(a, "broadcast_to.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"broadcast_to", device, dt, shape};

    // Capture input and output shapes upfront so that the lambda closes over
    // stable values even if a temporary contiguous copy is created below.
    auto build_with_grad = [&](Storage&& out_storage) {
        auto out = std::make_shared<TensorImpl>(std::move(out_storage), shape, dt, device, false);
        auto bwd = std::make_shared<BroadcastBackward>();
        bwd->input_shape_ = a->shape();
        bwd->output_shape_ = shape;
        kernel::NaryKernel<BroadcastBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        return out;
    };

    // A non-contiguous input (e.g. a transposed view) cannot be broadcast
    // directly by the backend; materialise a dense copy first so the stride
    // metadata is guaranteed to be trivial before the broadcast kernel runs.
    const TensorImplPtr a_c = a->is_contiguous() ? a : contiguous_op(a);
    const std::size_t nin = a_c->shape().size();
    const std::size_t nout = shape.size();
    if (nin > nout)
        throw ShapeMismatch(shape, a_c->shape(), "broadcast_to");
    Shape padded(nout, 1);
    std::copy(a_c->shape().begin(), a_c->shape().end(), padded.begin() + (nout - nin));
    for (std::size_t d = 0; d < nout; ++d) {
        if (padded[d] != shape[d] && padded[d] != 1)
            throw ShapeMismatch(shape, a_c->shape(), "broadcast_to");
    }

    Storage out_storage =
        backend::Dispatcher::for_device(device).broadcast(a_c->storage(), a_c->shape(), shape, dt);
    return build_with_grad(std::move(out_storage));
}

// expand_op is a direct alias for broadcast_to_op, provided for API symmetry
// with PyTorch's Tensor::expand.  Both functions behave identically.
TensorImplPtr expand_op(const TensorImplPtr& a, const Shape& shape) {
    return broadcast_to_op(a, shape);
}

}  // namespace lucid
