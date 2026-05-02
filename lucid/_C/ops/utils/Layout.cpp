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
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "Contiguous.h"          // contiguous_op for non-contig inputs
#include "View.h"                // reshape_op / ViewBackward
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;
using utils_detail::numel;

}  // namespace

TensorImplPtr flatten_op(const TensorImplPtr& a, int start_axis, int end_axis) {
    Validator::input(a, "flatten.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    int s = start_axis < 0 ? start_axis + ndim : start_axis;
    int e = end_axis < 0 ? end_axis + ndim : end_axis;
    if (s < 0 || e >= ndim || s > e)
        ErrorBuilder("flatten").fail("invalid axis range");

    // Delegate to reshape_op so we inherit the ViewBackward autograd wiring.
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

const OpSchema BroadcastBackward::schema_v1{"broadcast_to", 1, AmpPolicy::KeepInput, true};

namespace {

// Sum `grad` (shape = output_shape) down to `input_shape` by reducing along
// the broadcast axes. Handles right-aligned shape padding (PyTorch / NumPy
// semantics).
Storage reduce_broadcast(const Storage& grad,
                         const Shape& input_shape,
                         const Shape& output_shape,
                         Dtype dt,
                         Device device) {
    return backend::Dispatcher::for_device(device).reduce_broadcast(grad, input_shape, output_shape,
                                                                    dt);
}

}  // namespace

std::vector<Storage> BroadcastBackward::apply(Storage grad_out) {
    return {reduce_broadcast(grad_out, input_shape_, output_shape_, dtype_, device_)};
}

LUCID_REGISTER_OP(BroadcastBackward)

TensorImplPtr broadcast_to_op(const TensorImplPtr& a, const Shape& shape) {
    Validator::input(a, "broadcast_to.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"broadcast_to", device, dt, shape};

    auto build_with_grad = [&](Storage&& out_storage) {
        auto out = std::make_shared<TensorImpl>(std::move(out_storage), shape, dt, device,
                                                /*requires_grad=*/false);
        auto bwd = std::make_shared<BroadcastBackward>();
        bwd->input_shape_ = a->shape();
        bwd->output_shape_ = shape;
        kernel::NaryKernel<BroadcastBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                                /*save_ins=*/false);
        return out;
    };

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

TensorImplPtr expand_op(const TensorImplPtr& a, const Shape& shape) {
    return broadcast_to_op(a, shape);
}

}  // namespace lucid
