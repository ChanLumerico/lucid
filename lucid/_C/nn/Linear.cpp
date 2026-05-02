#include "Linear.h"

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

const OpSchema LinearBackward::schema_v1{"linear", 1, AmpPolicy::Promote, true};

namespace {

struct FlatX {
    std::size_t M;
    std::size_t K;
};

FlatX flatten_x(const Shape& x_shape) {
    if (x_shape.empty()) {
        ErrorBuilder("linear").fail("x must be at least 1-D");
    }
    std::size_t m = 1;
    for (std::size_t d = 0; d + 1 < x_shape.size(); ++d) {
        m *= static_cast<std::size_t>(x_shape[d]);
    }
    return {m, static_cast<std::size_t>(x_shape.back())};
}

}  // namespace

TensorImplPtr
LinearBackward::forward(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b) {
    Validator::input(x, "linear.x").non_null();
    Validator::input(W, "linear.W").non_null().ndim(2);
    Validator::input(b, "linear.b").non_null().ndim(1);
    Validator::pair(x, W, "linear").same_dtype().same_device();
    Validator::pair(x, b, "linear").same_dtype().same_device();

    const auto fx = flatten_x(x->shape());
    const std::size_t M = fx.M;
    const std::size_t K = fx.K;
    const std::size_t N = static_cast<std::size_t>(W->shape()[0]);

    if (W->shape()[1] != static_cast<std::int64_t>(K))
        throw ShapeMismatch(W->shape(), x->shape(), "linear: W.shape[1] != x.last_dim");
    if (b->shape()[0] != static_cast<std::int64_t>(N))
        throw ShapeMismatch(b->shape(), W->shape(), "linear: b.shape[0] != W.shape[0]");

    Shape out_shape = x->shape();
    out_shape.back() = static_cast<std::int64_t>(N);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), out_shape};
    scope.set_flops(static_cast<std::int64_t>(2 * M * N * K));

    Storage out_storage = backend::Dispatcher::for_device(x->device())
                              .linear(x->storage(), W->storage(), b->storage(), x->shape(),
                                      W->shape(), out_shape, x->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);

    kernel::NaryKernel<LinearBackward, 3>::wire_autograd({x, W, b}, out);
    return out;
}

std::vector<Storage> LinearBackward::apply(Storage grad_out) {
    return backend::Dispatcher::for_device(device_).linear_backward(
        grad_out, saved_inputs_[0], saved_inputs_[1], input_shapes_[0], input_shapes_[1],
        input_shapes_[2], dtype_);
}

TensorImplPtr linear_op(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b) {
    return LinearBackward::forward(x, W, b);
}

LUCID_REGISTER_OP(LinearBackward)

}  // namespace lucid
