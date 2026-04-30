#include "Repeat.h"

#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
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
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;
using utils_detail::wrap_axis;

class RepeatBackward : public FuncOp<RepeatBackward, 1> {
public:
    static const OpSchema schema_v1;

    int axis_ = 0;
    std::int64_t repeats_ = 1;

    std::vector<Storage> apply(Storage grad_out) override {
        return {backend::Dispatcher::for_device(device_).repeat_backward(
            grad_out, input_shapes_[0], out_shape_, axis_, repeats_, dtype_)};
    }
};

const OpSchema RepeatBackward::schema_v1{"repeat", 1, AmpPolicy::KeepInput, true};

class TileBackward : public FuncOp<TileBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape padded_shape_;
    std::vector<std::int64_t> reps_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {backend::Dispatcher::for_device(device_).tile_backward(
            grad_out, input_shapes_[0], padded_shape_, out_shape_, reps_, dtype_)};
    }
};

const OpSchema TileBackward::schema_v1{"tile", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_repeat_grad(const TensorImplPtr& a,
                                 TensorImplPtr out,
                                 int axis,
                                 std::int64_t repeats) {
    auto bwd = std::make_shared<RepeatBackward>();
    bwd->axis_ = axis;
    bwd->repeats_ = repeats;
    kernel::NaryKernel<RepeatBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                         /*save_ins=*/false);
    return out;
}

TensorImplPtr attach_tile_grad(const TensorImplPtr& a,
                               TensorImplPtr out,
                               Shape padded_shape,
                               std::vector<std::int64_t> reps) {
    auto bwd = std::make_shared<TileBackward>();
    bwd->padded_shape_ = std::move(padded_shape);
    bwd->reps_ = std::move(reps);
    kernel::NaryKernel<TileBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                       /*save_ins=*/false);
    return out;
}

LUCID_REGISTER_OP(RepeatBackward)
LUCID_REGISTER_OP(TileBackward)

}  // namespace

TensorImplPtr repeat_op(const TensorImplPtr& a, std::int64_t repeats, int axis) {
    Validator::input(a, "repeat.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"repeat", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    Shape out_shape = a->shape();
    out_shape[ax] *= repeats;
    Storage out_storage =
        backend::Dispatcher::for_device(device).repeat(a->storage(), a->shape(), dt, repeats, ax);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    return attach_repeat_grad(a, std::move(result), ax, repeats);
}

TensorImplPtr tile_op(const TensorImplPtr& a, std::vector<std::int64_t> reps) {
    Validator::input(a, "tile.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"tile", device, dt, a->shape()};
    const std::size_t nout = reps.size();
    if (nout < a->shape().size())
        ErrorBuilder("tile").fail("reps must be at least as long as ndim");

    Shape padded(nout, 1);
    const std::size_t lead = nout - a->shape().size();
    for (std::size_t d = 0; d < a->shape().size(); ++d)
        padded[lead + d] = a->shape()[d];
    Shape out_shape(nout);
    for (std::size_t d = 0; d < nout; ++d)
        out_shape[d] = padded[d] * reps[d];

    Storage out_storage =
        backend::Dispatcher::for_device(device).tile(a->storage(), a->shape(), dt, reps);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    return attach_tile_grad(a, std::move(result), std::move(padded), std::move(reps));
}

}  // namespace lucid
