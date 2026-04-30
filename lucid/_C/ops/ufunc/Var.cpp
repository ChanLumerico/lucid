#include "Var.h"

#include <vector>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::fresh;

// VarBackward: dx = (2/N) * (x - mean) * broadcast(grad)
class VarBackward : public AutogradNode<VarBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    std::vector<int> axes_;
    bool keepdims_;
    std::int64_t count_;
    Storage saved_input_;
    Storage saved_mean_;  // broadcast to input_shape_

    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t n = shape_numel(input_shape_);
        Storage centered = subtract_storages(saved_input_, saved_mean_, n, dtype_, device_);
        Storage g_b = broadcast_back_for_reduce(grad_out, out_shape_, input_shape_, axes_,
                                                keepdims_, dtype_, device_);
        Storage scaled = mul_scalar_storage(g_b, 2.0 / (double)count_, n, dtype_, device_);
        Storage dx = multiply_storages(centered, scaled, n, dtype_, device_);
        return {std::move(dx)};
    }
};

const OpSchema VarBackward::schema_v1{"var", 1, AmpPolicy::KeepInput, true};

}  // namespace

TensorImplPtr var_op(const TensorImplPtr& a, const std::vector<int>& axes_user, bool keepdims) {
    Validator::input(a, "var.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    const auto axes = normalize_axes(axes_user, static_cast<int>(a->shape().size()));
    const Shape out_shape = reduce_output_shape(a->shape(), axes, keepdims);
    OpScopeFull scope{"var", device, dt, out_shape};
    backend::ReduceOpts reduce_opts{axes, keepdims};
    auto& be = backend::Dispatcher::for_device(device);

    Storage out_storage = be.variance(a->storage(), a->shape(), reduce_opts, dt);
    TensorImplPtr out = fresh(std::move(out_storage), out_shape, dt, device);
    std::size_t reduced = 1;
    for (auto ax : axes)
        reduced *= static_cast<std::size_t>(a->shape()[ax]);
    if (reduced == 0)
        reduced = 1;

    if (GradMode::is_enabled() && a->requires_grad()) {
        Shape mean_keepdims_shape = reduce_output_shape(a->shape(), axes, /*keepdims=*/true);
        Storage mean_keepdims =
            be.reduce_mean(a->storage(), a->shape(), backend::ReduceOpts{axes, true}, dt);
        Storage mean_storage = be.broadcast(mean_keepdims, mean_keepdims_shape, a->shape(), dt);

        auto bwd = std::make_shared<VarBackward>();
        bwd->input_shape_ = a->shape();
        bwd->axes_ = axes;
        bwd->keepdims_ = keepdims;
        bwd->count_ = static_cast<std::int64_t>(reduced);
        bwd->saved_input_ = a->storage();
        bwd->saved_mean_ = std::move(mean_storage);
        kernel::NaryKernel<VarBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                          /*save_ins=*/false);
    }
    return out;
}

}  // namespace lucid
