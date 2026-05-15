// lucid/_C/ops/ufunc/Var.cpp
//
// Biased variance reduction (denominator = N, not N-1) along arbitrary axes.
//
// Forward: delegates to IBackend::variance, which uses Accelerate on CPU and
// MLX on GPU.
//
// Backward formula (biased variance):
//   dL/dx_i = (2 / N) * (x_i - mean(x)) * broadcast(dL/dy)
// where N is the product of the reduced dimension sizes and mean(x) is
// precomputed during the forward pass and saved on the backward node to avoid
// a second mean reduction at backward time.
//
// VarBackward is defined in an anonymous namespace because its interface is not
// part of the public API; var_op is the only external entry point.

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
#include "../bfunc/_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::fresh;

// Private backward node for variance.
//
// Saved state:
//   input_shape_ — original input shape, needed to broadcast the gradient.
//   axes_        — normalised reduction axes.
//   keepdims_    — whether collapsed axes were kept as size-1 dims.
//   count_       — number of elements reduced (N); guards against empty axes.
//   saved_input_ — a copy of the forward input, used to compute (x - mean).
//   saved_mean_  — mean(x) already broadcast to the input shape, so apply()
//                  can subtract it directly without an extra broadcast call.
class VarBackward : public AutogradNode<VarBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    std::vector<int> axes_;
    bool keepdims_;
    std::int64_t count_;
    Storage saved_input_;
    Storage saved_mean_;

    // dL/dx = (2/N) * (x - mean) * broadcast_back(dL/dy).
    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t n = shape_numel(input_shape_);
        // (x - mean): mean is already broadcast to input shape from forward.
        Storage centered = subtract_storages(saved_input_, saved_mean_, n, dtype_, device_);
        // Expand the reduced gradient back to the full input shape.
        Storage g_b = broadcast_back_for_reduce(grad_out, out_shape_, input_shape_, axes_,
                                                keepdims_, dtype_, device_);
        Storage scaled = mul_scalar_storage(g_b, 2.0 / (double)count_, n, dtype_, device_);
        Storage dx = multiply_storages(centered, scaled, n, dtype_, device_);
        return {std::move(dx)};
    }
};

// KeepInput: variance is computed in the input's own dtype (the backend
// internally promotes to float if needed).
const OpSchema VarBackward::schema_v1{"var", 1, AmpPolicy::KeepInput, true};

}  // namespace

// Compute the biased variance, then build VarBackward when autograd is active.
// The mean is computed with keepdims=true so that it can be passed to
// IBackend::broadcast to expand it to the input shape in one call.
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

    // Count the number of elements collapsed; clamp to 1 to avoid division by
    // zero for empty-axis edge cases.
    std::size_t reduced = 1;
    for (auto ax : axes)
        reduced *= static_cast<std::size_t>(a->shape()[ax]);
    if (reduced == 0)
        reduced = 1;

    if (GradMode::is_enabled() && a->requires_grad()) {
        // Compute mean with keepdims=true, then broadcast to full input shape
        // so that VarBackward::apply can compute (x - mean) without a second
        // broadcast.
        Shape mean_keepdims_shape = reduce_output_shape(a->shape(), axes, true);
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
        // save_output=false: VarBackward saves the input and mean manually above.
        kernel::NaryKernel<VarBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

}  // namespace lucid
