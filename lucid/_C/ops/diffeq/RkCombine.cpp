// lucid/_C/ops/diffeq/RkCombine.cpp
//
// Implements the fused Runge-Kutta stage combination and its backward node.
//
//   RkCombineBackward — routes the output gradient straight to `y0` and a
//                       `dt * coeffs[i]`-scaled copy to each stage input.
//
// The node derives from VariadicKernel because the stage count is a runtime
// property of the Butcher tableau, so the fixed-arity AutogradNode<Derived,N>
// (which stores its edges in a compile-time std::array) cannot express it.

#include "RkCombine.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/OpSchema.h"
#include "../../core/Scope.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/VariadicKernel.h"
#include "../bfunc/Mul.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Contiguous.h"
#include "Operand.h"

namespace lucid {

namespace {

// Backward node for rk_combine_op.
//
// Invariants:
//   scales_ — `dt * coeffs[i]` folded into a single factor per stage input,
//             computed once at forward time.  Its length equals the number
//             of `ks` inputs, so `scales_.size() + 1` gradients are returned.
//
// Backward formula: the forward is affine in every input, so the Jacobian is
// constant.  `dL/dy0 = g` (a clone, because the engine may not alias the same
// buffer into several gradient slots) and `dL/dk_i = dt * coeffs[i] * g`.
// Gradients are returned in input order — `y0` first, then the stages.
class RkCombineBackward : public kernel::VariadicKernel<RkCombineBackward> {
public:
    static const OpSchema schema_v1;

    std::vector<double> scales_;

    std::string node_name() const override { return std::string(schema_v1.name); }

    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t n = shape_numel(out_shape_);
        std::vector<Storage> grads;
        grads.reserve(scales_.size() + 1);
        grads.push_back(clone_storage(grad_out, n, dtype_, device_));
        for (const double scale : scales_)
            grads.push_back(mul_scalar_storage(grad_out, scale, n, dtype_, device_));
        return grads;
    }

    // Graph-recording backward, so a fused step stays as differentiable as
    // the unfused `y + dt * c * k` spelling it replaces — without this the
    // fusion would quietly cost the caller second-order gradients.
    //
    // The scale is materialised as a 0-D tensor and applied with `mul_op`
    // rather than a raw scalar kernel: that is the same route Python's
    // `tensor * scalar` takes, so the resulting node is differentiable in
    // turn.  `dL/dy0` is the identity, so `grad_out` is forwarded as-is.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        std::vector<TensorImplPtr> grads;
        grads.reserve(scales_.size() + 1);
        grads.push_back(grad_out);
        for (const double scale : scales_) {
            auto factor = full_op(Shape{}, scale, grad_out->dtype(), grad_out->device());
            grads.push_back(mul_op(grad_out, factor));
        }
        return grads;
    }
};

const OpSchema RkCombineBackward::schema_v1{"rk_combine",     1, AmpPolicy::Promote,
                                            /*det=*/true,
                                            /*note=*/"",
                                            /*in_arity=*/-1,
                                            /*out_arity=*/1,
                                            /*stable_ins=*/{}};

LUCID_REGISTER_OP(RkCombineBackward)

}  // namespace

// Validate that every stage tensor matches `y0` in shape, dtype, and device,
// then accumulate `y0 + dt * sum_i coeffs[i] * ks[i]` through the backend
// dispatcher and attach RkCombineBackward.  Stage terms whose folded scale is
// exactly zero contribute nothing and are skipped — Butcher rows are strictly
// lower triangular, so the zeros are the common case rather than the corner.
TensorImplPtr rk_combine_op(const TensorImplPtr& y0,
                            const std::vector<TensorImplPtr>& ks,
                            const std::vector<double>& coeffs,
                            double dt) {
    Validator::input(y0, "rk_combine.y0").non_null();
    if (ks.size() != coeffs.size())
        ErrorBuilder("rk_combine").fail("ks and coeffs must have the same length");

    const Dtype dtype = y0->dtype();
    const Device device = y0->device();
    const Shape shape = y0->shape();

    const diffeq::OperandSpec spec = diffeq::OperandSpec::from(y0, "rk_combine");
    for (std::size_t i = 0; i < ks.size(); ++i)
        diffeq::check_operand(ks[i], "rk_combine.ks[" + std::to_string(i) + "]", spec);

    OpScopeFull scope{"rk_combine", device, dtype, shape};
    scope.set_attr("stages", static_cast<std::int64_t>(ks.size()));

    // The backend arithmetic below is stride-agnostic, so any view input has
    // to be materialised first.  contiguous_op is a no-op for tensors that
    // already are contiguous, and wires its own backward when they are not.
    const TensorImplPtr y0_c = y0->is_contiguous() ? y0 : contiguous_op(y0);
    std::vector<TensorImplPtr> inputs;
    inputs.reserve(ks.size() + 1);
    inputs.push_back(y0_c);
    for (const auto& k : ks)
        inputs.push_back(k->is_contiguous() ? k : contiguous_op(k));

    backend::IBackend& dev_backend = backend::Dispatcher::for_device(device);
    Storage acc;
    bool has_acc = false;
    std::vector<double> scales;
    scales.reserve(ks.size());
    for (std::size_t i = 0; i < ks.size(); ++i) {
        const double scale = dt * coeffs[i];
        scales.push_back(scale);
        if (scale == 0.0)
            continue;
        Storage term = dev_backend.mul_scalar(inputs[i + 1]->storage(), shape, dtype, scale);
        acc = has_acc ? dev_backend.add(acc, term, shape, dtype) : std::move(term);
        has_acc = true;
    }

    Storage out_storage = has_acc
                              ? dev_backend.add(y0_c->storage(), acc, shape, dtype)
                              : clone_storage(y0_c->storage(), shape_numel(shape), dtype, device);
    auto out = helpers::fresh(std::move(out_storage), shape, dtype, device);

    auto bwd = std::make_shared<RkCombineBackward>();
    bwd->scales_ = std::move(scales);
    kernel::VariadicKernel<RkCombineBackward>::wire_autograd(std::move(bwd), inputs, out,
                                                             /*save_ins=*/false);
    return out;
}

}  // namespace lucid
