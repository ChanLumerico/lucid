// lucid/_C/ops/ufunc/Activation.cpp
//
// Implementations of activation backward nodes and forward entry points.
// Activations with complex backward formulas (gelu, elu, selu, mish,
// hard_sigmoid, hard_swish) delegate to the backend dispatcher so that
// platform-specific implementations (Accelerate on CPU, MLX on GPU) can be
// used without duplicating the math here.

#include "Activation.h"

#include <cmath>

#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
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
#include "../bfunc/Add.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"
#include "Arith.h"

namespace lucid {

// relu — AmpPolicy::KeepInput: ReLU is valid on integer types.
const OpSchema ReluBackward::schema_v1{"relu", 1, AmpPolicy::KeepInput, true};

// dL/dx = (x > 0) * dL/dy: zero out gradient where x was non-positive.
Storage ReluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = positive_mask_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr ReluBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    // mask = (x > 0): sign(relu(x)) gives 0 for x<=0 and 1 for x>0
    auto mask = sign_op(relu_op(x));
    return mul_op(g, mask);
}

TensorImplPtr relu_op(const TensorImplPtr& a) {
    return ReluBackward::forward(a);
}
LUCID_REGISTER_OP(ReluBackward)

// sigmoid — AmpPolicy::Promote: output requires float for stability.
const OpSchema SigmoidBackward::schema_v1{"sigmoid", 1, AmpPolicy::Promote, true};

// dL/dx = z*(1-z) * dL/dy  where z = saved_output_.
// Building (1-z) via mul_scalar(-1) + add_scalar(1) avoids a dedicated
// subtract kernel, reusing storage primitives already in the hot path.
Storage SigmoidBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage neg_z = mul_scalar_storage(saved_output_, -1.0, n, dtype_, device_);
    Storage one_m_z = add_scalar_storage(neg_z, 1.0, n, dtype_, device_);
    Storage z_omz = multiply_storages(saved_output_, one_m_z, n, dtype_, device_);
    return multiply_storages(z_omz, g, n, dtype_, device_);
}

TensorImplPtr SigmoidBackward::grad_formula_impl(const TensorImplPtr& g,
                                                 const TensorImplPtr&,
                                                 const TensorImplPtr& out) {
    // dx = out*(1-out)*g
    auto one_minus_out = sub_op(ones_like_op(out), out);
    return mul_op(mul_op(out, one_minus_out), g);
}

TensorImplPtr sigmoid_op(const TensorImplPtr& a) {
    return SigmoidBackward::forward(a);
}
LUCID_REGISTER_OP(SigmoidBackward)

// silu — AmpPolicy::Promote; gradient derived analytically from y = x*σ(x).
const OpSchema SiluBackward::schema_v1{"silu", 1, AmpPolicy::Promote, true};

// dL/dx = σ(x) * (1 + x*(1 - σ(x))) * dL/dy.
// Step-by-step using storage primitives:
//   sx       = sigmoid(x)
//   (1-sx)   = -sx + 1
//   x*(1-sx) = x * (1-sx)
//   (1+…)    = x*(1-sx) + 1
//   dx       = sx * (1 + x*(1-sx))
Storage SiluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage sx = sigmoid_storage(saved_inputs_[0], n, dtype_, device_);
    Storage neg_sx = mul_scalar_storage(sx, -1.0, n, dtype_, device_);
    Storage one_m_sx = add_scalar_storage(neg_sx, 1.0, n, dtype_, device_);
    Storage x_omsx = multiply_storages(saved_inputs_[0], one_m_sx, n, dtype_, device_);
    Storage one_p = add_scalar_storage(x_omsx, 1.0, n, dtype_, device_);
    Storage dx = multiply_storages(sx, one_p, n, dtype_, device_);
    return multiply_storages(dx, g, n, dtype_, device_);
}

TensorImplPtr silu_op(const TensorImplPtr& a) {
    return SiluBackward::forward(a);
}
LUCID_REGISTER_OP(SiluBackward)

// gelu — ForceFP32 because the tanh approximation used inside the backend is
// not numerically safe in half precision.
const OpSchema GeluBackward::schema_v1{"gelu", 1, AmpPolicy::ForceFP32, true};

// Delegate to the backend; the exact formula involves the Gaussian CDF
// 0.5*(1 + erf(x/sqrt(2))) and its derivative.
Storage GeluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).gelu_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr gelu_op(const TensorImplPtr& a) {
    return GeluBackward::forward(a);
}
LUCID_REGISTER_OP(GeluBackward)

// leaky_relu — KeepInput (valid for integer slopes).
const OpSchema LeakyReluBackward::schema_v1{"leaky_relu", 1, AmpPolicy::KeepInput, true};

// dL/dx = (x > 0 ? 1 : slope_) * dL/dy  (leaky mask).
Storage LeakyReluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = leaky_mask_storage(saved_inputs_[0], slope_, n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

// Custom forward: the slope parameter must be captured on the backward node,
// which is not possible through the standard dispatch() path.
TensorImplPtr LeakyReluBackward::forward(const TensorImplPtr& a, double slope) {
    Validator::input(a, "leaky_relu.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .leaky_relu(a->storage(), a->shape(), a->dtype(), slope);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()));

    auto bwd = std::make_shared<LeakyReluBackward>();
    bwd->slope_ = slope;
    kernel::NaryKernel<LeakyReluBackward, 1>::wire_autograd(std::move(bwd), {a}, out);
    return out;
}

TensorImplPtr leaky_relu_op(const TensorImplPtr& a, double slope) {
    return LeakyReluBackward::forward(a, slope);
}
LUCID_REGISTER_OP(LeakyReluBackward)

// softplus — ForceFP32 because log(1 + e^x) overflows in float16.
const OpSchema SoftplusBackward::schema_v1{"softplus", 1, AmpPolicy::ForceFP32, true};

// dL/dx = sigmoid(x) * dL/dy  (derivative of log(1 + e^x) is sigmoid(x)).
Storage SoftplusBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage sx = sigmoid_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(sx, g, n, dtype_, device_);
}

TensorImplPtr softplus_op(const TensorImplPtr& a) {
    return SoftplusBackward::forward(a);
}
LUCID_REGISTER_OP(SoftplusBackward)

// elu — ForceFP32; backward depends on alpha_ captured in the node.
const OpSchema EluBackward::schema_v1{"elu", 1, AmpPolicy::ForceFP32, true};

// Delegate to the backend because the piecewise formula (1 if x>=0 else
// alpha*e^x) requires a conditional that the generic storage primitives do
// not express without an explicit branch kernel.
Storage EluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).elu_backward(saved_inputs_[0], g, out_shape_,
                                                                 dtype_, alpha_);
}

// Custom forward: alpha_ must be captured on the backward node.
TensorImplPtr EluBackward::forward(const TensorImplPtr& a, double alpha) {
    Validator::input(a, "elu.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .elu(a->storage(), a->shape(), a->dtype(), alpha);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()));

    auto bwd = std::make_shared<EluBackward>();
    bwd->alpha_ = alpha;
    kernel::NaryKernel<EluBackward, 1>::wire_autograd(std::move(bwd), {a}, out);
    return out;
}

TensorImplPtr elu_op(const TensorImplPtr& a, double alpha) {
    return EluBackward::forward(a, alpha);
}
LUCID_REGISTER_OP(EluBackward)

// selu — ForceFP32; fixed α/λ constants embedded in the backend.
const OpSchema SeluBackward::schema_v1{"selu", 1, AmpPolicy::ForceFP32, true};

// Delegate; the backend applies the standard SELU backward with its fixed
// self-normalising constants.
Storage SeluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).selu_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr selu_op(const TensorImplPtr& a) {
    return SeluBackward::forward(a);
}
LUCID_REGISTER_OP(SeluBackward)

// mish — ForceFP32; composed function requires tanh and softplus in backward.
const OpSchema MishBackward::schema_v1{"mish", 1, AmpPolicy::ForceFP32, true};

// Delegate; the backend computes d/dx[x*tanh(log(1+e^x))] which involves the
// product rule applied to tanh(softplus(x)).
Storage MishBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).mish_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr mish_op(const TensorImplPtr& a) {
    return MishBackward::forward(a);
}
LUCID_REGISTER_OP(MishBackward)

// hard_sigmoid — KeepInput; piecewise linear, valid for float inputs.
const OpSchema HardSigmoidBackward::schema_v1{"hard_sigmoid", 1, AmpPolicy::KeepInput, true};

// Delegate; the backend returns 1/6 inside the active region, 0 outside.
Storage HardSigmoidBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).hard_sigmoid_backward(saved_inputs_[0], g,
                                                                          out_shape_, dtype_);
}

TensorImplPtr hard_sigmoid_op(const TensorImplPtr& a) {
    return HardSigmoidBackward::forward(a);
}
LUCID_REGISTER_OP(HardSigmoidBackward)

// hard_swish — KeepInput; piecewise linear backward.
const OpSchema HardSwishBackward::schema_v1{"hard_swish", 1, AmpPolicy::KeepInput, true};

// Delegate; the backend handles the three-region piecewise formula.
Storage HardSwishBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).hard_swish_backward(saved_inputs_[0], g,
                                                                        out_shape_, dtype_);
}

TensorImplPtr hard_swish_op(const TensorImplPtr& a) {
    return HardSwishBackward::forward(a);
}
LUCID_REGISTER_OP(HardSwishBackward)

// relu6 — KeepInput; gradient is non-zero only in the open interval (0, 6).
const OpSchema Relu6Backward::schema_v1{"relu6", 1, AmpPolicy::KeepInput, true};

// dL/dx = (0 < x < 6) * dL/dy  using an in-range boolean mask.
Storage Relu6Backward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = in_range_mask_storage(saved_inputs_[0], 0.0, 6.0, n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr relu6_op(const TensorImplPtr& a) {
    return Relu6Backward::forward(a);
}
LUCID_REGISTER_OP(Relu6Backward)

}  // namespace lucid
