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

namespace lucid {

// =================== Relu ===================
const OpSchema ReluBackward::schema_v1{"relu", 1, AmpPolicy::KeepInput, true};

Storage ReluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = positive_mask_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr relu_op(const TensorImplPtr& a) {
    return ReluBackward::forward(a);
}
LUCID_REGISTER_OP(ReluBackward)

// =================== Sigmoid ===================
const OpSchema SigmoidBackward::schema_v1{"sigmoid", 1, AmpPolicy::Promote, true};

Storage SigmoidBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = z (1 - z) g, z = saved output.
    Storage neg_z = mul_scalar_storage(saved_output_, -1.0, n, dtype_, device_);
    Storage one_m_z = add_scalar_storage(neg_z, 1.0, n, dtype_, device_);
    Storage z_omz = multiply_storages(saved_output_, one_m_z, n, dtype_, device_);
    return multiply_storages(z_omz, g, n, dtype_, device_);
}

TensorImplPtr sigmoid_op(const TensorImplPtr& a) {
    return SigmoidBackward::forward(a);
}
LUCID_REGISTER_OP(SigmoidBackward)

// =================== SiLU (Swish) ===================
const OpSchema SiluBackward::schema_v1{"silu", 1, AmpPolicy::Promote, true};

Storage SiluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = σ(x) · (1 + x(1 - σ(x)))
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

// =================== GeLU ===================
const OpSchema GeluBackward::schema_v1{"gelu", 1, AmpPolicy::ForceFP32, true};

Storage GeluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).gelu_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr gelu_op(const TensorImplPtr& a) {
    return GeluBackward::forward(a);
}
LUCID_REGISTER_OP(GeluBackward)

// =================== LeakyReLU (scalar param — custom forward) ===================
const OpSchema LeakyReluBackward::schema_v1{"leaky_relu", 1, AmpPolicy::KeepInput, true};

// LeakyReluBackward::cpu_kernel was removed: forward() routes through
// Dispatcher (be.leaky_relu), so the cpu_kernel path is unreachable dead code.

Storage LeakyReluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = leaky_mask_storage(saved_inputs_[0], slope_, n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

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

// =================== Softplus ===================
const OpSchema SoftplusBackward::schema_v1{"softplus", 1, AmpPolicy::ForceFP32, true};

Storage SoftplusBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage sx = sigmoid_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(sx, g, n, dtype_, device_);
}

TensorImplPtr softplus_op(const TensorImplPtr& a) {
    return SoftplusBackward::forward(a);
}
LUCID_REGISTER_OP(SoftplusBackward)

// =================== ELU (scalar param — custom forward) ===================
const OpSchema EluBackward::schema_v1{"elu", 1, AmpPolicy::ForceFP32, true};

// EluBackward::cpu_kernel was removed: forward() routes through
// Dispatcher (be.elu), so the cpu_kernel path is unreachable dead code.

Storage EluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).elu_backward(saved_inputs_[0], g, out_shape_,
                                                                 dtype_, alpha_);
}

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

// =================== SELU ===================
const OpSchema SeluBackward::schema_v1{"selu", 1, AmpPolicy::ForceFP32, true};

Storage SeluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).selu_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr selu_op(const TensorImplPtr& a) {
    return SeluBackward::forward(a);
}
LUCID_REGISTER_OP(SeluBackward)

// =================== Mish ===================
const OpSchema MishBackward::schema_v1{"mish", 1, AmpPolicy::ForceFP32, true};

Storage MishBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).mish_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr mish_op(const TensorImplPtr& a) {
    return MishBackward::forward(a);
}
LUCID_REGISTER_OP(MishBackward)

// =================== HardSigmoid ===================
const OpSchema HardSigmoidBackward::schema_v1{"hard_sigmoid", 1, AmpPolicy::KeepInput, true};

Storage HardSigmoidBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).hard_sigmoid_backward(saved_inputs_[0], g,
                                                                          out_shape_, dtype_);
}

TensorImplPtr hard_sigmoid_op(const TensorImplPtr& a) {
    return HardSigmoidBackward::forward(a);
}
LUCID_REGISTER_OP(HardSigmoidBackward)

// =================== HardSwish ===================
const OpSchema HardSwishBackward::schema_v1{"hard_swish", 1, AmpPolicy::KeepInput, true};

Storage HardSwishBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).hard_swish_backward(saved_inputs_[0], g,
                                                                        out_shape_, dtype_);
}

TensorImplPtr hard_swish_op(const TensorImplPtr& a) {
    return HardSwishBackward::forward(a);
}
LUCID_REGISTER_OP(HardSwishBackward)

// =================== ReLU6 ===================
const OpSchema Relu6Backward::schema_v1{"relu6", 1, AmpPolicy::KeepInput, true};

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
