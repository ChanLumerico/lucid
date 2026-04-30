#include "Activation.h"

#include <cmath>

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/cpu/Vdsp.h"
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

namespace lucid {

namespace {
constexpr double kGeluC1 = 0.7978845608028654;  // sqrt(2/pi)
constexpr double kGeluC2 = 0.044715;

CpuStorage allocate_unary(const Shape& out_shape, Dtype dt) {
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = shape_numel(out_shape) * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    return out;
}

template <typename T>
inline T softplus_stable(T x) {
    const T ax = std::abs(x);
    return std::max(x, T{0}) + std::log1p(std::exp(-ax));
}
}  // namespace

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
    if (device_ == Device::GPU) {
        const auto& x_arr = *std::get<GpuStorage>(saved_inputs_[0]).arr;
        const auto& g_arr = *std::get<GpuStorage>(g).arr;
        auto mlx_dt = gpu::to_mlx_dtype(dtype_);
        ::mlx::core::array c1{static_cast<float>(kGeluC1), mlx_dt};
        ::mlx::core::array c2{static_cast<float>(kGeluC2), mlx_dt};
        ::mlx::core::array three{3.0f, mlx_dt};
        ::mlx::core::array half{0.5f, mlx_dt};
        ::mlx::core::array one{1.0f, mlx_dt};
        auto x2 = ::mlx::core::multiply(x_arr, x_arr);
        auto x3 = ::mlx::core::multiply(x2, x_arr);
        auto inner =
            ::mlx::core::multiply(c1, ::mlx::core::add(x_arr, ::mlx::core::multiply(c2, x3)));
        auto t = ::mlx::core::tanh(inner);
        auto dinner = ::mlx::core::multiply(
            c1, ::mlx::core::add(one, ::mlx::core::multiply(three, ::mlx::core::multiply(c2, x2))));
        auto t2 = ::mlx::core::multiply(t, t);
        auto term1 = ::mlx::core::multiply(half, ::mlx::core::add(one, t));
        auto term2 = ::mlx::core::multiply(
            half, ::mlx::core::multiply(
                      x_arr, ::mlx::core::multiply(::mlx::core::subtract(one, t2), dinner)));
        auto dx = ::mlx::core::add(term1, term2);
        auto out = ::mlx::core::multiply(dx, g_arr);
        return Storage{gpu::wrap_mlx_array(std::move(out), dtype_)};
    }
    const std::size_t n = shape_numel(out_shape_);
    const auto& x_cpu = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& g_cpu = std::get<CpuStorage>(g);
    CpuStorage out;
    out.dtype = dtype_;
    out.nbytes = n * dtype_size(dtype_);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    switch (dtype_) {
        case Dtype::F32: {
            auto* x = reinterpret_cast<const float*>(x_cpu.ptr.get());
            auto* gg = reinterpret_cast<const float*>(g_cpu.ptr.get());
            auto* qq = reinterpret_cast<float*>(out.ptr.get());
            const float c1 = static_cast<float>(kGeluC1);
            const float c2 = static_cast<float>(kGeluC2);
            for (std::size_t i = 0; i < n; ++i) {
                const float xi = x[i];
                const float inner = c1 * (xi + c2 * xi * xi * xi);
                const float t = std::tanh(inner);
                const float dinner = c1 * (1.f + 3.f * c2 * xi * xi);
                const float dx = 0.5f * (1.f + t) + 0.5f * xi * (1.f - t * t) * dinner;
                qq[i] = dx * gg[i];
            }
            break;
        }
        case Dtype::F64: {
            auto* x = reinterpret_cast<const double*>(x_cpu.ptr.get());
            auto* gg = reinterpret_cast<const double*>(g_cpu.ptr.get());
            auto* qq = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double xi = x[i];
                const double inner = kGeluC1 * (xi + kGeluC2 * xi * xi * xi);
                const double t = std::tanh(inner);
                const double dinner = kGeluC1 * (1.0 + 3.0 * kGeluC2 * xi * xi);
                const double dx = 0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * dinner;
                qq[i] = dx * gg[i];
            }
            break;
        }
        default:
            ErrorBuilder("gelu backward").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr gelu_op(const TensorImplPtr& a) {
    return GeluBackward::forward(a);
}
LUCID_REGISTER_OP(GeluBackward)

// =================== LeakyReLU (scalar param — custom forward) ===================
const OpSchema LeakyReluBackward::schema_v1{"leaky_relu", 1, AmpPolicy::KeepInput, true};

CpuStorage LeakyReluBackward::cpu_kernel(const CpuStorage& a,
                                         const Shape& out_shape,
                                         Dtype dt,
                                         double slope) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const auto fs = static_cast<float>(slope);
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] >= 0.f) ? p[i] : fs * p[i];
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] >= 0.0) ? p[i] : slope * p[i];
            break;
        }
        default:
            ErrorBuilder("leaky_relu").not_implemented("dtype not supported");
    }
    return out;
}

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

CpuStorage EluBackward::cpu_kernel(const CpuStorage& a,
                                   const Shape& out_shape,
                                   Dtype dt,
                                   double alpha) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const float fa = static_cast<float>(alpha);
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                q[i] = (x >= 0.f) ? x : fa * (std::exp(x) - 1.f);
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                q[i] = (x >= 0.0) ? x : alpha * (std::exp(x) - 1.0);
            }
            break;
        }
        default:
            ErrorBuilder("elu").not_implemented("dtype not supported");
    }
    return out;
}

Storage EluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_)
        .elu_backward(saved_inputs_[0], g, out_shape_, dtype_, alpha_);
}

TensorImplPtr EluBackward::forward(const TensorImplPtr& a, double alpha) {
    Validator::input(a, "elu.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
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
    return backend::Dispatcher::for_device(device_)
        .selu_backward(saved_inputs_[0], g, out_shape_, dtype_);
}

TensorImplPtr selu_op(const TensorImplPtr& a) {
    return SeluBackward::forward(a);
}
LUCID_REGISTER_OP(SeluBackward)

// =================== Mish ===================
const OpSchema MishBackward::schema_v1{"mish", 1, AmpPolicy::ForceFP32, true};

Storage MishBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(g);
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dtype_));
        auto pos = ::mlx::core::maximum(*gx.arr, zero);
        auto neg_abs = ::mlx::core::negative(::mlx::core::abs(*gx.arr));
        auto sp = ::mlx::core::add(pos, ::mlx::core::log1p(::mlx::core::exp(neg_abs)));
        auto t = ::mlx::core::tanh(sp);
        auto sig = ::mlx::core::sigmoid(*gx.arr);
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dtype_));
        auto one_minus_t2 = ::mlx::core::subtract(one, ::mlx::core::square(t));
        auto deriv = ::mlx::core::add(
            t, ::mlx::core::multiply(*gx.arr, ::mlx::core::multiply(one_minus_t2, sig)));
        auto out = ::mlx::core::multiply(deriv, *gg.arr);
        return Storage{gpu::wrap_mlx_array(std::move(out), dtype_)};
    }
    const auto& xc = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& gc = std::get<CpuStorage>(g);
    CpuStorage out;
    out.dtype = dtype_;
    out.nbytes = n * dtype_size(dtype_);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    switch (dtype_) {
        case Dtype::F32: {
            auto* xp = reinterpret_cast<const float*>(xc.ptr.get());
            auto* gp = reinterpret_cast<const float*>(gc.ptr.get());
            auto* qp = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xp[i];
                const float t = std::tanh(softplus_stable(x));
                const float sig = 1.f / (1.f + std::exp(-x));
                qp[i] = (t + x * (1.f - t * t) * sig) * gp[i];
            }
            break;
        }
        case Dtype::F64: {
            auto* xp = reinterpret_cast<const double*>(xc.ptr.get());
            auto* gp = reinterpret_cast<const double*>(gc.ptr.get());
            auto* qp = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = xp[i];
                const double t = std::tanh(softplus_stable(x));
                const double sig = 1.0 / (1.0 + std::exp(-x));
                qp[i] = (t + x * (1.0 - t * t) * sig) * gp[i];
            }
            break;
        }
        default:
            ErrorBuilder("mish backward").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr mish_op(const TensorImplPtr& a) {
    return MishBackward::forward(a);
}
LUCID_REGISTER_OP(MishBackward)

// =================== HardSigmoid ===================
const OpSchema HardSigmoidBackward::schema_v1{"hard_sigmoid", 1, AmpPolicy::KeepInput, true};

Storage HardSigmoidBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(g);
        ::mlx::core::array m3(-3.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array p3(3.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array s(1.0 / 6.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dtype_));
        auto in_range = ::mlx::core::logical_and(::mlx::core::greater(*gx.arr, m3),
                                                 ::mlx::core::less(*gx.arr, p3));
        auto s_b = ::mlx::core::broadcast_to(s, gx.arr->shape());
        auto z_b = ::mlx::core::broadcast_to(zero, gx.arr->shape());
        auto deriv = ::mlx::core::where(in_range, s_b, z_b);
        auto out = ::mlx::core::multiply(deriv, *gg.arr);
        return Storage{gpu::wrap_mlx_array(std::move(out), dtype_)};
    }
    const auto& xc = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& gc = std::get<CpuStorage>(g);
    CpuStorage out;
    out.dtype = dtype_;
    out.nbytes = n * dtype_size(dtype_);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    switch (dtype_) {
        case Dtype::F32: {
            auto* xp = reinterpret_cast<const float*>(xc.ptr.get());
            auto* gp = reinterpret_cast<const float*>(gc.ptr.get());
            auto* qp = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                qp[i] = (xp[i] > -3.f && xp[i] < 3.f) ? gp[i] / 6.f : 0.f;
            break;
        }
        case Dtype::F64: {
            auto* xp = reinterpret_cast<const double*>(xc.ptr.get());
            auto* gp = reinterpret_cast<const double*>(gc.ptr.get());
            auto* qp = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                qp[i] = (xp[i] > -3.0 && xp[i] < 3.0) ? gp[i] / 6.0 : 0.0;
            break;
        }
        default:
            ErrorBuilder("hard_sigmoid backward").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr hard_sigmoid_op(const TensorImplPtr& a) {
    return HardSigmoidBackward::forward(a);
}
LUCID_REGISTER_OP(HardSigmoidBackward)

// =================== HardSwish ===================
const OpSchema HardSwishBackward::schema_v1{"hard_swish", 1, AmpPolicy::KeepInput, true};

Storage HardSwishBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(g);
        ::mlx::core::array m3(-3.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array p3(3.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array half(0.5, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array third(1.0 / 3.0, gpu::to_mlx_dtype(dtype_));
        auto mid_branch = ::mlx::core::add(::mlx::core::multiply(*gx.arr, third), half);
        auto le_m3 = ::mlx::core::less_equal(*gx.arr, m3);
        auto ge_p3 = ::mlx::core::greater_equal(*gx.arr, p3);
        auto z_b = ::mlx::core::broadcast_to(zero, gx.arr->shape());
        auto o_b = ::mlx::core::broadcast_to(one, gx.arr->shape());
        auto step1 = ::mlx::core::where(ge_p3, o_b, mid_branch);
        auto deriv = ::mlx::core::where(le_m3, z_b, step1);
        auto out = ::mlx::core::multiply(deriv, *gg.arr);
        return Storage{gpu::wrap_mlx_array(std::move(out), dtype_)};
    }
    const auto& xc = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& gc = std::get<CpuStorage>(g);
    CpuStorage out;
    out.dtype = dtype_;
    out.nbytes = n * dtype_size(dtype_);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    switch (dtype_) {
        case Dtype::F32: {
            auto* xp = reinterpret_cast<const float*>(xc.ptr.get());
            auto* gp = reinterpret_cast<const float*>(gc.ptr.get());
            auto* qp = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xp[i];
                float dx;
                if (x <= -3.f) dx = 0.f;
                else if (x >= 3.f) dx = 1.f;
                else dx = x / 3.f + 0.5f;
                qp[i] = dx * gp[i];
            }
            break;
        }
        case Dtype::F64: {
            auto* xp = reinterpret_cast<const double*>(xc.ptr.get());
            auto* gp = reinterpret_cast<const double*>(gc.ptr.get());
            auto* qp = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = xp[i];
                double dx;
                if (x <= -3.0) dx = 0.0;
                else if (x >= 3.0) dx = 1.0;
                else dx = x / 3.0 + 0.5;
                qp[i] = dx * gp[i];
            }
            break;
        }
        default:
            ErrorBuilder("hard_swish backward").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
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
