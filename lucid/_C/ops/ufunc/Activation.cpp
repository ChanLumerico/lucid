#include "Activation.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <mlx/ops.h>

#include "../../backend/cpu/Vdsp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

namespace {
CpuStorage allocate_unary(const Shape& out_shape, Dtype dt) {
    CpuStorage out;
    out.dtype  = dt;
    out.nbytes = shape_numel(out_shape) * dtype_size(dt);
    out.ptr    = allocate_aligned_bytes(out.nbytes);
    return out;
}
}  // namespace

const OpSchema ReluBackward::schema_v1{"relu", 1, AmpPolicy::KeepInput, true};

CpuStorage ReluBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    const std::size_t numel = shape_numel(out_shape);
    CpuStorage out;
    out.dtype  = dt;
    out.nbytes = numel * dtype_size(dt);
    out.ptr    = allocate_aligned_bytes(out.nbytes);
    switch (dt) {
        case Dtype::F32:
            backend::cpu::vrelu_f32(
                reinterpret_cast<const float*>(a.ptr.get()),
                reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            backend::cpu::vrelu_f64(
                reinterpret_cast<const double*>(a.ptr.get()),
                reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            throw NotImplementedError("relu: dtype not supported");
    }
    return out;
}

Storage ReluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = positive_mask_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr relu_op(const TensorImplPtr& a) { return ReluBackward::forward(a); }
LUCID_REGISTER_OP(ReluBackward)

// =================== Sigmoid ===================
const OpSchema SigmoidBackward::schema_v1{"sigmoid", 1, AmpPolicy::Promote, true};

CpuStorage SigmoidBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape,
                                       Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    Storage tmp = sigmoid_storage(Storage{a}, n, dt, Device::CPU);
    return std::get<CpuStorage>(std::move(tmp));
}

Storage SigmoidBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = z (1 - z) g, z = saved output.
    Storage neg_z   = mul_scalar_storage(saved_output_, -1.0, n, dtype_, device_);
    Storage one_m_z = add_scalar_storage(neg_z, 1.0, n, dtype_, device_);
    Storage z_omz   = multiply_storages(saved_output_, one_m_z, n, dtype_, device_);
    return multiply_storages(z_omz, g, n, dtype_, device_);
}

TensorImplPtr sigmoid_op(const TensorImplPtr& a) { return SigmoidBackward::forward(a); }
LUCID_REGISTER_OP(SigmoidBackward)

// =================== SiLU (Swish) ===================
const OpSchema SiluBackward::schema_v1{"silu", 1, AmpPolicy::Promote, true};

CpuStorage SiluBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    // y = x * σ(x). Compute σ first, then multiply.
    const std::size_t n = shape_numel(out_shape);
    Storage sx = sigmoid_storage(Storage{a}, n, dt, Device::CPU);
    return std::get<CpuStorage>(multiply_storages(Storage{a}, sx, n, dt, Device::CPU));
}

Storage SiluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = σ(x) + x · σ(x) · (1 - σ(x)) = σ(x) · (1 + x(1 - σ(x)))
    Storage sx       = sigmoid_storage(saved_inputs_[0], n, dtype_, device_);
    Storage neg_sx   = mul_scalar_storage(sx, -1.0, n, dtype_, device_);
    Storage one_m_sx = add_scalar_storage(neg_sx, 1.0, n, dtype_, device_);
    Storage x_omsx   = multiply_storages(saved_inputs_[0], one_m_sx, n, dtype_, device_);
    Storage one_p    = add_scalar_storage(x_omsx, 1.0, n, dtype_, device_);
    Storage dx       = multiply_storages(sx, one_p, n, dtype_, device_);
    return multiply_storages(dx, g, n, dtype_, device_);
}

TensorImplPtr silu_op(const TensorImplPtr& a) { return SiluBackward::forward(a); }
LUCID_REGISTER_OP(SiluBackward)

// =================== GeLU (tanh approximation) ===================
//
// gelu(x) = 0.5 x (1 + tanh(c1 (x + c2 x³)))
// with c1 = √(2/π), c2 = 0.044715.
//
// d/dx gelu(x) = 0.5 (1+t) + 0.5 x (1-t²) c1 (1 + 3 c2 x²)
// where t = tanh(c1 (x + c2 x³)).
//
// Backward recomputes t (cheap vs storing it).

const OpSchema GeluBackward::schema_v1{"gelu", 1, AmpPolicy::ForceFP32, true};

namespace {
constexpr double kGeluC1 = 0.7978845608028654;  // √(2/π)
constexpr double kGeluC2 = 0.044715;
}

CpuStorage GeluBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const float c1 = static_cast<float>(kGeluC1);
            const float c2 = static_cast<float>(kGeluC2);
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                const float inner = c1 * (x + c2 * x * x * x);
                const float t = std::tanh(inner);
                q[i] = 0.5f * x * (1.f + t);
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                const double inner = kGeluC1 * (x + kGeluC2 * x * x * x);
                const double t = std::tanh(inner);
                q[i] = 0.5 * x * (1.0 + t);
            }
            break;
        }
        default:
            throw NotImplementedError("gelu: dtype not supported");
    }
    return out;
}

Storage GeluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    const auto& x_cpu = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& g_cpu = std::get<CpuStorage>(g);
    CpuStorage out;
    out.dtype  = dtype_;
    out.nbytes = n * dtype_size(dtype_);
    out.ptr    = allocate_aligned_bytes(out.nbytes);

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
            throw NotImplementedError("gelu backward: dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr gelu_op(const TensorImplPtr& a) { return GeluBackward::forward(a); }
LUCID_REGISTER_OP(GeluBackward)

// =================== LeakyReLU ===================
const OpSchema LeakyReluBackward::schema_v1{"leaky_relu", 1, AmpPolicy::KeepInput, true};

CpuStorage LeakyReluBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape,
                                         Dtype dt, double slope) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const auto fs = static_cast<float>(slope);
            for (std::size_t i = 0; i < n; ++i) q[i] = (p[i] >= 0.f) ? p[i] : fs * p[i];
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) q[i] = (p[i] >= 0.0) ? p[i] : slope * p[i];
            break;
        }
        default:
            throw NotImplementedError("leaky_relu: dtype not supported");
    }
    return out;
}

Storage LeakyReluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = leaky_mask_storage(saved_inputs_[0], slope_, n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr LeakyReluBackward::forward(const TensorImplPtr& a, double slope) {
    if (!a) throw LucidError("leaky_relu: null input");
    if (a->device_ == Device::CPU && !a->is_contiguous())
        throw NotImplementedError(
            "leaky_relu: non-contiguous input not supported (call .contiguous() first)");

    OpScope scope{schema_v1.name, a->device_, a->dtype_, a->shape_};
    Storage out_storage;
    if (a->device_ == Device::GPU) {
        const auto& g = std::get<GpuStorage>(a->storage_);
        if (!g.arr) throw LucidError("leaky_relu: null GPU input");
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(a->dtype_));
        ::mlx::core::array slope_arr(slope, gpu::to_mlx_dtype(a->dtype_));
        // y = where(x >= 0, x, slope * x)
        auto pos_mask = ::mlx::core::greater_equal(*g.arr, zero);
        auto neg_branch = ::mlx::core::multiply(slope_arr, *g.arr);
        auto out = ::mlx::core::where(pos_mask, *g.arr, neg_branch);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)};
    } else {
        out_storage = Storage{cpu_kernel(std::get<CpuStorage>(a->storage_),
                                         a->shape_, a->dtype_, slope)};
    }
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape_,
                                            a->dtype_, a->device_, false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()));

    if (!GradMode::is_enabled() || !a->requires_grad_) return out;

    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<LeakyReluBackward>();
    bwd->input_shapes_  = {a->shape_};
    bwd->out_shape_     = a->shape_;
    bwd->dtype_         = a->dtype_;
    bwd->device_        = a->device_;
    bwd->input_tensors_ = {a};
    bwd->saved_inputs_  = {a->storage_};
    bwd->slope_         = slope;
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, /*input_nr=*/0)});
    bwd->set_saved_versions({a->version_});

    out->grad_fn_       = std::move(bwd);
    out->is_leaf_       = false;
    out->requires_grad_ = true;
    return out;
}

TensorImplPtr leaky_relu_op(const TensorImplPtr& a, double slope) {
    return LeakyReluBackward::forward(a, slope);
}
LUCID_REGISTER_OP(LeakyReluBackward)

// =================== Softplus ===================
//
// Forward (numerically stable): softplus(x) = max(x, 0) + log(1 + exp(-|x|))
// Backward: dx = σ(x) · g
const OpSchema SoftplusBackward::schema_v1{"softplus", 1, AmpPolicy::ForceFP32, true};

CpuStorage SoftplusBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape,
                                        Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                const float ax = std::abs(x);
                q[i] = std::max(x, 0.f) + std::log1p(std::exp(-ax));
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                const double ax = std::abs(x);
                q[i] = std::max(x, 0.0) + std::log1p(std::exp(-ax));
            }
            break;
        }
        default:
            throw NotImplementedError("softplus: dtype not supported");
    }
    return out;
}

Storage SoftplusBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage sx = sigmoid_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(sx, g, n, dtype_, device_);
}

TensorImplPtr softplus_op(const TensorImplPtr& a) { return SoftplusBackward::forward(a); }
LUCID_REGISTER_OP(SoftplusBackward)

// =================== ELU(x; α) ===================
//   y = x          if x >= 0
//       α(eˣ - 1)   if x < 0
//   dy/dx = 1            if x >= 0
//           α·eˣ          if x < 0
//
// Backward recomputes the branch from saved x (cheap, avoids saving y).

const OpSchema EluBackward::schema_v1{"elu", 1, AmpPolicy::ForceFP32, true};

CpuStorage EluBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape,
                                    Dtype dt, double alpha) {
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
            throw NotImplementedError("elu: dtype not supported");
    }
    return out;
}

Storage EluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(g);
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array alpha_arr(alpha_, gpu::to_mlx_dtype(dtype_));
        auto pos_mask = ::mlx::core::greater_equal(*gx.arr, zero);
        auto neg_branch = ::mlx::core::multiply(alpha_arr, ::mlx::core::exp(*gx.arr));
        auto ones_arr = ::mlx::core::ones_like(*gx.arr);
        auto deriv = ::mlx::core::where(pos_mask, ones_arr, neg_branch);
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
            const float fa = static_cast<float>(alpha_);
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xp[i];
                const float dx = (x >= 0.f) ? 1.f : fa * std::exp(x);
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
                const double dx = (x >= 0.0) ? 1.0 : alpha_ * std::exp(x);
                qp[i] = dx * gp[i];
            }
            break;
        }
        default:
            throw NotImplementedError("elu backward: dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr EluBackward::forward(const TensorImplPtr& a, double alpha) {
    if (!a) throw LucidError("elu: null input");
    if (a->device_ == Device::CPU && !a->is_contiguous())
        throw NotImplementedError(
            "elu: non-contiguous input not supported (call .contiguous() first)");

    OpScope scope{schema_v1.name, a->device_, a->dtype_, a->shape_};
    Storage out_storage;
    if (a->device_ == Device::GPU) {
        const auto& g = std::get<GpuStorage>(a->storage_);
        if (!g.arr) throw LucidError("elu: null GPU input");
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(a->dtype_));
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(a->dtype_));
        ::mlx::core::array alpha_arr(alpha, gpu::to_mlx_dtype(a->dtype_));
        auto pos_mask = ::mlx::core::greater_equal(*g.arr, zero);
        auto neg = ::mlx::core::multiply(
            alpha_arr,
            ::mlx::core::subtract(::mlx::core::exp(*g.arr), one));
        auto out = ::mlx::core::where(pos_mask, *g.arr, neg);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)};
    } else {
        out_storage = Storage{cpu_kernel(std::get<CpuStorage>(a->storage_),
                                          a->shape_, a->dtype_, alpha)};
    }
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape_,
                                             a->dtype_, a->device_, false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()));

    if (!GradMode::is_enabled() || !a->requires_grad_) return out;
    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<EluBackward>();
    bwd->input_shapes_  = {a->shape_};
    bwd->out_shape_     = a->shape_;
    bwd->dtype_         = a->dtype_;
    bwd->device_        = a->device_;
    bwd->input_tensors_ = {a};
    bwd->saved_inputs_  = {a->storage_};
    bwd->alpha_         = alpha;
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
    bwd->set_saved_versions({a->version_});
    out->grad_fn_       = std::move(bwd);
    out->is_leaf_       = false;
    out->requires_grad_ = true;
    return out;
}

TensorImplPtr elu_op(const TensorImplPtr& a, double alpha) {
    return EluBackward::forward(a, alpha);
}
LUCID_REGISTER_OP(EluBackward)

// =================== SELU ===================
namespace {
constexpr double kSeluScale = 1.0507009873554805;
constexpr double kSeluAlpha = 1.6732632423543772;
}

const OpSchema SeluBackward::schema_v1{"selu", 1, AmpPolicy::ForceFP32, true};

CpuStorage SeluBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const float s = static_cast<float>(kSeluScale);
            const float al = static_cast<float>(kSeluAlpha);
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                q[i] = (x >= 0.f) ? s * x : s * al * (std::exp(x) - 1.f);
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                q[i] = (x >= 0.0) ? kSeluScale * x
                                   : kSeluScale * kSeluAlpha * (std::exp(x) - 1.0);
            }
            break;
        }
        default:
            throw NotImplementedError("selu: dtype not supported");
    }
    return out;
}

Storage SeluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(g);
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array s_arr(kSeluScale, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array sa_arr(kSeluScale * kSeluAlpha, gpu::to_mlx_dtype(dtype_));
        auto pos_mask = ::mlx::core::greater_equal(*gx.arr, zero);
        auto pos_branch = ::mlx::core::broadcast_to(s_arr, gx.arr->shape());
        auto neg_branch = ::mlx::core::multiply(sa_arr, ::mlx::core::exp(*gx.arr));
        auto deriv = ::mlx::core::where(pos_mask, pos_branch, neg_branch);
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
            const float s = static_cast<float>(kSeluScale);
            const float sa = static_cast<float>(kSeluScale * kSeluAlpha);
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xp[i];
                const float dx = (x >= 0.f) ? s : sa * std::exp(x);
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
                const double dx = (x >= 0.0) ? kSeluScale
                                              : kSeluScale * kSeluAlpha * std::exp(x);
                qp[i] = dx * gp[i];
            }
            break;
        }
        default:
            throw NotImplementedError("selu backward: dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr selu_op(const TensorImplPtr& a) { return SeluBackward::forward(a); }
LUCID_REGISTER_OP(SeluBackward)

// =================== Mish ===================
//   y = x * tanh(softplus(x))
//
// Let s = softplus(x), t = tanh(s), σ = sigmoid(x). Then:
//   ds/dx = σ
//   dt/dx = (1 - t²) σ
//   dy/dx = t + x · (1 - t²) σ
//
// Backward recomputes t and σ from saved x.

const OpSchema MishBackward::schema_v1{"mish", 1, AmpPolicy::ForceFP32, true};

namespace {
template <typename T>
inline T softplus_stable(T x) {
    const T ax = std::abs(x);
    return std::max(x, T{0}) + std::log1p(std::exp(-ax));
}
}  // namespace

CpuStorage MishBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                q[i] = x * std::tanh(softplus_stable(x));
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                q[i] = x * std::tanh(softplus_stable(x));
            }
            break;
        }
        default:
            throw NotImplementedError("mish: dtype not supported");
    }
    return out;
}

Storage MishBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(g);
        // softplus(x) = max(x, 0) + log1p(exp(-|x|)) — numerically stable
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dtype_));
        auto pos = ::mlx::core::maximum(*gx.arr, zero);
        auto neg_abs = ::mlx::core::negative(::mlx::core::abs(*gx.arr));
        auto sp = ::mlx::core::add(pos, ::mlx::core::log1p(::mlx::core::exp(neg_abs)));
        auto t = ::mlx::core::tanh(sp);
        auto sig = ::mlx::core::sigmoid(*gx.arr);
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dtype_));
        auto one_minus_t2 = ::mlx::core::subtract(one, ::mlx::core::square(t));
        auto deriv = ::mlx::core::add(
            t, ::mlx::core::multiply(*gx.arr,
                                     ::mlx::core::multiply(one_minus_t2, sig)));
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
                const float dx = t + x * (1.f - t * t) * sig;
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
                const double t = std::tanh(softplus_stable(x));
                const double sig = 1.0 / (1.0 + std::exp(-x));
                const double dx = t + x * (1.0 - t * t) * sig;
                qp[i] = dx * gp[i];
            }
            break;
        }
        default:
            throw NotImplementedError("mish backward: dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr mish_op(const TensorImplPtr& a) { return MishBackward::forward(a); }
LUCID_REGISTER_OP(MishBackward)

// =================== HardSigmoid ===================
//   y = clip((x + 3) / 6, 0, 1)
//   dy/dx = 1/6 if -3 < x < 3 else 0

const OpSchema HardSigmoidBackward::schema_v1{"hard_sigmoid", 1, AmpPolicy::KeepInput, true};

CpuStorage HardSigmoidBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape,
                                            Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float v = (p[i] + 3.f) / 6.f;
                q[i] = std::min(std::max(v, 0.f), 1.f);
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double v = (p[i] + 3.0) / 6.0;
                q[i] = std::min(std::max(v, 0.0), 1.0);
            }
            break;
        }
        default:
            throw NotImplementedError("hard_sigmoid: dtype not supported");
    }
    return out;
}

Storage HardSigmoidBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(g);
        ::mlx::core::array m3(-3.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array p3(3.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array s(1.0 / 6.0, gpu::to_mlx_dtype(dtype_));
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dtype_));
        auto in_range = ::mlx::core::logical_and(
            ::mlx::core::greater(*gx.arr, m3),
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
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xp[i];
                qp[i] = (x > -3.f && x < 3.f) ? gp[i] / 6.f : 0.f;
            }
            break;
        }
        case Dtype::F64: {
            auto* xp = reinterpret_cast<const double*>(xc.ptr.get());
            auto* gp = reinterpret_cast<const double*>(gc.ptr.get());
            auto* qp = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = xp[i];
                qp[i] = (x > -3.0 && x < 3.0) ? gp[i] / 6.0 : 0.0;
            }
            break;
        }
        default:
            throw NotImplementedError("hard_sigmoid backward: dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr hard_sigmoid_op(const TensorImplPtr& a) {
    return HardSigmoidBackward::forward(a);
}
LUCID_REGISTER_OP(HardSigmoidBackward)

// =================== HardSwish ===================
//   y = x * HardSigmoid(x) = x · clip((x+3)/6, 0, 1)
//   dy/dx = 0                  if x ≤ -3
//           x/3 + 0.5           if -3 < x < 3
//           1                   if x ≥ 3

const OpSchema HardSwishBackward::schema_v1{"hard_swish", 1, AmpPolicy::KeepInput, true};

CpuStorage HardSwishBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape,
                                          Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                const float h = std::min(std::max((x + 3.f) / 6.f, 0.f), 1.f);
                q[i] = x * h;
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                const double h = std::min(std::max((x + 3.0) / 6.0, 0.0), 1.0);
                q[i] = x * h;
            }
            break;
        }
        default:
            throw NotImplementedError("hard_swish: dtype not supported");
    }
    return out;
}

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
        auto mid_branch = ::mlx::core::add(
            ::mlx::core::multiply(*gx.arr, third), half);
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
            throw NotImplementedError("hard_swish backward: dtype not supported");
    }
    return Storage{std::move(out)};
}

TensorImplPtr hard_swish_op(const TensorImplPtr& a) {
    return HardSwishBackward::forward(a);
}
LUCID_REGISTER_OP(HardSwishBackward)

// =================== ReLU6 ===================
//   y = clip(x, 0, 6)
//   dy/dx = 1 if 0 < x < 6 else 0

const OpSchema Relu6Backward::schema_v1{"relu6", 1, AmpPolicy::KeepInput, true};

CpuStorage Relu6Backward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) q[i] = std::min(std::max(p[i], 0.f), 6.f);
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i) q[i] = std::min(std::max(p[i], 0.0), 6.0);
            break;
        }
        default:
            throw NotImplementedError("relu6: dtype not supported");
    }
    return out;
}

Storage Relu6Backward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = in_range_mask_storage(saved_inputs_[0], 0.0, 6.0, n,
                                          dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr relu6_op(const TensorImplPtr& a) { return Relu6Backward::forward(a); }
LUCID_REGISTER_OP(Relu6Backward)

}  // namespace lucid
