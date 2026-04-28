#include "ScalarParam.h"

#include <cmath>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../backend/cpu/Vforce.h"
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
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

namespace {

CpuStorage allocate_unary(const Shape& out_shape, Dtype dt) {
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = shape_numel(out_shape) * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    return out;
}

// Wire the autograd graph for a single-input scalar-parameterized op. Mirrors
// UnaryOp::forward's grad-wiring path so we don't duplicate boilerplate per op.
template <class Derived>
void wire_grad_node(const std::shared_ptr<TensorImpl>& a,
                    std::shared_ptr<TensorImpl>& out,
                    const std::shared_ptr<Derived>& bwd) {
    auto a_edge = detail::ensure_grad_fn(a);
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, /*input_nr=*/0)});
    bwd->set_saved_versions({a->version()});
    out->set_grad_fn(bwd);
    out->set_leaf(false);
    out->set_requires_grad(true);
}

}  // namespace

// =================== PowScalar : x ^ exp ===================

const OpSchema PowScalarBackward::schema_v1{"pow_scalar", 1, AmpPolicy::ForceFP32, true};

CpuStorage PowScalarBackward::cpu_kernel(const CpuStorage& a,
                                         const Shape& out_shape,
                                         Dtype dt,
                                         double exp) {
    // Reuse vpow with a tensor-of-exp filled with `exp`; cheap for backward
    // because we save the input only (exp is a tiny scalar).
    const std::size_t numel = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            std::vector<float> exp_buf(numel, static_cast<float>(exp));
            backend::cpu::vpow_f32(reinterpret_cast<const float*>(a.ptr.get()), exp_buf.data(),
                                   reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        }
        case Dtype::F64: {
            std::vector<double> exp_buf(numel, exp);
            backend::cpu::vpow_f64(reinterpret_cast<const double*>(a.ptr.get()), exp_buf.data(),
                                   reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        }
        default:
            ErrorBuilder("pow_scalar").not_implemented("dtype not supported");
    }
    return out;
}

Storage PowScalarBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = exp * x^(exp - 1) * g
    std::vector<double> exp_minus(1, exp_ - 1.0);  // unused, just docs intent
    (void)exp_minus;
    // Use mul_scalar(x^(exp-1), exp) * g.
    // Build x^(exp-1) via vpow with a tensor filled with (exp-1).
    auto pow_kernel = [&](const Storage& base, double e) {
        const auto& s = std::get<CpuStorage>(base);
        auto out = allocate_unary(out_shape_, dtype_);
        switch (dtype_) {
            case Dtype::F32: {
                std::vector<float> ev(n, static_cast<float>(e));
                backend::cpu::vpow_f32(reinterpret_cast<const float*>(s.ptr.get()), ev.data(),
                                       reinterpret_cast<float*>(out.ptr.get()), n);
                break;
            }
            case Dtype::F64: {
                std::vector<double> ev(n, e);
                backend::cpu::vpow_f64(reinterpret_cast<const double*>(s.ptr.get()), ev.data(),
                                       reinterpret_cast<double*>(out.ptr.get()), n);
                break;
            }
            default:
                ErrorBuilder("pow_scalar grad").not_implemented("dtype not supported");
        }
        return Storage{std::move(out)};
    };

    Storage x_pow_em1 = pow_kernel(saved_inputs_[0], exp_ - 1.0);
    Storage scaled = mul_scalar_storage(x_pow_em1, exp_, n, dtype_, device_);
    return multiply_storages(scaled, g, n, dtype_, device_);
}

TensorImplPtr PowScalarBackward::forward(const TensorImplPtr& a, double exp) {
    Validator::input(a, "pow_scalar.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage;
    if (a->device() == Device::GPU) {
        const auto& g = std::get<GpuStorage>(a->storage());
        if (!g.arr)
            ErrorBuilder("pow_scalar").fail("null GPU input");
        ::mlx::core::array c(exp, gpu::to_mlx_dtype(a->dtype()));
        auto out = ::mlx::core::power(*g.arr, c);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype())};
    } else {
        out_storage =
            Storage{cpu_kernel(std::get<CpuStorage>(a->storage()), a->shape(), a->dtype(), exp)};
    }
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()) * 11);

    if (!GradMode::is_enabled() || !a->requires_grad())
        return out;

    auto bwd = std::make_shared<PowScalarBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = a->shape();
    bwd->dtype_ = a->dtype();
    bwd->device_ = a->device();
    bwd->saved_inputs_ = {a->storage()};
    bwd->input_tensors_ = {a};  // Item #9 — for version check
    bwd->exp_ = exp;
    wire_grad_node(a, out, bwd);
    return out;
}

TensorImplPtr pow_scalar_op(const TensorImplPtr& a, double exp) {
    return PowScalarBackward::forward(a, exp);
}
LUCID_REGISTER_OP(PowScalarBackward)

// =================== RPowScalar : base ^ x ===================

const OpSchema RPowScalarBackward::schema_v1{"rpow_scalar", 1, AmpPolicy::ForceFP32, true};

CpuStorage RPowScalarBackward::cpu_kernel(const CpuStorage& a,
                                          const Shape& out_shape,
                                          Dtype dt,
                                          double base) {
    const std::size_t numel = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            std::vector<float> base_buf(numel, static_cast<float>(base));
            backend::cpu::vpow_f32(base_buf.data(), reinterpret_cast<const float*>(a.ptr.get()),
                                   reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        }
        case Dtype::F64: {
            std::vector<double> base_buf(numel, base);
            backend::cpu::vpow_f64(base_buf.data(), reinterpret_cast<const double*>(a.ptr.get()),
                                   reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        }
        default:
            ErrorBuilder("rpow_scalar").not_implemented("dtype not supported");
    }
    return out;
}

Storage RPowScalarBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = ln(base) * base^x * g = ln(base) * output * g
    const double ln_base = std::log(base_);
    Storage scaled_out = mul_scalar_storage(saved_output_, ln_base, n, dtype_, device_);
    return multiply_storages(scaled_out, g, n, dtype_, device_);
}

TensorImplPtr RPowScalarBackward::forward(double base, const TensorImplPtr& a) {
    Validator::input(a, "rpow_scalar.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage;
    if (a->device() == Device::GPU) {
        const auto& g = std::get<GpuStorage>(a->storage());
        if (!g.arr)
            ErrorBuilder("rpow_scalar").fail("null GPU input");
        ::mlx::core::array c(base, gpu::to_mlx_dtype(a->dtype()));
        auto out = ::mlx::core::power(c, *g.arr);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype())};
    } else {
        out_storage =
            Storage{cpu_kernel(std::get<CpuStorage>(a->storage()), a->shape(), a->dtype(), base)};
    }
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()) * 11);

    if (!GradMode::is_enabled() || !a->requires_grad())
        return out;

    auto bwd = std::make_shared<RPowScalarBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = a->shape();
    bwd->dtype_ = a->dtype();
    bwd->device_ = a->device();
    bwd->saved_output_ = out->storage();
    bwd->input_tensors_ = {a};  // Item #9 — for version check
    bwd->base_ = base;
    wire_grad_node(a, out, bwd);
    return out;
}

TensorImplPtr rpow_scalar_op(double base, const TensorImplPtr& a) {
    return RPowScalarBackward::forward(base, a);
}
LUCID_REGISTER_OP(RPowScalarBackward)

// =================== Clip : clamp(x, lo, hi) ===================

const OpSchema ClipBackward::schema_v1{"clip", 1, AmpPolicy::KeepInput, true};

CpuStorage ClipBackward::cpu_kernel(
    const CpuStorage& a, const Shape& out_shape, Dtype dt, double min_v, double max_v) {
    const std::size_t numel = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            const auto flo = static_cast<float>(min_v);
            const auto fhi = static_cast<float>(max_v);
            for (std::size_t i = 0; i < numel; ++i) {
                float v = p[i];
                if (v < flo)
                    v = flo;
                else if (v > fhi)
                    v = fhi;
                q[i] = v;
            }
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) {
                double v = p[i];
                if (v < min_v)
                    v = min_v;
                else if (v > max_v)
                    v = max_v;
                q[i] = v;
            }
            break;
        }
        default:
            ErrorBuilder("clip").not_implemented("dtype not supported");
    }
    return out;
}

Storage ClipBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = in_range_mask_storage(saved_inputs_[0], min_, max_, n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr ClipBackward::forward(const TensorImplPtr& a, double min_v, double max_v) {
    Validator::input(a, "clip.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage;
    if (a->device() == Device::GPU) {
        const auto& g = std::get<GpuStorage>(a->storage());
        if (!g.arr)
            ErrorBuilder("clip").fail("null GPU input");
        ::mlx::core::array lo(min_v, gpu::to_mlx_dtype(a->dtype()));
        ::mlx::core::array hi(max_v, gpu::to_mlx_dtype(a->dtype()));
        auto out = ::mlx::core::clip(*g.arr, lo, hi);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype())};
    } else {
        out_storage = Storage{
            cpu_kernel(std::get<CpuStorage>(a->storage()), a->shape(), a->dtype(), min_v, max_v)};
    }
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    if (!GradMode::is_enabled() || !a->requires_grad())
        return out;

    auto bwd = std::make_shared<ClipBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = a->shape();
    bwd->dtype_ = a->dtype();
    bwd->device_ = a->device();
    bwd->saved_inputs_ = {a->storage()};
    bwd->input_tensors_ = {a};  // Item #9 — for version check
    bwd->min_ = min_v;
    bwd->max_ = max_v;
    wire_grad_node(a, out, bwd);
    return out;
}

TensorImplPtr clip_op(const TensorImplPtr& a, double min_v, double max_v) {
    return ClipBackward::forward(a, min_v, max_v);
}
LUCID_REGISTER_OP(ClipBackward)

}  // namespace lucid
