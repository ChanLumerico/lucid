#include "LayerNorm.h"

#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/cpu/Norm.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/ErrorBuilder.h"
#include "../core/Exceptions.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../ops/bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

const OpSchema LayerNormBackward::schema_v1{"layer_norm", 1, AmpPolicy::ForceFP32, true};

namespace {

// γ defines the normalized shape (last dims of x). Validate alignment.
struct LayerNormShapes {
    std::size_t outer;
    std::size_t N;
};

LayerNormShapes resolve_shapes(const Shape& x_shape, const Shape& gamma_shape) {
    if (gamma_shape.size() > x_shape.size()) {
        throw ShapeMismatch(x_shape, gamma_shape, "layer_norm: γ has more dims than x");
    }
    const std::size_t Dn = gamma_shape.size();
    const std::size_t lead = x_shape.size() - Dn;
    for (std::size_t i = 0; i < Dn; ++i) {
        if (x_shape[lead + i] != gamma_shape[i]) {
            throw ShapeMismatch(x_shape, gamma_shape,
                                "layer_norm: γ shape must match trailing dims of x");
        }
    }
    std::size_t outer = 1;
    for (std::size_t i = 0; i < lead; ++i)
        outer *= static_cast<std::size_t>(x_shape[i]);
    std::size_t N = 1;
    for (std::size_t i = 0; i < Dn; ++i)
        N *= static_cast<std::size_t>(gamma_shape[i]);
    return {outer, N};
}

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

}  // namespace

TensorImplPtr LayerNormBackward::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& gamma,
                                         const TensorImplPtr& beta,
                                         double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("layer_norm").fail("null input");
    if (x->dtype_ != gamma->dtype_ || x->dtype_ != beta->dtype_)
        throw DtypeMismatch(std::string(dtype_name(x->dtype_)),
                            std::string(dtype_name(gamma->dtype_)), "layer_norm");
    if (x->device_ != gamma->device_ || x->device_ != beta->device_)
        throw DeviceMismatch(std::string(device_name(x->device_)),
                             std::string(device_name(gamma->device_)), "layer_norm");
    if (x->device_ == Device::CPU &&
        (!x->is_contiguous() || !gamma->is_contiguous() || !beta->is_contiguous()))
        ErrorBuilder("layer_norm").not_implemented("non-contiguous input not supported");
    if (gamma->shape_ != beta->shape_)
        throw ShapeMismatch(gamma->shape_, beta->shape_,
                            "layer_norm: γ and β must have the same shape");

    const auto [outer, N] = resolve_shapes(x->shape_, gamma->shape_);

    OpScopeFull scope{schema_v1.name, x->device_, x->dtype_, x->shape_};

    Storage out_storage;
    Storage saved_mean;
    Storage saved_rstd;

    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto& gg = std::get<GpuStorage>(gamma->storage_);
        const auto& gb = std::get<GpuStorage>(beta->storage_);
        if (!gx.arr || !gg.arr || !gb.arr) {
            ErrorBuilder("layer_norm").fail("null GPU input");
        }
        // Flatten to (outer, N) then reshape γ/β to (1, N) so broadcast works.
        Shape flatX{static_cast<std::int64_t>(outer), static_cast<std::int64_t>(N)};
        Shape flatG{1, static_cast<std::int64_t>(N)};
        auto x_2d = ::mlx::core::reshape(*gx.arr, gpu::to_mlx_shape(flatX));
        auto g_2d = ::mlx::core::reshape(*gg.arr, gpu::to_mlx_shape(flatG));
        auto b_2d = ::mlx::core::reshape(*gb.arr, gpu::to_mlx_shape(flatG));
        // mean = sum(x, axis=1, keepdims) / N
        auto mean = ::mlx::core::mean(x_2d, std::vector<int>{1},
                                      /*keepdims=*/true);
        auto centered = ::mlx::core::subtract(x_2d, mean);
        auto var = ::mlx::core::mean(::mlx::core::square(centered), std::vector<int>{1},
                                     /*keepdims=*/true);
        ::mlx::core::array eps_arr(eps, gpu::to_mlx_dtype(x->dtype_));
        auto rstd = ::mlx::core::rsqrt(::mlx::core::add(var, eps_arr));
        auto xnorm = ::mlx::core::multiply(centered, rstd);
        auto y_2d = ::mlx::core::add(::mlx::core::multiply(xnorm, g_2d), b_2d);
        auto y = ::mlx::core::reshape(y_2d, gpu::to_mlx_shape(x->shape_));
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype_)};
        // Save mean and rstd at flat shape (outer, 1) — used by backward.
        saved_mean = Storage{gpu::wrap_mlx_array(std::move(mean), x->dtype_)};
        saved_rstd = Storage{gpu::wrap_mlx_array(std::move(rstd), x->dtype_)};
    } else {
        auto y_cpu = allocate_size(outer * N, x->dtype_);
        auto mean_cpu = allocate_size(outer, x->dtype_);
        auto rstd_cpu = allocate_size(outer, x->dtype_);

        if (outer * N > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x->storage_);
            const auto& g_cpu = std::get<CpuStorage>(gamma->storage_);
            const auto& b_cpu = std::get<CpuStorage>(beta->storage_);
            switch (x->dtype_) {
                case Dtype::F32:
                    backend::cpu::layer_norm_forward_f32(
                        reinterpret_cast<const float*>(x_cpu.ptr.get()),
                        reinterpret_cast<const float*>(g_cpu.ptr.get()),
                        reinterpret_cast<const float*>(b_cpu.ptr.get()),
                        reinterpret_cast<float*>(y_cpu.ptr.get()),
                        reinterpret_cast<float*>(mean_cpu.ptr.get()),
                        reinterpret_cast<float*>(rstd_cpu.ptr.get()), outer, N, eps);
                    break;
                case Dtype::F64:
                    backend::cpu::layer_norm_forward_f64(
                        reinterpret_cast<const double*>(x_cpu.ptr.get()),
                        reinterpret_cast<const double*>(g_cpu.ptr.get()),
                        reinterpret_cast<const double*>(b_cpu.ptr.get()),
                        reinterpret_cast<double*>(y_cpu.ptr.get()),
                        reinterpret_cast<double*>(mean_cpu.ptr.get()),
                        reinterpret_cast<double*>(rstd_cpu.ptr.get()), outer, N, eps);
                    break;
                default:
                    ErrorBuilder("layer_norm").not_implemented("dtype not supported (F32/F64)");
            }
        }
        out_storage = Storage{std::move(y_cpu)};
        saved_mean = Storage{std::move(mean_cpu)};
        saved_rstd = Storage{std::move(rstd_cpu)};
    }
    scope.set_flops(static_cast<std::int64_t>(outer * N) * 5);

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), x->shape_, x->dtype_, x->device_,
                                     /*requires_grad=*/false);

    const bool needs_grad = GradMode::is_enabled() &&
                            (x->requires_grad_ || gamma->requires_grad_ || beta->requires_grad_);
    if (!needs_grad)
        return out;

    auto x_edge = detail::ensure_grad_fn(x);
    auto g_edge = detail::ensure_grad_fn(gamma);
    auto b_edge = detail::ensure_grad_fn(beta);

    auto bwd = std::make_shared<LayerNormBackward>();
    bwd->input_shapes_ = {x->shape_, gamma->shape_, beta->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = x->dtype_;
    bwd->device_ = x->device_;
    bwd->input_tensors_ = {x, gamma, beta};
    bwd->saved_inputs_ = {x->storage_, gamma->storage_, beta->storage_};
    bwd->saved_mean_ = std::move(saved_mean);
    bwd->saved_rstd_ = std::move(saved_rstd);
    bwd->outer_ = outer;
    bwd->N_ = N;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(g_edge, 0), Edge(b_edge, 0)});
    bwd->set_saved_versions({x->version_, gamma->version_, beta->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> LayerNormBackward::apply(Storage grad_out) {
    if (device_ == Device::GPU) {
        // Standard LayerNorm backward via raw MLX ops.
        // Saved: mean and rstd shape (outer, 1).
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gM = std::get<GpuStorage>(saved_mean_);
        const auto& gR = std::get<GpuStorage>(saved_rstd_);
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gx.arr || !gg.arr || !gM.arr || !gR.arr || !gG.arr) {
            ErrorBuilder("layer_norm backward").fail("null GPU array");
        }
        Shape flatX{static_cast<std::int64_t>(outer_), static_cast<std::int64_t>(N_)};
        Shape flatG{1, static_cast<std::int64_t>(N_)};
        auto x_2d = ::mlx::core::reshape(*gx.arr, gpu::to_mlx_shape(flatX));
        auto g_2d = ::mlx::core::reshape(*gG.arr, gpu::to_mlx_shape(flatX));
        auto gamma_2d = ::mlx::core::reshape(*gg.arr, gpu::to_mlx_shape(flatG));

        auto centered = ::mlx::core::subtract(x_2d, *gM.arr);
        auto xnorm = ::mlx::core::multiply(centered, *gR.arr);
        // dbeta = sum(g, axis=0) → (N,); dgamma = sum(g * xnorm, axis=0) → (N,)
        auto dbeta_2d = ::mlx::core::sum(g_2d, std::vector<int>{0},
                                         /*keepdims=*/false);
        auto dgamma_2d = ::mlx::core::sum(::mlx::core::multiply(g_2d, xnorm), std::vector<int>{0},
                                          /*keepdims=*/false);
        // gx = g * gamma; mean1 = mean(gx); mean2 = mean(gx * xnorm)
        // dx = rstd * (gx - mean1 - xnorm * mean2)
        auto gx_scaled = ::mlx::core::multiply(g_2d, gamma_2d);
        auto mean1 = ::mlx::core::mean(gx_scaled, std::vector<int>{1},
                                       /*keepdims=*/true);
        auto mean2 = ::mlx::core::mean(::mlx::core::multiply(gx_scaled, xnorm), std::vector<int>{1},
                                       /*keepdims=*/true);
        auto dx_2d = ::mlx::core::multiply(
            *gR.arr, ::mlx::core::subtract(::mlx::core::subtract(gx_scaled, mean1),
                                           ::mlx::core::multiply(xnorm, mean2)));
        // Reshape outputs back: dx → input_shapes_[0]; dgamma/dbeta → γ shape
        auto dx = ::mlx::core::reshape(dx_2d, gpu::to_mlx_shape(input_shapes_[0]));
        auto dgamma = ::mlx::core::reshape(dgamma_2d, gpu::to_mlx_shape(input_shapes_[1]));
        auto dbeta = ::mlx::core::reshape(dbeta_2d, gpu::to_mlx_shape(input_shapes_[2]));
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dgamma), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dbeta), dtype_)}};
    }

    auto dx_cpu = allocate_size(outer_ * N_, dtype_);
    auto dgamma_cpu = allocate_size(N_, dtype_);
    auto dbeta_cpu = allocate_size(N_, dtype_);

    if (outer_ * N_ > 0) {
        const auto& x_cpu = std::get<CpuStorage>(saved_inputs_[0]);
        const auto& gamma_cpu = std::get<CpuStorage>(saved_inputs_[1]);
        const auto& mean_cpu = std::get<CpuStorage>(saved_mean_);
        const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd_);
        const auto& g_cpu = std::get<CpuStorage>(grad_out);

        switch (dtype_) {
            case Dtype::F32:
                backend::cpu::layer_norm_backward_f32(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const float*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                    reinterpret_cast<float*>(dx_cpu.ptr.get()),
                    reinterpret_cast<float*>(dgamma_cpu.ptr.get()),
                    reinterpret_cast<float*>(dbeta_cpu.ptr.get()), outer_, N_);
                break;
            case Dtype::F64:
                backend::cpu::layer_norm_backward_f64(
                    reinterpret_cast<const double*>(x_cpu.ptr.get()),
                    reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const double*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const double*>(g_cpu.ptr.get()),
                    reinterpret_cast<double*>(dx_cpu.ptr.get()),
                    reinterpret_cast<double*>(dgamma_cpu.ptr.get()),
                    reinterpret_cast<double*>(dbeta_cpu.ptr.get()), outer_, N_);
                break;
            default:
                ErrorBuilder("layer_norm backward").not_implemented("dtype not supported");
        }
    } else {
        if (dx_cpu.nbytes)
            std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
        if (dgamma_cpu.nbytes)
            std::memset(dgamma_cpu.ptr.get(), 0, dgamma_cpu.nbytes);
        if (dbeta_cpu.nbytes)
            std::memset(dbeta_cpu.ptr.get(), 0, dbeta_cpu.nbytes);
    }

    return {Storage{std::move(dx_cpu)}, Storage{std::move(dgamma_cpu)},
            Storage{std::move(dbeta_cpu)}};
}

TensorImplPtr layer_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            double eps) {
    return LayerNormBackward::forward(x, gamma, beta, eps);
}

LUCID_REGISTER_OP(LayerNormBackward)

}  // namespace lucid
