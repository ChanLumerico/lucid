#include "GroupNorm.h"

#include <cmath>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Exceptions.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/TensorImpl.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema GroupNormBackward::schema_v1{"group_norm", 1, AmpPolicy::ForceFP32, true};

namespace {

CpuStorage alloc_bytes(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

template <typename T>
void group_norm_forward_typed(const T* x,
                              const T* gamma,
                              const T* beta,
                              T* y,
                              T* mean_bg,
                              T* rstd_bg,
                              int B,
                              int C,
                              int spatial,
                              int G,
                              double eps) {
    const int Cg = C / G;
    const std::size_t per_group = static_cast<std::size_t>(Cg) * spatial;
    const T inv_pg = T{1} / static_cast<T>(per_group);

    for (int b = 0; b < B; ++b) {
        for (int g = 0; g < G; ++g) {
            // mean / var across (Cg, spatial) for this (b, g).
            T mean = T{};
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = g * Cg + cc;
                const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
                for (int i = 0; i < spatial; ++i)
                    mean += xb[i];
            }
            mean *= inv_pg;
            T var = T{};
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = g * Cg + cc;
                const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    const T d = xb[i] - mean;
                    var += d * d;
                }
            }
            var *= inv_pg;
            const T rstd = T{1} / std::sqrt(var + static_cast<T>(eps));
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = g * Cg + cc;
                const T gc = gamma[c];
                const T bc = beta[c];
                const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
                T* yb = y + (static_cast<std::size_t>(b) * C + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    yb[i] = gc * (xb[i] - mean) * rstd + bc;
                }
            }
            mean_bg[b * G + g] = mean;
            rstd_bg[b * G + g] = rstd;
        }
    }
}

template <typename T>
void group_norm_backward_typed(const T* x,
                               const T* gamma,
                               const T* mean_bg,
                               const T* rstd_bg,
                               const T* g,
                               T* dx,
                               T* dgamma,
                               T* dbeta,
                               int B,
                               int C,
                               int spatial,
                               int G) {
    const int Cg = C / G;
    const std::size_t per_group = static_cast<std::size_t>(Cg) * spatial;
    const T inv_pg = T{1} / static_cast<T>(per_group);

    for (int co = 0; co < C; ++co) {
        dgamma[co] = T{};
        dbeta[co] = T{};
    }

    for (int b = 0; b < B; ++b) {
        for (int gi = 0; gi < G; ++gi) {
            const T m = mean_bg[b * G + gi];
            const T r = rstd_bg[b * G + gi];
            T sum_dxn = T{};
            T sum_dxn_xn = T{};
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = gi * Cg + cc;
                const T gc = gamma[c];
                const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
                const T* gb = g + (static_cast<std::size_t>(b) * C + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    const T xn_i = (xb[i] - m) * r;
                    const T dxn_i = gc * gb[i];
                    sum_dxn += dxn_i;
                    sum_dxn_xn += dxn_i * xn_i;
                    dgamma[c] += gb[i] * xn_i;
                    dbeta[c] += gb[i];
                }
            }
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = gi * Cg + cc;
                const T gc = gamma[c];
                const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
                const T* gb = g + (static_cast<std::size_t>(b) * C + c) * spatial;
                T* dxb = dx + (static_cast<std::size_t>(b) * C + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    const T xn_i = (xb[i] - m) * r;
                    const T dxn_i = gc * gb[i];
                    dxb[i] = inv_pg * r *
                             (static_cast<T>(per_group) * dxn_i - sum_dxn - xn_i * sum_dxn_xn);
                }
            }
        }
    }
}

}  // namespace

TensorImplPtr GroupNormBackward::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& gamma,
                                         const TensorImplPtr& beta,
                                         int G,
                                         double eps) {
    if (!x || !gamma || !beta)
        throw LucidError("group_norm: null input");
    if (x->dtype_ != gamma->dtype_ || x->dtype_ != beta->dtype_)
        throw DtypeMismatch(std::string(dtype_name(x->dtype_)),
                            std::string(dtype_name(gamma->dtype_)), "group_norm");
    if (x->device_ != gamma->device_ || x->device_ != beta->device_)
        throw DeviceMismatch(std::string(device_name(x->device_)),
                             std::string(device_name(gamma->device_)), "group_norm");
    if (x->device_ == Device::CPU &&
        (!x->is_contiguous() || !gamma->is_contiguous() || !beta->is_contiguous()))
        throw NotImplementedError("group_norm: non-contiguous input not supported");
    if (x->shape_.size() < 2)
        throw ShapeMismatch(x->shape_, Shape{}, "group_norm: x must be at least (B, C, ...)");
    if (gamma->shape_.size() != 1 || beta->shape_.size() != 1)
        throw ShapeMismatch(gamma->shape_, beta->shape_, "group_norm: γ, β must be 1-D");

    const int B = static_cast<int>(x->shape_[0]);
    const int C = static_cast<int>(x->shape_[1]);
    if (C % G != 0)
        throw LucidError("group_norm: C must be divisible by num_groups");
    if (gamma->shape_[0] != C || beta->shape_[0] != C)
        throw ShapeMismatch(gamma->shape_, x->shape_, "group_norm: γ/β must have length C");

    const int N_spatial = static_cast<int>(x->shape_.size()) - 2;
    std::vector<int> S(N_spatial);
    int spatial_total = 1;
    for (int i = 0; i < N_spatial; ++i) {
        S[i] = static_cast<int>(x->shape_[2 + i]);
        spatial_total *= S[i];
    }
    const int Cg = C / G;

    OpScope scope{schema_v1.name, x->device_, x->dtype_, x->shape_};

    Storage out_storage;
    Storage saved_mean;
    Storage saved_rstd;

    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto& gg = std::get<GpuStorage>(gamma->storage_);
        const auto& gb = std::get<GpuStorage>(beta->storage_);
        if (!gx.arr || !gg.arr || !gb.arr) {
            throw LucidError("group_norm: null GPU input");
        }
        // Reshape x to (B, G, C/G, *S).
        ::mlx::core::Shape grouped;
        grouped.reserve(N_spatial + 3);
        grouped.push_back(static_cast<::mlx::core::ShapeElem>(B));
        grouped.push_back(static_cast<::mlx::core::ShapeElem>(G));
        grouped.push_back(static_cast<::mlx::core::ShapeElem>(Cg));
        for (int i = 0; i < N_spatial; ++i)
            grouped.push_back(static_cast<::mlx::core::ShapeElem>(S[i]));
        auto x_g = ::mlx::core::reshape(*gx.arr, grouped);

        std::vector<int> reduce_axes;
        reduce_axes.reserve(N_spatial + 1);
        reduce_axes.push_back(2);  // C/G
        for (int i = 0; i < N_spatial; ++i)
            reduce_axes.push_back(3 + i);

        auto mean = ::mlx::core::mean(x_g, reduce_axes, /*keepdims=*/true);
        auto centered = ::mlx::core::subtract(x_g, mean);
        auto var = ::mlx::core::mean(::mlx::core::square(centered), reduce_axes,
                                     /*keepdims=*/true);
        ::mlx::core::array eps_arr(eps, gpu::to_mlx_dtype(x->dtype_));
        auto rstd = ::mlx::core::rsqrt(::mlx::core::add(var, eps_arr));
        auto xnorm_g = ::mlx::core::multiply(centered, rstd);
        auto xnorm = ::mlx::core::reshape(xnorm_g, gpu::to_mlx_shape(x->shape_));
        // Per-channel γ/β shape (1, C, 1, ..., 1)
        ::mlx::core::Shape brC(N_spatial + 2, 1);
        brC[1] = static_cast<::mlx::core::ShapeElem>(C);
        auto g_view = ::mlx::core::reshape(*gg.arr, brC);
        auto b_view = ::mlx::core::reshape(*gb.arr, brC);
        auto y = ::mlx::core::add(::mlx::core::multiply(xnorm, g_view), b_view);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype_)};
        // Save mean/rstd at (B, G) shape.
        ::mlx::core::Shape mr{static_cast<::mlx::core::ShapeElem>(B),
                              static_cast<::mlx::core::ShapeElem>(G)};
        saved_mean = Storage{gpu::wrap_mlx_array(::mlx::core::reshape(mean, mr), x->dtype_)};
        saved_rstd = Storage{gpu::wrap_mlx_array(::mlx::core::reshape(rstd, mr), x->dtype_)};
    } else {
        auto y_cpu = alloc_bytes(static_cast<std::size_t>(B) * C * spatial_total, x->dtype_);
        auto mean_cpu = alloc_bytes(static_cast<std::size_t>(B) * G, x->dtype_);
        auto rstd_cpu = alloc_bytes(static_cast<std::size_t>(B) * G, x->dtype_);
        if (B * C * spatial_total > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x->storage_);
            const auto& g_cpu = std::get<CpuStorage>(gamma->storage_);
            const auto& b_cpu = std::get<CpuStorage>(beta->storage_);
            switch (x->dtype_) {
                case Dtype::F32:
                    group_norm_forward_typed<float>(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                                    reinterpret_cast<const float*>(b_cpu.ptr.get()),
                                                    reinterpret_cast<float*>(y_cpu.ptr.get()),
                                                    reinterpret_cast<float*>(mean_cpu.ptr.get()),
                                                    reinterpret_cast<float*>(rstd_cpu.ptr.get()), B,
                                                    C, spatial_total, G, eps);
                    break;
                case Dtype::F64:
                    group_norm_forward_typed<double>(
                        reinterpret_cast<const double*>(x_cpu.ptr.get()),
                        reinterpret_cast<const double*>(g_cpu.ptr.get()),
                        reinterpret_cast<const double*>(b_cpu.ptr.get()),
                        reinterpret_cast<double*>(y_cpu.ptr.get()),
                        reinterpret_cast<double*>(mean_cpu.ptr.get()),
                        reinterpret_cast<double*>(rstd_cpu.ptr.get()), B, C, spatial_total, G, eps);
                    break;
                default:
                    throw NotImplementedError("group_norm: dtype not supported (F32/F64)");
            }
        }
        out_storage = Storage{std::move(y_cpu)};
        saved_mean = Storage{std::move(mean_cpu)};
        saved_rstd = Storage{std::move(rstd_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x->shape_, x->dtype_,
                                            x->device_, false);
    if (!GradMode::is_enabled() ||
        !(x->requires_grad_ || gamma->requires_grad_ || beta->requires_grad_)) {
        return out;
    }

    auto x_edge = detail::ensure_grad_fn(x);
    auto g_edge = detail::ensure_grad_fn(gamma);
    auto b_edge = detail::ensure_grad_fn(beta);

    auto bwd = std::make_shared<GroupNormBackward>();
    bwd->input_shapes_ = {x->shape_, gamma->shape_, beta->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = x->dtype_;
    bwd->device_ = x->device_;
    bwd->input_tensors_ = {x, gamma, beta};
    bwd->saved_inputs_ = {x->storage_, gamma->storage_, beta->storage_};
    bwd->saved_mean_ = std::move(saved_mean);
    bwd->saved_rstd_ = std::move(saved_rstd);
    bwd->B_ = B;
    bwd->C_ = C;
    bwd->G_ = G;
    bwd->spatial_dims_ = std::move(S);
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(g_edge, 0), Edge(b_edge, 0)});
    bwd->set_saved_versions({x->version_, gamma->version_, beta->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> GroupNormBackward::apply(Storage grad_out) {
    const int N_spatial = static_cast<int>(spatial_dims_.size());
    const int Cg = C_ / G_;
    int spatial_total = 1;
    for (int s : spatial_dims_)
        spatial_total *= s;

    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gM = std::get<GpuStorage>(saved_mean_);
        const auto& gR = std::get<GpuStorage>(saved_rstd_);
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gx.arr || !gg.arr || !gM.arr || !gR.arr || !gG.arr) {
            throw LucidError("group_norm backward: null GPU array");
        }
        using SE = ::mlx::core::ShapeElem;
        // Grouped shape: (B, G, C/G, *S)
        ::mlx::core::Shape grouped;
        grouped.reserve(N_spatial + 3);
        grouped.push_back(static_cast<SE>(B_));
        grouped.push_back(static_cast<SE>(G_));
        grouped.push_back(static_cast<SE>(Cg));
        for (int s : spatial_dims_)
            grouped.push_back(static_cast<SE>(s));
        // Mean/rstd broadcast shape: (B, G, 1, *1)
        ::mlx::core::Shape mr_shape;
        mr_shape.reserve(N_spatial + 3);
        mr_shape.push_back(static_cast<SE>(B_));
        mr_shape.push_back(static_cast<SE>(G_));
        for (int i = 0; i < N_spatial + 1; ++i)
            mr_shape.push_back(1);
        // Per-channel γ broadcast shape: (1, C, 1, *1)
        ::mlx::core::Shape brC(N_spatial + 2, 1);
        brC[1] = static_cast<SE>(C_);

        auto x_g = ::mlx::core::reshape(*gx.arr, grouped);
        auto g_g = ::mlx::core::reshape(*gG.arr, grouped);
        auto mean_g = ::mlx::core::reshape(*gM.arr, mr_shape);
        auto rstd_g = ::mlx::core::reshape(*gR.arr, mr_shape);
        auto gamma_view = ::mlx::core::reshape(*gg.arr, brC);

        auto centered = ::mlx::core::subtract(x_g, mean_g);
        auto xnorm_g = ::mlx::core::multiply(centered, rstd_g);

        // Per-channel grads (axes={0, 2, ..., N+1} on (B, C, *S)).
        auto xnorm = ::mlx::core::reshape(xnorm_g, gpu::to_mlx_shape(input_shapes_[0]));
        std::vector<int> ch_axes;
        ch_axes.reserve(N_spatial + 1);
        ch_axes.push_back(0);
        for (int i = 0; i < N_spatial; ++i)
            ch_axes.push_back(2 + i);
        auto dgamma =
            ::mlx::core::sum(::mlx::core::multiply(*gG.arr, xnorm), ch_axes, /*keepdims=*/false);
        auto dbeta = ::mlx::core::sum(*gG.arr, ch_axes, /*keepdims=*/false);

        // dx via standard formula on the grouped tensor.
        // Reduction axes on grouped (B, G, C/G, *S): {2, 3, ..., N+2}.
        std::vector<int> red_axes;
        red_axes.reserve(N_spatial + 1);
        red_axes.push_back(2);
        for (int i = 0; i < N_spatial; ++i)
            red_axes.push_back(3 + i);
        auto gx_scaled4 = ::mlx::core::multiply(gamma_view, *gG.arr);
        auto gx_scaled_g = ::mlx::core::reshape(gx_scaled4, grouped);
        auto mean1 = ::mlx::core::mean(gx_scaled_g, red_axes, /*keepdims=*/true);
        auto mean2 = ::mlx::core::mean(::mlx::core::multiply(gx_scaled_g, xnorm_g), red_axes,
                                       /*keepdims=*/true);
        auto inner = ::mlx::core::subtract(::mlx::core::subtract(gx_scaled_g, mean1),
                                           ::mlx::core::multiply(xnorm_g, mean2));
        auto dx_g = ::mlx::core::multiply(rstd_g, inner);
        auto dx = ::mlx::core::reshape(dx_g, gpu::to_mlx_shape(input_shapes_[0]));
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dgamma), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dbeta), dtype_)}};
    }

    auto dx_cpu = alloc_bytes(static_cast<std::size_t>(B_) * C_ * spatial_total, dtype_);
    auto dgamma_cpu = alloc_bytes(static_cast<std::size_t>(C_), dtype_);
    auto dbeta_cpu = alloc_bytes(static_cast<std::size_t>(C_), dtype_);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);

    if (B_ * C_ * spatial_total > 0) {
        const auto& x_cpu = std::get<CpuStorage>(saved_inputs_[0]);
        const auto& gamma_cpu = std::get<CpuStorage>(saved_inputs_[1]);
        const auto& mean_cpu = std::get<CpuStorage>(saved_mean_);
        const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd_);
        const auto& g_cpu = std::get<CpuStorage>(grad_out);
        switch (dtype_) {
            case Dtype::F32:
                group_norm_backward_typed<float>(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const float*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                    reinterpret_cast<float*>(dx_cpu.ptr.get()),
                    reinterpret_cast<float*>(dgamma_cpu.ptr.get()),
                    reinterpret_cast<float*>(dbeta_cpu.ptr.get()), B_, C_, spatial_total, G_);
                break;
            case Dtype::F64:
                group_norm_backward_typed<double>(
                    reinterpret_cast<const double*>(x_cpu.ptr.get()),
                    reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const double*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const double*>(g_cpu.ptr.get()),
                    reinterpret_cast<double*>(dx_cpu.ptr.get()),
                    reinterpret_cast<double*>(dgamma_cpu.ptr.get()),
                    reinterpret_cast<double*>(dbeta_cpu.ptr.get()), B_, C_, spatial_total, G_);
                break;
            default:
                throw NotImplementedError("group_norm backward: dtype not supported");
        }
    } else {
        if (dgamma_cpu.nbytes)
            std::memset(dgamma_cpu.ptr.get(), 0, dgamma_cpu.nbytes);
        if (dbeta_cpu.nbytes)
            std::memset(dbeta_cpu.ptr.get(), 0, dbeta_cpu.nbytes);
    }

    return {Storage{std::move(dx_cpu)}, Storage{std::move(dgamma_cpu)},
            Storage{std::move(dbeta_cpu)}};
}

TensorImplPtr group_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            int num_groups,
                            double eps) {
    return GroupNormBackward::forward(x, gamma, beta, num_groups, eps);
}

LUCID_REGISTER_OP(GroupNormBackward)

}  // namespace lucid
