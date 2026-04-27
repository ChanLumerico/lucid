#include "BatchNorm.h"

#include <cmath>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Exceptions.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/TensorImpl.h"
#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

template <> const OpSchema BatchNorm1dBackward::schema_v1{
    "batch_norm1d", 1, AmpPolicy::ForceFP32, true};
template <> const OpSchema BatchNorm2dBackward::schema_v1{
    "batch_norm", 1, AmpPolicy::ForceFP32, true};   // keep "batch_norm" for backwards compat
template <> const OpSchema BatchNorm3dBackward::schema_v1{
    "batch_norm3d", 1, AmpPolicy::ForceFP32, true};

namespace {

CpuStorage alloc_bytes(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

template <typename T>
void batch_norm_forward_typed(const T* x, const T* gamma, const T* beta,
                              T* y, T* mean_per_c, T* rstd_per_c,
                              int B, int C, int spatial, double eps) {
    const std::size_t M = static_cast<std::size_t>(B) * spatial;
    const T inv_M = T{1} / static_cast<T>(M);

    for (int c = 0; c < C; ++c) {
        T mean = T{};
        for (int b = 0; b < B; ++b) {
            const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
            for (int i = 0; i < spatial; ++i) mean += xb[i];
        }
        mean *= inv_M;

        T var = T{};
        for (int b = 0; b < B; ++b) {
            const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                const T d = xb[i] - mean;
                var += d * d;
            }
        }
        var *= inv_M;
        const T rstd = T{1} / std::sqrt(var + static_cast<T>(eps));
        const T g = gamma[c];
        const T be = beta[c];
        for (int b = 0; b < B; ++b) {
            const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
            T*       yb = y + (static_cast<std::size_t>(b) * C + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                yb[i] = g * (xb[i] - mean) * rstd + be;
            }
        }
        mean_per_c[c] = mean;
        rstd_per_c[c] = rstd;
    }
}

template <typename T>
void batch_norm_backward_typed(const T* x, const T* gamma, const T* mean,
                               const T* rstd, const T* g,
                               T* dx, T* dgamma, T* dbeta,
                               int B, int C, int spatial) {
    const std::size_t M = static_cast<std::size_t>(B) * spatial;
    const T inv_M = T{1} / static_cast<T>(M);

    for (int c = 0; c < C; ++c) {
        const T m = mean[c];
        const T r = rstd[c];
        const T gc = gamma[c];

        T sum_dxn = T{};
        T sum_dxn_xn = T{};
        T sum_g = T{};
        T sum_g_xn = T{};
        for (int b = 0; b < B; ++b) {
            const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
            const T* gb = g + (static_cast<std::size_t>(b) * C + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                const T xn_i  = (xb[i] - m) * r;
                const T dxn_i = gc * gb[i];
                sum_dxn    += dxn_i;
                sum_dxn_xn += dxn_i * xn_i;
                sum_g      += gb[i];
                sum_g_xn   += gb[i] * xn_i;
            }
        }

        for (int b = 0; b < B; ++b) {
            const T* xb = x + (static_cast<std::size_t>(b) * C + c) * spatial;
            const T* gb = g + (static_cast<std::size_t>(b) * C + c) * spatial;
            T*       dxb = dx + (static_cast<std::size_t>(b) * C + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                const T xn_i  = (xb[i] - m) * r;
                const T dxn_i = gc * gb[i];
                dxb[i] = inv_M * r *
                         (static_cast<T>(M) * dxn_i - sum_dxn - xn_i * sum_dxn_xn);
            }
        }
        dgamma[c] = sum_g_xn;
        dbeta[c]  = sum_g;
    }
}

}  // namespace

template <int N>
TensorImplPtr BatchNormNdBackward<N>::forward(const TensorImplPtr& x,
                                              const TensorImplPtr& gamma,
                                              const TensorImplPtr& beta, double eps) {
    if (!x || !gamma || !beta) throw LucidError("batch_norm: null input");
    if (x->dtype_ != gamma->dtype_ || x->dtype_ != beta->dtype_)
        throw DtypeMismatch(std::string(dtype_name(x->dtype_)),
                            std::string(dtype_name(gamma->dtype_)), "batch_norm");
    if (x->device_ != gamma->device_ || x->device_ != beta->device_)
        throw DeviceMismatch(std::string(device_name(x->device_)),
                             std::string(device_name(gamma->device_)), "batch_norm");
    if (x->device_ == Device::CPU &&
        (!x->is_contiguous() || !gamma->is_contiguous() || !beta->is_contiguous()))
        throw NotImplementedError("batch_norm: non-contiguous input not supported");
    if (static_cast<int>(x->shape_.size()) != N + 2)
        throw ShapeMismatch(x->shape_, Shape{},
                            "batch_norm: x rank mismatch");
    if (gamma->shape_.size() != 1 || beta->shape_.size() != 1)
        throw ShapeMismatch(gamma->shape_, beta->shape_, "batch_norm: γ, β must be 1-D");

    const int B = static_cast<int>(x->shape_[0]);
    const int C = static_cast<int>(x->shape_[1]);
    int S[N > 0 ? N : 1];
    int spatial_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape_[2 + i]);
        spatial_total *= S[i];
    }
    if (gamma->shape_[0] != C || beta->shape_[0] != C)
        throw ShapeMismatch(gamma->shape_, x->shape_,
                            "batch_norm: γ/β must have length C");

    OpScope scope{BatchNormNdBackward<N>::schema_v1.name, x->device_, x->dtype_, x->shape_};

    Storage out_storage;
    Storage saved_mean;
    Storage saved_rstd;

    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto& gg = std::get<GpuStorage>(gamma->storage_);
        const auto& gb = std::get<GpuStorage>(beta->storage_);
        if (!gx.arr || !gg.arr || !gb.arr) {
            throw LucidError("batch_norm: null GPU input");
        }
        // Broadcast shape for γ/β/mean/rstd: (1, C, 1, ..., 1).
        ::mlx::core::Shape brC(N + 2, 1);
        brC[1] = static_cast<::mlx::core::ShapeElem>(C);
        auto g_view = ::mlx::core::reshape(*gg.arr, brC);
        auto b_view = ::mlx::core::reshape(*gb.arr, brC);

        // Reduction axes: {0, 2, 3, ..., N+1}
        std::vector<int> axes;
        axes.reserve(N + 1);
        axes.push_back(0);
        for (int i = 0; i < N; ++i) axes.push_back(2 + i);

        auto mean = ::mlx::core::mean(*gx.arr, axes, /*keepdims=*/true);
        auto centered = ::mlx::core::subtract(*gx.arr, mean);
        auto var = ::mlx::core::mean(::mlx::core::square(centered), axes,
                                     /*keepdims=*/true);
        ::mlx::core::array eps_arr(eps, gpu::to_mlx_dtype(x->dtype_));
        auto rstd = ::mlx::core::rsqrt(::mlx::core::add(var, eps_arr));
        auto xnorm = ::mlx::core::multiply(centered, rstd);
        auto y = ::mlx::core::add(::mlx::core::multiply(xnorm, g_view), b_view);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype_)};
        saved_mean = Storage{gpu::wrap_mlx_array(std::move(mean), x->dtype_)};
        saved_rstd = Storage{gpu::wrap_mlx_array(std::move(rstd), x->dtype_)};
    } else {
        auto y_cpu    = alloc_bytes(static_cast<std::size_t>(B) * C * spatial_total,
                                    x->dtype_);
        auto mean_cpu = alloc_bytes(static_cast<std::size_t>(C), x->dtype_);
        auto rstd_cpu = alloc_bytes(static_cast<std::size_t>(C), x->dtype_);
        if (B * C * spatial_total > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x->storage_);
            const auto& g_cpu = std::get<CpuStorage>(gamma->storage_);
            const auto& b_cpu = std::get<CpuStorage>(beta->storage_);
            switch (x->dtype_) {
                case Dtype::F32:
                    batch_norm_forward_typed<float>(
                        reinterpret_cast<const float*>(x_cpu.ptr.get()),
                        reinterpret_cast<const float*>(g_cpu.ptr.get()),
                        reinterpret_cast<const float*>(b_cpu.ptr.get()),
                        reinterpret_cast<float*>(y_cpu.ptr.get()),
                        reinterpret_cast<float*>(mean_cpu.ptr.get()),
                        reinterpret_cast<float*>(rstd_cpu.ptr.get()),
                        B, C, spatial_total, eps);
                    break;
                case Dtype::F64:
                    batch_norm_forward_typed<double>(
                        reinterpret_cast<const double*>(x_cpu.ptr.get()),
                        reinterpret_cast<const double*>(g_cpu.ptr.get()),
                        reinterpret_cast<const double*>(b_cpu.ptr.get()),
                        reinterpret_cast<double*>(y_cpu.ptr.get()),
                        reinterpret_cast<double*>(mean_cpu.ptr.get()),
                        reinterpret_cast<double*>(rstd_cpu.ptr.get()),
                        B, C, spatial_total, eps);
                    break;
                default:
                    throw NotImplementedError("batch_norm: dtype not supported (F32/F64)");
            }
        }
        out_storage = Storage{std::move(y_cpu)};
        saved_mean = Storage{std::move(mean_cpu)};
        saved_rstd = Storage{std::move(rstd_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            x->shape_, x->dtype_, x->device_,
                                            false);
    if (!GradMode::is_enabled() ||
        !(x->requires_grad_ || gamma->requires_grad_ || beta->requires_grad_)) {
        return out;
    }

    auto x_edge = detail::ensure_grad_fn(x);
    auto g_edge = detail::ensure_grad_fn(gamma);
    auto b_edge = detail::ensure_grad_fn(beta);

    auto bwd = std::make_shared<BatchNormNdBackward<N>>();
    bwd->input_shapes_  = {x->shape_, gamma->shape_, beta->shape_};
    bwd->out_shape_     = out->shape_;
    bwd->dtype_         = x->dtype_;
    bwd->device_        = x->device_;
    bwd->input_tensors_ = {x, gamma, beta};
    bwd->saved_inputs_  = {x->storage_, gamma->storage_, beta->storage_};
    bwd->saved_mean_    = std::move(saved_mean);
    bwd->saved_rstd_    = std::move(saved_rstd);
    bwd->B_ = B; bwd->C_ = C;
    for (int i = 0; i < N; ++i) bwd->S_[i] = S[i];
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(g_edge, 0), Edge(b_edge, 0)});
    bwd->set_saved_versions({x->version_, gamma->version_, beta->version_});

    out->grad_fn_       = std::move(bwd);
    out->is_leaf_       = false;
    out->requires_grad_ = true;
    return out;
}

template <int N>
std::vector<Storage> BatchNormNdBackward<N>::apply(Storage grad_out) {
    int spatial_total = 1;
    for (int i = 0; i < N; ++i) spatial_total *= this->S_[i];

    if (this->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(this->saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(this->saved_inputs_[1]);
        const auto& gM = std::get<GpuStorage>(this->saved_mean_);
        const auto& gR = std::get<GpuStorage>(this->saved_rstd_);
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gx.arr || !gg.arr || !gM.arr || !gR.arr || !gG.arr) {
            throw LucidError("batch_norm backward: null GPU array");
        }
        ::mlx::core::Shape brC(N + 2, 1);
        brC[1] = static_cast<::mlx::core::ShapeElem>(this->C_);
        auto gamma_view = ::mlx::core::reshape(*gg.arr, brC);
        auto centered = ::mlx::core::subtract(*gx.arr, *gM.arr);
        auto xnorm = ::mlx::core::multiply(centered, *gR.arr);
        std::vector<int> axes;
        axes.reserve(N + 1);
        axes.push_back(0);
        for (int i = 0; i < N; ++i) axes.push_back(2 + i);
        auto dgamma = ::mlx::core::sum(::mlx::core::multiply(*gG.arr, xnorm),
                                       axes, /*keepdims=*/false);
        auto dbeta = ::mlx::core::sum(*gG.arr, axes, /*keepdims=*/false);
        auto mean_g = ::mlx::core::mean(*gG.arr, axes, /*keepdims=*/true);
        auto mean_g_xn = ::mlx::core::mean(::mlx::core::multiply(*gG.arr, xnorm),
                                           axes, /*keepdims=*/true);
        auto inner = ::mlx::core::subtract(
            ::mlx::core::subtract(*gG.arr, mean_g),
            ::mlx::core::multiply(xnorm, mean_g_xn));
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(gamma_view, *gR.arr),
                                        inner);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dgamma), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dbeta), this->dtype_)}};
    }

    auto dx_cpu     = alloc_bytes(static_cast<std::size_t>(this->B_) * this->C_ * spatial_total,
                                  this->dtype_);
    auto dgamma_cpu = alloc_bytes(static_cast<std::size_t>(this->C_), this->dtype_);
    auto dbeta_cpu  = alloc_bytes(static_cast<std::size_t>(this->C_), this->dtype_);
    if (dx_cpu.nbytes) std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);

    if (this->B_ * this->C_ * spatial_total > 0) {
        const auto& x_cpu     = std::get<CpuStorage>(this->saved_inputs_[0]);
        const auto& gamma_cpu = std::get<CpuStorage>(this->saved_inputs_[1]);
        const auto& mean_cpu  = std::get<CpuStorage>(this->saved_mean_);
        const auto& rstd_cpu  = std::get<CpuStorage>(this->saved_rstd_);
        const auto& g_cpu     = std::get<CpuStorage>(grad_out);
        switch (this->dtype_) {
            case Dtype::F32:
                batch_norm_backward_typed<float>(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const float*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                    reinterpret_cast<float*>(dx_cpu.ptr.get()),
                    reinterpret_cast<float*>(dgamma_cpu.ptr.get()),
                    reinterpret_cast<float*>(dbeta_cpu.ptr.get()),
                    this->B_, this->C_, spatial_total);
                break;
            case Dtype::F64:
                batch_norm_backward_typed<double>(
                    reinterpret_cast<const double*>(x_cpu.ptr.get()),
                    reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const double*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const double*>(g_cpu.ptr.get()),
                    reinterpret_cast<double*>(dx_cpu.ptr.get()),
                    reinterpret_cast<double*>(dgamma_cpu.ptr.get()),
                    reinterpret_cast<double*>(dbeta_cpu.ptr.get()),
                    this->B_, this->C_, spatial_total);
                break;
            default:
                throw NotImplementedError("batch_norm backward: dtype not supported");
        }
    } else {
        if (dgamma_cpu.nbytes) std::memset(dgamma_cpu.ptr.get(), 0, dgamma_cpu.nbytes);
        if (dbeta_cpu.nbytes)  std::memset(dbeta_cpu.ptr.get(), 0, dbeta_cpu.nbytes);
    }
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dgamma_cpu)},
            Storage{std::move(dbeta_cpu)}};
}

template class BatchNormNdBackward<1>;
template class BatchNormNdBackward<2>;
template class BatchNormNdBackward<3>;

TensorImplPtr batch_norm1d_op(const TensorImplPtr& x, const TensorImplPtr& gamma,
                              const TensorImplPtr& beta, double eps) {
    return BatchNorm1dBackward::forward(x, gamma, beta, eps);
}
TensorImplPtr batch_norm_op(const TensorImplPtr& x, const TensorImplPtr& gamma,
                            const TensorImplPtr& beta, double eps) {
    return BatchNorm2dBackward::forward(x, gamma, beta, eps);
}
TensorImplPtr batch_norm3d_op(const TensorImplPtr& x, const TensorImplPtr& gamma,
                              const TensorImplPtr& beta, double eps) {
    return BatchNorm3dBackward::forward(x, gamma, beta, eps);
}

LUCID_REGISTER_OP(BatchNorm1dBackward)
LUCID_REGISTER_OP(BatchNorm2dBackward)
LUCID_REGISTER_OP(BatchNorm3dBackward)

}  // namespace lucid
