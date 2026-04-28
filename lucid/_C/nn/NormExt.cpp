#include "NormExt.h"

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

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

}  // namespace

// ===================================================================
// BatchNormEval (inference-only)
// ===================================================================

const OpSchema BatchNormEvalBackward::schema_v1{"batch_norm_eval", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BatchNormEvalBackward::forward(const TensorImplPtr& x,
                                             const TensorImplPtr& mean,
                                             const TensorImplPtr& var,
                                             const TensorImplPtr& gamma,
                                             const TensorImplPtr& beta,
                                             double eps) {
    if (!x || !mean || !var || !gamma || !beta)
        throw LucidError("batch_norm_eval: null input");
    if (x->device_ == Device::CPU && !x->is_contiguous())
        throw NotImplementedError("batch_norm_eval: non-contiguous input not supported");
    if (x->shape_.size() < 2)
        throw ShapeMismatch(x->shape_, Shape{}, "batch_norm_eval: expected >=2-D x");

    const int B = static_cast<int>(x->shape_[0]);
    const int C = static_cast<int>(x->shape_[1]);
    if (mean->shape_.size() != 1 || mean->shape_[0] != C || var->shape_.size() != 1 ||
        var->shape_[0] != C || gamma->shape_.size() != 1 || gamma->shape_[0] != C ||
        beta->shape_.size() != 1 || beta->shape_[0] != C) {
        throw ShapeMismatch(mean->shape_, x->shape_, "batch_norm_eval: 1-D (C,) tensors required");
    }
    int spatial = 1;
    for (std::size_t i = 2; i < x->shape_.size(); ++i)
        spatial *= static_cast<int>(x->shape_[i]);

    OpScope scope{schema_v1.name, x->device_, x->dtype_, x->shape_};

    Storage out_storage;
    Storage rstd_storage;
    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto& gm = std::get<GpuStorage>(mean->storage_);
        const auto& gv = std::get<GpuStorage>(var->storage_);
        const auto& gg = std::get<GpuStorage>(gamma->storage_);
        const auto& gb = std::get<GpuStorage>(beta->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(x->dtype_);
        auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps)), mlx_dt);
        auto rstd = ::mlx::core::rsqrt(::mlx::core::add(*gv.arr, eps_arr));

        // Reshape mean/rstd/gamma/beta to broadcast against [B, C, *spatial].
        ::mlx::core::Shape b_shape(x->shape_.size(), 1);
        b_shape[1] = C;
        auto m_b = ::mlx::core::reshape(*gm.arr, b_shape);
        auto r_b = ::mlx::core::reshape(rstd, b_shape);
        auto g_b = ::mlx::core::reshape(*gg.arr, b_shape);
        auto bb_b = ::mlx::core::reshape(*gb.arr, b_shape);
        auto y = ::mlx::core::add(
            ::mlx::core::multiply(g_b,
                                  ::mlx::core::multiply(::mlx::core::subtract(*gx.arr, m_b), r_b)),
            bb_b);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype_)};
        rstd_storage = Storage{gpu::wrap_mlx_array(std::move(rstd), x->dtype_)};
    } else {
        // Compute rstd = 1/sqrt(var + eps), one value per channel.
        auto rstd_cpu = allocate_size(static_cast<std::size_t>(C), x->dtype_);
        auto out_cpu = allocate_size(static_cast<std::size_t>(B) * C * spatial, x->dtype_);
        const auto& xs = std::get<CpuStorage>(x->storage_);
        const auto& ms = std::get<CpuStorage>(mean->storage_);
        const auto& vs = std::get<CpuStorage>(var->storage_);
        const auto& gs = std::get<CpuStorage>(gamma->storage_);
        const auto& bs = std::get<CpuStorage>(beta->storage_);

        switch (x->dtype_) {
            case Dtype::F32: {
                auto* xp = reinterpret_cast<const float*>(xs.ptr.get());
                auto* mp = reinterpret_cast<const float*>(ms.ptr.get());
                auto* vp = reinterpret_cast<const float*>(vs.ptr.get());
                auto* gp = reinterpret_cast<const float*>(gs.ptr.get());
                auto* bp = reinterpret_cast<const float*>(bs.ptr.get());
                auto* yp = reinterpret_cast<float*>(out_cpu.ptr.get());
                auto* rp = reinterpret_cast<float*>(rstd_cpu.ptr.get());
                for (int c = 0; c < C; ++c)
                    rp[c] = 1.f / std::sqrt(vp[c] + static_cast<float>(eps));
                for (int b = 0; b < B; ++b) {
                    for (int c = 0; c < C; ++c) {
                        const float m = mp[c];
                        const float r = rp[c];
                        const float g = gp[c];
                        const float bb = bp[c];
                        auto* xrow = xp + (b * C + c) * spatial;
                        auto* yrow = yp + (b * C + c) * spatial;
                        for (int s = 0; s < spatial; ++s)
                            yrow[s] = g * (xrow[s] - m) * r + bb;
                    }
                }
                break;
            }
            case Dtype::F64: {
                auto* xp = reinterpret_cast<const double*>(xs.ptr.get());
                auto* mp = reinterpret_cast<const double*>(ms.ptr.get());
                auto* vp = reinterpret_cast<const double*>(vs.ptr.get());
                auto* gp = reinterpret_cast<const double*>(gs.ptr.get());
                auto* bp = reinterpret_cast<const double*>(bs.ptr.get());
                auto* yp = reinterpret_cast<double*>(out_cpu.ptr.get());
                auto* rp = reinterpret_cast<double*>(rstd_cpu.ptr.get());
                for (int c = 0; c < C; ++c)
                    rp[c] = 1.0 / std::sqrt(vp[c] + eps);
                for (int b = 0; b < B; ++b) {
                    for (int c = 0; c < C; ++c) {
                        const double m = mp[c];
                        const double r = rp[c];
                        const double g = gp[c];
                        const double bb = bp[c];
                        auto* xrow = xp + (b * C + c) * spatial;
                        auto* yrow = yp + (b * C + c) * spatial;
                        for (int s = 0; s < spatial; ++s)
                            yrow[s] = g * (xrow[s] - m) * r + bb;
                    }
                }
                break;
            }
            default:
                throw NotImplementedError("batch_norm_eval: dtype not supported");
        }
        out_storage = Storage{std::move(out_cpu)};
        rstd_storage = Storage{std::move(rstd_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x->shape_, x->dtype_,
                                            x->device_, false);

    if (!GradMode::is_enabled() ||
        !(x->requires_grad_ || gamma->requires_grad_ || beta->requires_grad_)) {
        return out;
    }

    auto x_edge = detail::ensure_grad_fn(x);
    auto m_edge = detail::ensure_grad_fn(mean);
    auto v_edge = detail::ensure_grad_fn(var);
    auto g_edge = detail::ensure_grad_fn(gamma);
    auto bb_edge = detail::ensure_grad_fn(beta);
    auto bwd = std::make_shared<BatchNormEvalBackward>();
    bwd->input_shapes_ = {x->shape_, mean->shape_, var->shape_, gamma->shape_, beta->shape_};
    bwd->out_shape_ = x->shape_;
    bwd->dtype_ = x->dtype_;
    bwd->device_ = x->device_;
    bwd->input_tensors_ = {x, mean, var, gamma, beta};
    bwd->saved_inputs_ = {x->storage_, mean->storage_, var->storage_, gamma->storage_,
                          beta->storage_};
    bwd->eps_ = eps;
    bwd->rstd_ = std::move(rstd_storage);
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(m_edge, 0), Edge(v_edge, 0),
                                          Edge(g_edge, 0), Edge(bb_edge, 0)});
    bwd->set_saved_versions(
        {x->version_, mean->version_, var->version_, gamma->version_, beta->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> BatchNormEvalBackward::apply(Storage grad_out) {
    const Shape& xs = this->input_shapes_[0];
    const int B = static_cast<int>(xs[0]);
    const int C = static_cast<int>(xs[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < xs.size(); ++i)
        spatial *= static_cast<int>(xs[i]);
    const std::size_t numel = static_cast<std::size_t>(B) * C * spatial;

    if (this->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(this->saved_inputs_[0]);
        const auto& gm = std::get<GpuStorage>(this->saved_inputs_[1]);
        const auto& gg = std::get<GpuStorage>(this->saved_inputs_[3]);
        const auto& rs = std::get<GpuStorage>(this->rstd_);
        const auto& go = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(this->dtype_);

        ::mlx::core::Shape b_shape(xs.size(), 1);
        b_shape[1] = C;
        auto m_b = ::mlx::core::reshape(*gm.arr, b_shape);
        auto r_b = ::mlx::core::reshape(*rs.arr, b_shape);
        auto g_b = ::mlx::core::reshape(*gg.arr, b_shape);

        auto x_minus_m = ::mlx::core::subtract(*gx.arr, m_b);
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(g_b, r_b), *go.arr);

        // Reduce axes: all except channel (=1).
        std::vector<int> reduce_axes;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            if (i != 1)
                reduce_axes.push_back(static_cast<int>(i));
        }
        auto sum_g_per_c = ::mlx::core::sum(*go.arr, reduce_axes, /*keepdims=*/false);
        auto xm_g = ::mlx::core::multiply(x_minus_m, *go.arr);
        auto sum_xm_g_per_c = ::mlx::core::sum(xm_g, reduce_axes, /*keepdims=*/false);

        auto db = sum_g_per_c;
        auto dg = ::mlx::core::multiply(sum_xm_g_per_c, *rs.arr);
        auto dm = ::mlx::core::negative(
            ::mlx::core::multiply(*gg.arr, ::mlx::core::multiply(*rs.arr, sum_g_per_c)));
        auto half = ::mlx::core::astype(::mlx::core::array(-0.5f), mlx_dt);
        auto r3 = ::mlx::core::multiply(*rs.arr, ::mlx::core::multiply(*rs.arr, *rs.arr));
        auto dv = ::mlx::core::multiply(
            half, ::mlx::core::multiply(*gg.arr, ::mlx::core::multiply(r3, sum_xm_g_per_c)));

        return {Storage{gpu::wrap_mlx_array(std::move(dx), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dm), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dv), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dg), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(db), this->dtype_)}};
    }

    auto dx_cpu = allocate_size(numel, this->dtype_);
    auto dm_cpu = allocate_size(static_cast<std::size_t>(C), this->dtype_);
    auto dv_cpu = allocate_size(static_cast<std::size_t>(C), this->dtype_);
    auto dg_cpu = allocate_size(static_cast<std::size_t>(C), this->dtype_);
    auto db_cpu = allocate_size(static_cast<std::size_t>(C), this->dtype_);

    const auto& xs_ = std::get<CpuStorage>(this->saved_inputs_[0]);
    const auto& ms = std::get<CpuStorage>(this->saved_inputs_[1]);
    const auto& gs = std::get<CpuStorage>(this->saved_inputs_[3]);
    const auto& gg = std::get<CpuStorage>(grad_out);
    const auto& rs = std::get<CpuStorage>(this->rstd_);

    switch (this->dtype_) {
        case Dtype::F32: {
            auto* xp = reinterpret_cast<const float*>(xs_.ptr.get());
            auto* mp = reinterpret_cast<const float*>(ms.ptr.get());
            auto* gp = reinterpret_cast<const float*>(gs.ptr.get());
            auto* gop = reinterpret_cast<const float*>(gg.ptr.get());
            auto* rp = reinterpret_cast<const float*>(rs.ptr.get());
            auto* dxp = reinterpret_cast<float*>(dx_cpu.ptr.get());
            auto* dmp = reinterpret_cast<float*>(dm_cpu.ptr.get());
            auto* dvp = reinterpret_cast<float*>(dv_cpu.ptr.get());
            auto* dgp = reinterpret_cast<float*>(dg_cpu.ptr.get());
            auto* dbp = reinterpret_cast<float*>(db_cpu.ptr.get());
            for (int c = 0; c < C; ++c) {
                dmp[c] = 0.f;
                dvp[c] = 0.f;
                dgp[c] = 0.f;
                dbp[c] = 0.f;
                const float r = rp[c], g = gp[c], m = mp[c];
                float sum_g = 0.f, sum_xm_g = 0.f;
                for (int b = 0; b < B; ++b) {
                    auto* xr = xp + (b * C + c) * spatial;
                    auto* go = gop + (b * C + c) * spatial;
                    auto* dx = dxp + (b * C + c) * spatial;
                    for (int s = 0; s < spatial; ++s) {
                        dx[s] = g * r * go[s];
                        sum_g += go[s];
                        sum_xm_g += (xr[s] - m) * go[s];
                    }
                }
                dgp[c] = sum_xm_g * r;
                dbp[c] = sum_g;
                dmp[c] = -g * r * sum_g;
                dvp[c] = -0.5f * g * r * r * r * sum_xm_g;
            }
            break;
        }
        case Dtype::F64: {
            auto* xp = reinterpret_cast<const double*>(xs_.ptr.get());
            auto* mp = reinterpret_cast<const double*>(ms.ptr.get());
            auto* gp = reinterpret_cast<const double*>(gs.ptr.get());
            auto* gop = reinterpret_cast<const double*>(gg.ptr.get());
            auto* rp = reinterpret_cast<const double*>(rs.ptr.get());
            auto* dxp = reinterpret_cast<double*>(dx_cpu.ptr.get());
            auto* dmp = reinterpret_cast<double*>(dm_cpu.ptr.get());
            auto* dvp = reinterpret_cast<double*>(dv_cpu.ptr.get());
            auto* dgp = reinterpret_cast<double*>(dg_cpu.ptr.get());
            auto* dbp = reinterpret_cast<double*>(db_cpu.ptr.get());
            for (int c = 0; c < C; ++c) {
                dmp[c] = 0.0;
                dvp[c] = 0.0;
                dgp[c] = 0.0;
                dbp[c] = 0.0;
                const double r = rp[c], g = gp[c], m = mp[c];
                double sum_g = 0.0, sum_xm_g = 0.0;
                for (int b = 0; b < B; ++b) {
                    auto* xr = xp + (b * C + c) * spatial;
                    auto* go = gop + (b * C + c) * spatial;
                    auto* dx = dxp + (b * C + c) * spatial;
                    for (int s = 0; s < spatial; ++s) {
                        dx[s] = g * r * go[s];
                        sum_g += go[s];
                        sum_xm_g += (xr[s] - m) * go[s];
                    }
                }
                dgp[c] = sum_xm_g * r;
                dbp[c] = sum_g;
                dmp[c] = -g * r * sum_g;
                dvp[c] = -0.5 * g * r * r * r * sum_xm_g;
            }
            break;
        }
        default:
            throw NotImplementedError("batch_norm_eval backward: dtype not supported");
    }
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dm_cpu)}, Storage{std::move(dv_cpu)},
            Storage{std::move(dg_cpu)}, Storage{std::move(db_cpu)}};
}

TensorImplPtr batch_norm_eval_op(const TensorImplPtr& x,
                                 const TensorImplPtr& mean,
                                 const TensorImplPtr& var,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps) {
    return BatchNormEvalBackward::forward(x, mean, var, gamma, beta, eps);
}

LUCID_REGISTER_OP(BatchNormEvalBackward)

// ===================================================================
// Lp Normalize
// ===================================================================

const OpSchema LpNormalizeBackward::schema_v1{"lp_normalize", 1, AmpPolicy::ForceFP32, true};

namespace {

template <typename T>
void lp_normalize_typed(
    const T* x, T* y, T* norm, const Shape& shape, int axis, double ord, double eps) {
    // Compute per-slice norm along `axis` then divide.
    const int rank = static_cast<int>(shape.size());
    int outer = 1, axis_len = 1, inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= static_cast<int>(shape[i]);
    axis_len = static_cast<int>(shape[axis]);
    for (int i = axis + 1; i < rank; ++i)
        inner *= static_cast<int>(shape[i]);

    // norm shape is (outer, 1, inner) flattened to (outer*inner) entries.
    for (int o = 0; o < outer; ++o) {
        for (int n = 0; n < inner; ++n) {
            T acc = T{0};
            for (int a = 0; a < axis_len; ++a) {
                const T v = x[(o * axis_len + a) * inner + n];
                acc += static_cast<T>(std::pow(std::abs(static_cast<double>(v)), ord));
            }
            const T nm = static_cast<T>(std::pow(static_cast<double>(acc), 1.0 / ord));
            const T denom = nm > static_cast<T>(eps) ? nm : static_cast<T>(eps);
            norm[o * inner + n] = denom;
            for (int a = 0; a < axis_len; ++a) {
                const std::size_t idx = (o * axis_len + a) * inner + n;
                y[idx] = x[idx] / denom;
            }
        }
    }
}

}  // namespace

TensorImplPtr LpNormalizeBackward::forward(const TensorImplPtr& x,
                                           double ord,
                                           int axis,
                                           double eps) {
    if (!x)
        throw LucidError("lp_normalize: null input");
    if (x->device_ == Device::CPU && !x->is_contiguous())
        throw NotImplementedError("lp_normalize: non-contiguous input not supported");
    const int rank = static_cast<int>(x->shape_.size());
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
        throw LucidError("lp_normalize: axis out of range");

    OpScope scope{schema_v1.name, x->device_, x->dtype_, x->shape_};
    const std::size_t numel = x->numel();
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= static_cast<int>(x->shape_[i]);
    for (int i = axis + 1; i < rank; ++i)
        inner *= static_cast<int>(x->shape_[i]);

    Storage y_storage;
    Storage norm_storage;
    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(x->dtype_);
        // norm = (sum |x|^ord)^(1/ord) along axis, keepdims=true
        auto abs_x = ::mlx::core::abs(*gx.arr);
        auto ord_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(ord)), mlx_dt);
        auto inv_ord =
            ::mlx::core::astype(::mlx::core::array(static_cast<float>(1.0 / ord)), mlx_dt);
        auto pow_x = ::mlx::core::power(abs_x, ord_arr);
        auto sum_p = ::mlx::core::sum(pow_x, std::vector<int>{axis}, /*keepdims=*/true);
        auto N = ::mlx::core::power(sum_p, inv_ord);
        auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps)), mlx_dt);
        auto N_clip = ::mlx::core::maximum(N, eps_arr);
        auto y = ::mlx::core::divide(*gx.arr, N_clip);
        y_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype_)};
        norm_storage = Storage{gpu::wrap_mlx_array(std::move(N_clip), x->dtype_)};
    } else {
        auto y_cpu = allocate_size(numel, x->dtype_);
        auto norm_cpu = allocate_size(static_cast<std::size_t>(outer) * inner, x->dtype_);
        const auto& xs = std::get<CpuStorage>(x->storage_);

        switch (x->dtype_) {
            case Dtype::F32:
                lp_normalize_typed<float>(reinterpret_cast<const float*>(xs.ptr.get()),
                                          reinterpret_cast<float*>(y_cpu.ptr.get()),
                                          reinterpret_cast<float*>(norm_cpu.ptr.get()), x->shape_,
                                          axis, ord, eps);
                break;
            case Dtype::F64:
                lp_normalize_typed<double>(reinterpret_cast<const double*>(xs.ptr.get()),
                                           reinterpret_cast<double*>(y_cpu.ptr.get()),
                                           reinterpret_cast<double*>(norm_cpu.ptr.get()), x->shape_,
                                           axis, ord, eps);
                break;
            default:
                throw NotImplementedError("lp_normalize: dtype not supported");
        }
        y_storage = Storage{std::move(y_cpu)};
        norm_storage = Storage{std::move(norm_cpu)};
    }

    auto out =
        std::make_shared<TensorImpl>(std::move(y_storage), x->shape_, x->dtype_, x->device_, false);

    if (!GradMode::is_enabled() || !x->requires_grad_)
        return out;

    auto x_edge = detail::ensure_grad_fn(x);
    auto bwd = std::make_shared<LpNormalizeBackward>();
    bwd->input_shapes_ = {x->shape_};
    bwd->out_shape_ = x->shape_;
    bwd->dtype_ = x->dtype_;
    bwd->device_ = x->device_;
    bwd->input_tensors_ = {x};
    bwd->saved_inputs_ = {x->storage_};
    bwd->ord_ = ord;
    bwd->axis_ = axis;
    bwd->eps_ = eps;
    bwd->saved_norm_ = std::move(norm_storage);
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions({x->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

namespace {

template <typename T>
void lp_normalize_grad_typed(
    const T* x, const T* g_out, T* dx, const T* norm, const Shape& shape, int axis, double ord) {
    // y_i = x_i / N(x), where N(x) = (sum |x_j|^p)^(1/p) clipped at eps
    // dy_i/dx_j = δ_ij / N − x_i · sign(x_j) · |x_j|^(p-1) / N^(p+1)
    // Hence: dx_j = (g_j / N) − sign(x_j)·|x_j|^(p-1)/N^(p+1) · dot(g, x)
    //         where dot(g, x) = sum_i g_i · x_i along the axis.
    const int rank = static_cast<int>(shape.size());
    int outer = 1, axis_len = static_cast<int>(shape[axis]), inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= static_cast<int>(shape[i]);
    for (int i = axis + 1; i < rank; ++i)
        inner *= static_cast<int>(shape[i]);

    for (int o = 0; o < outer; ++o) {
        for (int n = 0; n < inner; ++n) {
            const T N = norm[o * inner + n];
            // Sum_i g_i · x_i along axis (the projection onto x).
            T proj = T{0};
            for (int a = 0; a < axis_len; ++a) {
                const std::size_t idx = (o * axis_len + a) * inner + n;
                proj += g_out[idx] * x[idx];
            }
            const T pow_factor = static_cast<T>(std::pow(static_cast<double>(N), ord + 1.0));
            for (int a = 0; a < axis_len; ++a) {
                const std::size_t idx = (o * axis_len + a) * inner + n;
                const T xi = x[idx];
                const T sgn = (xi > T{0}) ? T{1} : (xi < T{0} ? T{-1} : T{0});
                const T abs_pm1 =
                    static_cast<T>(std::pow(std::abs(static_cast<double>(xi)), ord - 1.0));
                dx[idx] = g_out[idx] / N - sgn * abs_pm1 * proj / pow_factor;
            }
        }
    }
}

}  // namespace

std::vector<Storage> LpNormalizeBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(this->out_shape_);

    if (this->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(this->saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto& gn = std::get<GpuStorage>(this->saved_norm_);
        const auto mlx_dt = gpu::to_mlx_dtype(this->dtype_);
        // dx_j = g_j / N - sgn(x_j) * |x_j|^(p-1) / N^(p+1) * proj
        // proj = sum_along_axis(g * x), keepdims
        auto proj = ::mlx::core::sum(::mlx::core::multiply(*gg.arr, *gx.arr),
                                     std::vector<int>{axis_}, /*keepdims=*/true);
        auto first = ::mlx::core::divide(*gg.arr, *gn.arr);
        auto sign_x = ::mlx::core::sign(*gx.arr);
        auto abs_x = ::mlx::core::abs(*gx.arr);
        auto ord_m1 =
            ::mlx::core::astype(::mlx::core::array(static_cast<float>(ord_ - 1.0)), mlx_dt);
        auto ord_p1 =
            ::mlx::core::astype(::mlx::core::array(static_cast<float>(ord_ + 1.0)), mlx_dt);
        auto abs_pm1 = ::mlx::core::power(abs_x, ord_m1);
        auto N_pp1 = ::mlx::core::power(*gn.arr, ord_p1);
        auto second = ::mlx::core::divide(
            ::mlx::core::multiply(::mlx::core::multiply(sign_x, abs_pm1), proj), N_pp1);
        auto dx = ::mlx::core::subtract(first, second);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), this->dtype_)}};
    }

    auto dx_cpu = allocate_size(numel, this->dtype_);
    const auto& xs = std::get<CpuStorage>(this->saved_inputs_[0]);
    const auto& gs = std::get<CpuStorage>(grad_out);
    const auto& ns = std::get<CpuStorage>(this->saved_norm_);

    switch (this->dtype_) {
        case Dtype::F32:
            lp_normalize_grad_typed<float>(reinterpret_cast<const float*>(xs.ptr.get()),
                                           reinterpret_cast<const float*>(gs.ptr.get()),
                                           reinterpret_cast<float*>(dx_cpu.ptr.get()),
                                           reinterpret_cast<const float*>(ns.ptr.get()),
                                           this->out_shape_, axis_, ord_);
            break;
        case Dtype::F64:
            lp_normalize_grad_typed<double>(reinterpret_cast<const double*>(xs.ptr.get()),
                                            reinterpret_cast<const double*>(gs.ptr.get()),
                                            reinterpret_cast<double*>(dx_cpu.ptr.get()),
                                            reinterpret_cast<const double*>(ns.ptr.get()),
                                            this->out_shape_, axis_, ord_);
            break;
        default:
            throw NotImplementedError("lp_normalize backward: dtype not supported");
    }
    return {Storage{std::move(dx_cpu)}};
}

TensorImplPtr lp_normalize_op(const TensorImplPtr& x, double ord, int axis, double eps) {
    return LpNormalizeBackward::forward(x, ord, axis, eps);
}

LUCID_REGISTER_OP(LpNormalizeBackward)

// ===================================================================
// Global Response Norm (ConvNeXt-v2)
// ===================================================================

const OpSchema GlobalResponseNormBackward::schema_v1{"global_response_norm", 1,
                                                     AmpPolicy::ForceFP32, true};

namespace {

// Forward in NCHW:
//   Gx[b, c]   = ||x[b, c, :, :]||_2          shape (B, C)
//   meanGx[b]  = mean(Gx[b, :])               shape (B,)
//   Nx[b, c]   = Gx[b, c] / (meanGx[b] + eps) shape (B, C)
//   y = γ · (x · Nx_bc_broadcast) + β · x
template <typename T>
void grn_forward_typed(const T* x,
                       T* y,
                       T* Nx,
                       int B,
                       int C,
                       int H,
                       int W,
                       const T* gamma,
                       const T* beta,
                       double eps) {
    const int spatial = H * W;
    std::vector<T> Gx(static_cast<std::size_t>(B) * C);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            T acc = T{0};
            const T* xr = x + (b * C + c) * spatial;
            for (int s = 0; s < spatial; ++s)
                acc += xr[s] * xr[s];
            Gx[b * C + c] = static_cast<T>(std::sqrt(static_cast<double>(acc)));
        }
    }
    for (int b = 0; b < B; ++b) {
        T mean = T{0};
        for (int c = 0; c < C; ++c)
            mean += Gx[b * C + c];
        mean /= static_cast<T>(C);
        const T denom = mean + static_cast<T>(eps);
        for (int c = 0; c < C; ++c) {
            const T nx = Gx[b * C + c] / denom;
            Nx[b * C + c] = nx;
            const T* xr = x + (b * C + c) * spatial;
            T* yr = y + (b * C + c) * spatial;
            const T g = gamma[c];
            const T bb = beta[c];
            for (int s = 0; s < spatial; ++s) {
                yr[s] = g * (xr[s] * nx) + bb * xr[s];
            }
        }
    }
}

}  // namespace

TensorImplPtr GlobalResponseNormBackward::forward(const TensorImplPtr& x,
                                                  const TensorImplPtr& gamma,
                                                  const TensorImplPtr& beta,
                                                  double eps) {
    if (!x || !gamma || !beta)
        throw LucidError("global_response_norm: null input");
    if (x->device_ == Device::CPU && !x->is_contiguous())
        throw NotImplementedError("global_response_norm: non-contiguous input");
    if (x->shape_.size() != 4)
        throw ShapeMismatch(x->shape_, Shape{}, "global_response_norm: x must be 4-D");

    const int B = static_cast<int>(x->shape_[0]);
    const int C = static_cast<int>(x->shape_[1]);
    const int H = static_cast<int>(x->shape_[2]);
    const int W = static_cast<int>(x->shape_[3]);
    if (gamma->numel() != static_cast<std::size_t>(C) ||
        beta->numel() != static_cast<std::size_t>(C))
        throw ShapeMismatch(gamma->shape_, x->shape_,
                            "global_response_norm: gamma/beta must have C elements");

    OpScope scope{schema_v1.name, x->device_, x->dtype_, x->shape_};
    const std::size_t numel = x->numel();

    Storage out_storage;
    Storage nx_storage;
    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto& gg = std::get<GpuStorage>(gamma->storage_);
        const auto& gb = std::get<GpuStorage>(beta->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(x->dtype_);
        // G[b,c] = sqrt(sum_(h,w) x²)  shape [B,C,1,1]
        auto x_sq = ::mlx::core::multiply(*gx.arr, *gx.arr);
        auto G_sq = ::mlx::core::sum(x_sq, std::vector<int>{2, 3}, /*keepdims=*/true);
        auto G = ::mlx::core::sqrt(G_sq);
        // m[b] = mean over channel  shape [B,1,1,1]
        auto m = ::mlx::core::mean(G, std::vector<int>{1}, /*keepdims=*/true);
        auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps)), mlx_dt);
        auto denom = ::mlx::core::add(m, eps_arr);
        auto Nx = ::mlx::core::divide(G, denom);  // [B,C,1,1]
        // gamma/beta reshape to [1,C,1,1]
        auto g_b = ::mlx::core::reshape(*gg.arr, {1, C, 1, 1});
        auto bb_b = ::mlx::core::reshape(*gb.arr, {1, C, 1, 1});
        auto y = ::mlx::core::add(::mlx::core::multiply(g_b, ::mlx::core::multiply(*gx.arr, Nx)),
                                  ::mlx::core::multiply(bb_b, *gx.arr));
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype_)};
        // Save Nx (broadcast to [B, C, 1, 1] reshape to [B, C] for layout consistency).
        auto Nx_flat = ::mlx::core::reshape(Nx, {B, C});
        nx_storage = Storage{gpu::wrap_mlx_array(std::move(Nx_flat), x->dtype_)};
    } else {
        auto y_cpu = allocate_size(numel, x->dtype_);
        auto nx_cpu = allocate_size(static_cast<std::size_t>(B) * C, x->dtype_);

        const auto& xs = std::get<CpuStorage>(x->storage_);
        const auto& gs = std::get<CpuStorage>(gamma->storage_);
        const auto& bs = std::get<CpuStorage>(beta->storage_);

        switch (x->dtype_) {
            case Dtype::F32:
                grn_forward_typed<float>(reinterpret_cast<const float*>(xs.ptr.get()),
                                         reinterpret_cast<float*>(y_cpu.ptr.get()),
                                         reinterpret_cast<float*>(nx_cpu.ptr.get()), B, C, H, W,
                                         reinterpret_cast<const float*>(gs.ptr.get()),
                                         reinterpret_cast<const float*>(bs.ptr.get()), eps);
                break;
            case Dtype::F64:
                grn_forward_typed<double>(reinterpret_cast<const double*>(xs.ptr.get()),
                                          reinterpret_cast<double*>(y_cpu.ptr.get()),
                                          reinterpret_cast<double*>(nx_cpu.ptr.get()), B, C, H, W,
                                          reinterpret_cast<const double*>(gs.ptr.get()),
                                          reinterpret_cast<const double*>(bs.ptr.get()), eps);
                break;
            default:
                throw NotImplementedError("global_response_norm: dtype not supported");
        }
        out_storage = Storage{std::move(y_cpu)};
        nx_storage = Storage{std::move(nx_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x->shape_, x->dtype_,
                                            x->device_, false);

    if (!GradMode::is_enabled() ||
        !(x->requires_grad_ || gamma->requires_grad_ || beta->requires_grad_)) {
        return out;
    }

    auto x_edge = detail::ensure_grad_fn(x);
    auto g_edge = detail::ensure_grad_fn(gamma);
    auto bb_edge = detail::ensure_grad_fn(beta);
    auto bwd = std::make_shared<GlobalResponseNormBackward>();
    bwd->input_shapes_ = {x->shape_, gamma->shape_, beta->shape_};
    bwd->out_shape_ = x->shape_;
    bwd->dtype_ = x->dtype_;
    bwd->device_ = x->device_;
    bwd->input_tensors_ = {x, gamma, beta};
    bwd->saved_inputs_ = {x->storage_, gamma->storage_, beta->storage_};
    bwd->eps_ = eps;
    bwd->saved_Nx_ = std::move(nx_storage);
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(g_edge, 0), Edge(bb_edge, 0)});
    bwd->set_saved_versions({x->version_, gamma->version_, beta->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

namespace {

// Backward.
// Let G[b,c] = ||x[b,c,:,:]||_2,  m[b] = mean(G[b,:]),
//     N[b,c] = G[b,c] / (m[b] + eps),
//     y[b,c,s] = γ_c · x[b,c,s] · N[b,c] + β_c · x[b,c,s]
//
// dβ_c = sum_{b,s} g[b,c,s] · x[b,c,s]
// dγ_c = sum_{b,s} g[b,c,s] · x[b,c,s] · N[b,c]
// For dx, we differentiate y w.r.t. x[b,c,s]:
//   dy/dx[b,c,s] = γ_c · (N[b,c] + x[b,c,s] · ∂N[b,c]/∂x[b,c,s]) + β_c
//                + γ_c · x[b,c,s'] · ∂N[b,c']/∂x[b,c,s] for c' ≠ c, s'
// This is intricate; we use the chain rule via G, m. Define:
//   dG_bc = ∂L/∂G[b,c]  (depends on ∂y/∂N[b,c'])
// Two-step approach (see legacy lucid):
//   For each (b,c): denote A_bc = sum_s y_grad_bcs · x_bcs · γ_c
//   ∂L/∂N[b,c] = A_bc
//   ∂L/∂G[b,c] = ∂L/∂N[b,c] / (m[b] + eps)
//                + sum_{c'} ∂L/∂N[b,c'] · (-G[b,c']/(m[b]+eps)²) · (1/C)
//   ∂L/∂x[b,c,s] = γ_c·N[b,c]·g_bcs + β_c·g_bcs + ∂L/∂G[b,c] · x[b,c,s]/G[b,c]
// (G[b,c]==0 → that fragment is zero — clamp to avoid div-by-zero.)
template <typename T>
void grn_backward_typed(const T* x,
                        const T* g_out,
                        const T* gamma,
                        const T* beta,
                        const T* Nx,
                        T* dx,
                        T* dgamma,
                        T* dbeta,
                        int B,
                        int C,
                        int H,
                        int W,
                        double eps) {
    const int spatial = H * W;
    std::vector<T> A(static_cast<std::size_t>(B) * C, T{0});
    std::vector<T> Gx(static_cast<std::size_t>(B) * C, T{0});
    std::vector<T> m(static_cast<std::size_t>(B), T{0});

    // Initialize dgamma / dbeta.
    for (int c = 0; c < C; ++c) {
        dgamma[c] = T{0};
        dbeta[c] = T{0};
    }

    // Recompute G and m from x (cheaper than saving them).
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            T acc = T{0};
            const T* xr = x + (b * C + c) * spatial;
            for (int s = 0; s < spatial; ++s)
                acc += xr[s] * xr[s];
            Gx[b * C + c] = static_cast<T>(std::sqrt(static_cast<double>(acc)));
        }
        T sum = T{0};
        for (int c = 0; c < C; ++c)
            sum += Gx[b * C + c];
        m[b] = sum / static_cast<T>(C);
    }

    // Compute A[b,c] = sum_s g·x · γ_c, plus dγ, dβ.
    // Forward is y = γ_c·(x·N) + β_c·x  (β multiplies x), so dβ = sum_s g·x.
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xr = x + (b * C + c) * spatial;
            const T* gr = g_out + (b * C + c) * spatial;
            T sum_gx = T{0};
            for (int s = 0; s < spatial; ++s) {
                sum_gx += gr[s] * xr[s];
            }
            const T nbc = Nx[b * C + c];
            A[b * C + c] = sum_gx * gamma[c];  // ∂L/∂N[b,c]
            dgamma[c] += sum_gx * nbc;
            dbeta[c] += sum_gx;
        }
    }

    // Compute dG[b,c] from A and a per-batch correction term.
    std::vector<T> dG(static_cast<std::size_t>(B) * C, T{0});
    for (int b = 0; b < B; ++b) {
        const T denom = m[b] + static_cast<T>(eps);
        const T denom2 = denom * denom;
        T sum_A_G = T{0};
        for (int c = 0; c < C; ++c)
            sum_A_G += A[b * C + c] * Gx[b * C + c];
        const T common = -sum_A_G / denom2 / static_cast<T>(C);
        for (int c = 0; c < C; ++c) {
            dG[b * C + c] = A[b * C + c] / denom + common;
        }
    }

    // Final: dx = γ_c·N·g + β_c·g + dG[b,c] · x/G[b,c]   (when G > 0).
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T nbc = Nx[b * C + c];
            const T gG = Gx[b * C + c];
            const T inv_G = (gG > T{0}) ? T{1} / gG : T{0};
            const T dG_bc = dG[b * C + c];
            const T gc = gamma[c];
            const T bc = beta[c];
            const T* xr = x + (b * C + c) * spatial;
            const T* gr = g_out + (b * C + c) * spatial;
            T* dxr = dx + (b * C + c) * spatial;
            for (int s = 0; s < spatial; ++s) {
                dxr[s] = gc * nbc * gr[s] + bc * gr[s] + dG_bc * xr[s] * inv_G;
            }
        }
    }
}

}  // namespace

std::vector<Storage> GlobalResponseNormBackward::apply(Storage grad_out) {
    const Shape& xs = this->input_shapes_[0];
    const int B = static_cast<int>(xs[0]);
    const int C = static_cast<int>(xs[1]);
    const int H = static_cast<int>(xs[2]);
    const int W = static_cast<int>(xs[3]);
    const std::size_t numel = static_cast<std::size_t>(B) * C * H * W;

    if (this->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(this->saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(this->saved_inputs_[1]);
        const auto& gb = std::get<GpuStorage>(this->saved_inputs_[2]);
        const auto& gnx = std::get<GpuStorage>(this->saved_Nx_);
        const auto& go = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(this->dtype_);

        // Recompute G[B,C,1,1] and m[B,1,1,1].
        auto x_sq = ::mlx::core::multiply(*gx.arr, *gx.arr);
        auto G_sq = ::mlx::core::sum(x_sq, std::vector<int>{2, 3}, true);
        auto G = ::mlx::core::sqrt(G_sq);                            // [B,C,1,1]
        auto m_b = ::mlx::core::mean(G, std::vector<int>{1}, true);  // [B,1,1,1]
        auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps_)), mlx_dt);
        auto denom = ::mlx::core::add(m_b, eps_arr);  // [B,1,1,1]
        auto Nx = ::mlx::core::reshape(*gnx.arr, {B, C, 1, 1});

        // gamma/beta as [1,C,1,1].
        auto g_b = ::mlx::core::reshape(*gg.arr, {1, C, 1, 1});
        auto bb_b = ::mlx::core::reshape(*gb.arr, {1, C, 1, 1});

        // dβ_c = sum_{b,h,w} g · x
        auto gx_prod = ::mlx::core::multiply(*go.arr, *gx.arr);
        auto db = ::mlx::core::sum(gx_prod, std::vector<int>{0, 2, 3}, false);
        // dγ_c = sum_{b,h,w} g · x · N[b,c]
        auto gxN = ::mlx::core::multiply(gx_prod, Nx);
        auto dg = ::mlx::core::sum(gxN, std::vector<int>{0, 2, 3}, false);

        // A[b,c,1,1] = γ_c · sum_{h,w} g · x = γ_c · sum(g·x, [2,3], keepdims).
        auto sum_gx = ::mlx::core::sum(gx_prod, std::vector<int>{2, 3}, true);
        auto A = ::mlx::core::multiply(g_b, sum_gx);

        // Inner sum over channel: sum_c (A[b,c] · G[b,c]) → [B,1,1,1]
        auto AG = ::mlx::core::multiply(A, G);
        auto inner_sum = ::mlx::core::sum(AG, std::vector<int>{1}, true);
        auto C_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(C)), mlx_dt);
        auto denom_sq = ::mlx::core::multiply(denom, denom);
        auto second = ::mlx::core::divide(inner_sum, ::mlx::core::multiply(denom_sq, C_arr));
        // dG[b,c] = A/denom - second
        auto dG = ::mlx::core::subtract(::mlx::core::divide(A, denom), second);

        // dx[b,c,h,w] = γ_c·N·g + β_c·g + dG · x / G
        auto eps_g = ::mlx::core::astype(::mlx::core::array(1e-12f), mlx_dt);
        auto G_safe = ::mlx::core::maximum(G, eps_g);
        auto dG_term = ::mlx::core::divide(::mlx::core::multiply(dG, *gx.arr), G_safe);
        auto dx = ::mlx::core::add(
            ::mlx::core::add(::mlx::core::multiply(::mlx::core::multiply(g_b, Nx), *go.arr),
                             ::mlx::core::multiply(bb_b, *go.arr)),
            dG_term);

        return {Storage{gpu::wrap_mlx_array(std::move(dx), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dg), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(db), this->dtype_)}};
    }

    auto dx_cpu = allocate_size(numel, this->dtype_);
    auto dg_cpu = allocate_size(static_cast<std::size_t>(C), this->dtype_);
    auto db_cpu = allocate_size(static_cast<std::size_t>(C), this->dtype_);

    const auto& x_s = std::get<CpuStorage>(this->saved_inputs_[0]);
    const auto& g_s = std::get<CpuStorage>(this->saved_inputs_[1]);
    const auto& b_s = std::get<CpuStorage>(this->saved_inputs_[2]);
    const auto& nx_s = std::get<CpuStorage>(this->saved_Nx_);
    const auto& go_s = std::get<CpuStorage>(grad_out);

    switch (this->dtype_) {
        case Dtype::F32:
            grn_backward_typed<float>(reinterpret_cast<const float*>(x_s.ptr.get()),
                                      reinterpret_cast<const float*>(go_s.ptr.get()),
                                      reinterpret_cast<const float*>(g_s.ptr.get()),
                                      reinterpret_cast<const float*>(b_s.ptr.get()),
                                      reinterpret_cast<const float*>(nx_s.ptr.get()),
                                      reinterpret_cast<float*>(dx_cpu.ptr.get()),
                                      reinterpret_cast<float*>(dg_cpu.ptr.get()),
                                      reinterpret_cast<float*>(db_cpu.ptr.get()), B, C, H, W, eps_);
            break;
        case Dtype::F64:
            grn_backward_typed<double>(reinterpret_cast<const double*>(x_s.ptr.get()),
                                       reinterpret_cast<const double*>(go_s.ptr.get()),
                                       reinterpret_cast<const double*>(g_s.ptr.get()),
                                       reinterpret_cast<const double*>(b_s.ptr.get()),
                                       reinterpret_cast<const double*>(nx_s.ptr.get()),
                                       reinterpret_cast<double*>(dx_cpu.ptr.get()),
                                       reinterpret_cast<double*>(dg_cpu.ptr.get()),
                                       reinterpret_cast<double*>(db_cpu.ptr.get()), B, C, H, W,
                                       eps_);
            break;
        default:
            throw NotImplementedError("global_response_norm backward: dtype not supported");
    }
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dg_cpu)}, Storage{std::move(db_cpu)}};
}

TensorImplPtr global_response_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps) {
    return GlobalResponseNormBackward::forward(x, gamma, beta, eps);
}

LUCID_REGISTER_OP(GlobalResponseNormBackward)

}  // namespace lucid
