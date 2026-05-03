// lucid/_C/optim/Adam.cpp
//
// CPU and GPU implementations of Adam, AdamW, NAdam, and RAdam.
// The shared scalar kernel adam_step_cpu / adam_step_gpu handles both
// Adam and AdamW by toggling the decoupled_wd flag. NAdam and RAdam
// have their own loops because their update formulas diverge from
// the standard Adam structure.

#include "Adam.h"

#include <cmath>
#include <variant>

#include <mlx/ops.h>

#include "../autograd/Helpers.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "_OptimDetail.h"

namespace lucid {

using namespace lucid::optim_detail;

namespace {

// Scalar CPU loop for Adam and AdamW.
//
// decoupled_wd == true  → AdamW: apply p *= (1 - lr * wd) before the
//                          gradient step (weight decay not added to g).
// decoupled_wd == false → Adam:  add wd * p to g before the moment update.
//
// Bias corrections bc1 = 1 - beta1^t and bc2 = 1 - beta2^t are computed
// once outside the loop (step is the global step count at this call).
template <typename T>
void adam_step_cpu(T* param,
                   const T* grad,
                   T* m_buf,
                   T* v_buf,
                   std::size_t numel,
                   double lr,
                   double beta1,
                   double beta2,
                   double eps,
                   double weight_decay,
                   bool decoupled_wd,
                   std::int64_t step) {
    const T b1T = static_cast<T>(beta1);
    const T b2T = static_cast<T>(beta2);
    const T omb1 = static_cast<T>(1.0 - beta1);
    const T omb2 = static_cast<T>(1.0 - beta2);
    const T epsT = static_cast<T>(eps);
    const T lrT = static_cast<T>(lr);
    const T wdT = static_cast<T>(weight_decay);

    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    const T inv_bc1 = static_cast<T>(1.0 / bc1);
    const T inv_bc2 = static_cast<T>(1.0 / bc2);

    for (std::size_t i = 0; i < numel; ++i) {
        T g = grad[i];
        if (decoupled_wd) {
            // AdamW: decay parameter directly, leaving g unmodified.
            param[i] -= lrT * wdT * param[i];
        } else if (weight_decay != 0.0) {
            g += wdT * param[i];
        }
        m_buf[i] = b1T * m_buf[i] + omb1 * g;
        v_buf[i] = b2T * v_buf[i] + omb2 * g * g;
        const T m_hat = m_buf[i] * inv_bc1;
        const T v_hat = v_buf[i] * inv_bc2;
        param[i] -= lrT * m_hat / (std::sqrt(v_hat) + epsT);
    }
}

// MLX GPU path for Adam and AdamW. Follows the same decoupled_wd logic
// as the CPU version but expressed entirely in MLX array operations.
// Bias-correction factors are pre-computed as scalars and broadcast
// via zero-dimensional arrays.
void adam_step_gpu(GpuStorage& param_g,
                   const GpuStorage& grad_g,
                   GpuStorage& m_g,
                   GpuStorage& v_g,
                   Dtype dt,
                   double lr,
                   double beta1,
                   double beta2,
                   double eps,
                   double weight_decay,
                   bool decoupled_wd,
                   std::int64_t step) {
    if (!param_g.arr || !grad_g.arr || !m_g.arr || !v_g.arr) {
        ErrorBuilder("Adam GPU").fail("null array");
    }
    const auto mdt = gpu::to_mlx_dtype(dt);
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step));

    auto g = *grad_g.arr;
    if (decoupled_wd) {
        ::mlx::core::array f(1.0 - lr * weight_decay, mdt);
        auto new_param = ::mlx::core::multiply(*param_g.arr, f);
        param_g.arr = gpu::wrap_mlx_array(std::move(new_param), dt).arr;
    } else if (weight_decay != 0.0) {
        ::mlx::core::array wd_arr(weight_decay, mdt);
        g = ::mlx::core::add(g, ::mlx::core::multiply(wd_arr, *param_g.arr));
    }

    ::mlx::core::array b1_arr(beta1, mdt);
    ::mlx::core::array omb1(1.0 - beta1, mdt);
    ::mlx::core::array b2_arr(beta2, mdt);
    ::mlx::core::array omb2(1.0 - beta2, mdt);

    auto m_new =
        ::mlx::core::add(::mlx::core::multiply(b1_arr, *m_g.arr), ::mlx::core::multiply(omb1, g));
    auto v_new = ::mlx::core::add(::mlx::core::multiply(b2_arr, *v_g.arr),
                                  ::mlx::core::multiply(omb2, ::mlx::core::square(g)));
    m_g.arr = gpu::wrap_mlx_array(::mlx::core::array(m_new), dt).arr;
    v_g.arr = gpu::wrap_mlx_array(::mlx::core::array(v_new), dt).arr;

    ::mlx::core::array inv_bc1_arr(1.0 / bc1, mdt);
    ::mlx::core::array inv_bc2_arr(1.0 / bc2, mdt);
    ::mlx::core::array eps_arr(eps, mdt);
    ::mlx::core::array lr_arr(lr, mdt);

    auto m_hat = ::mlx::core::multiply(m_new, inv_bc1_arr);
    auto v_hat = ::mlx::core::multiply(v_new, inv_bc2_arr);
    auto denom = ::mlx::core::add(::mlx::core::sqrt(v_hat), eps_arr);
    auto step_arr = ::mlx::core::multiply(lr_arr, ::mlx::core::divide(m_hat, denom));
    auto new_param = ::mlx::core::subtract(*param_g.arr, step_arr);
    param_g.arr = gpu::wrap_mlx_array(std::move(new_param), dt).arr;
}

}  // namespace

Adam::Adam(std::vector<std::shared_ptr<TensorImpl>> params,
           double lr,
           double beta1,
           double beta2,
           double eps,
           double weight_decay,
           bool amsgrad)
    : Optimizer(std::move(params)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      amsgrad_(amsgrad),
      step_count_(0) {
    if (lr_ < 0.0)
        ErrorBuilder("Adam").fail("lr must be >= 0");
    if (beta1_ < 0.0 || beta1_ >= 1.0)
        ErrorBuilder("Adam").fail("beta1 must be in [0, 1)");
    if (beta2_ < 0.0 || beta2_ >= 1.0)
        ErrorBuilder("Adam").fail("beta2 must be in [0, 1)");
    if (eps_ < 0.0)
        ErrorBuilder("Adam").fail("eps must be >= 0");
    if (weight_decay_ < 0.0)
        ErrorBuilder("Adam").fail("weight_decay must be >= 0");
    if (amsgrad_)
        ErrorBuilder("Adam").not_implemented("amsgrad not yet supported");
}

// Allocate zero-initialized first- and second-moment buffers.
void Adam::init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) {
    if (m_.size() < params_.size())
        m_.resize(params_.size());
    if (v_.size() < params_.size())
        v_.resize(params_.size());
    m_[slot_idx] = make_zero_storage(param->shape(), param->dtype(), param->device());
    v_[slot_idx] = make_zero_storage(param->shape(), param->dtype(), param->device());
}

// Advance the global step counter on the first parameter of each step,
// then invoke the shared adam_step kernel with decoupled_wd = false.
void Adam::update_one(std::size_t slot_idx,
                      std::shared_ptr<TensorImpl>& param,
                      const Storage& grad) {
    // step_count_ is incremented exactly once per optimizer step, not
    // once per parameter, so the bias correction is consistent.
    if (slot_idx == 0)
        ++step_count_;

    if (param->device() == Device::GPU) {
        auto& pg = storage_gpu(param->mutable_storage());
        const auto& gg = storage_gpu(grad);
        auto& mg = storage_gpu(m_[slot_idx]);
        auto& vg = storage_gpu(v_[slot_idx]);
        adam_step_gpu(pg, gg, mg, vg, param->dtype(), lr_, beta1_, beta2_, eps_, weight_decay_,
                      false, step_count_);
        pg.bump_version();
        return;
    }
    auto& pc = storage_cpu(param->mutable_storage());
    const auto& gc = storage_cpu(grad);
    auto& mc = storage_cpu(m_[slot_idx]);
    auto& vc = storage_cpu(v_[slot_idx]);
    const std::size_t numel = pc.nbytes / dtype_size(param->dtype());
    switch (param->dtype()) {
    case Dtype::F32:
        adam_step_cpu<float>(
            reinterpret_cast<float*>(pc.ptr.get()), reinterpret_cast<const float*>(gc.ptr.get()),
            reinterpret_cast<float*>(mc.ptr.get()), reinterpret_cast<float*>(vc.ptr.get()), numel,
            lr_, beta1_, beta2_, eps_, weight_decay_, false, step_count_);
        break;
    case Dtype::F64:
        adam_step_cpu<double>(
            reinterpret_cast<double*>(pc.ptr.get()), reinterpret_cast<const double*>(gc.ptr.get()),
            reinterpret_cast<double*>(mc.ptr.get()), reinterpret_cast<double*>(vc.ptr.get()), numel,
            lr_, beta1_, beta2_, eps_, weight_decay_, false, step_count_);
        break;
    default:
        ErrorBuilder("Adam").not_implemented("dtype not supported (F32/F64)");
    }
    pc.bump_version();
}

AdamW::AdamW(std::vector<std::shared_ptr<TensorImpl>> params,
             double lr,
             double beta1,
             double beta2,
             double eps,
             double weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      step_count_(0) {
    if (lr_ < 0.0)
        ErrorBuilder("AdamW").fail("lr must be >= 0");
    if (beta1_ < 0.0 || beta1_ >= 1.0)
        ErrorBuilder("AdamW").fail("beta1 must be in [0, 1)");
    if (beta2_ < 0.0 || beta2_ >= 1.0)
        ErrorBuilder("AdamW").fail("beta2 must be in [0, 1)");
    if (eps_ < 0.0)
        ErrorBuilder("AdamW").fail("eps must be >= 0");
    if (weight_decay_ < 0.0)
        ErrorBuilder("AdamW").fail("weight_decay must be >= 0");
}

// Allocate zero-initialized first- and second-moment buffers.
void AdamW::init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) {
    if (m_.size() < params_.size())
        m_.resize(params_.size());
    if (v_.size() < params_.size())
        v_.resize(params_.size());
    m_[slot_idx] = make_zero_storage(param->shape(), param->dtype(), param->device());
    v_[slot_idx] = make_zero_storage(param->shape(), param->dtype(), param->device());
}

// Same as Adam::update_one but with decoupled_wd = true, directing the
// kernel to apply weight decay directly to the parameter rather than
// adding it to the gradient.
void AdamW::update_one(std::size_t slot_idx,
                       std::shared_ptr<TensorImpl>& param,
                       const Storage& grad) {
    if (slot_idx == 0)
        ++step_count_;
    if (param->device() == Device::GPU) {
        auto& pg = storage_gpu(param->mutable_storage());
        const auto& gg = storage_gpu(grad);
        auto& mg = storage_gpu(m_[slot_idx]);
        auto& vg = storage_gpu(v_[slot_idx]);
        adam_step_gpu(pg, gg, mg, vg, param->dtype(), lr_, beta1_, beta2_, eps_, weight_decay_,
                      true, step_count_);
        pg.bump_version();
        return;
    }
    auto& pc = storage_cpu(param->mutable_storage());
    const auto& gc = storage_cpu(grad);
    auto& mc = storage_cpu(m_[slot_idx]);
    auto& vc = storage_cpu(v_[slot_idx]);
    const std::size_t numel = pc.nbytes / dtype_size(param->dtype());
    switch (param->dtype()) {
    case Dtype::F32:
        adam_step_cpu<float>(
            reinterpret_cast<float*>(pc.ptr.get()), reinterpret_cast<const float*>(gc.ptr.get()),
            reinterpret_cast<float*>(mc.ptr.get()), reinterpret_cast<float*>(vc.ptr.get()), numel,
            lr_, beta1_, beta2_, eps_, weight_decay_, true, step_count_);
        break;
    case Dtype::F64:
        adam_step_cpu<double>(
            reinterpret_cast<double*>(pc.ptr.get()), reinterpret_cast<const double*>(gc.ptr.get()),
            reinterpret_cast<double*>(mc.ptr.get()), reinterpret_cast<double*>(vc.ptr.get()), numel,
            lr_, beta1_, beta2_, eps_, weight_decay_, true, step_count_);
        break;
    default:
        ErrorBuilder("AdamW").not_implemented("dtype not supported (F32/F64)");
    }
    pc.bump_version();
}

NAdam::NAdam(std::vector<std::shared_ptr<TensorImpl>> p,
             double lr,
             double b1,
             double b2,
             double eps,
             double wd,
             double mom_decay)
    : Optimizer(std::move(p)),
      lr_(lr),
      beta1_(b1),
      beta2_(b2),
      eps_(eps),
      weight_decay_(wd),
      momentum_decay_(mom_decay),
      step_count_(0) {}

// Allocate zero-initialized m and v buffers; initialize the per-parameter
// mu_product to 1.0 (the empty product).
void NAdam::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (m_.size() < params_.size())
        m_.resize(params_.size());
    if (v_.size() < params_.size())
        v_.resize(params_.size());
    if (mu_product_.size() < params_.size())
        mu_product_.resize(params_.size(), 1.0);
    m_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    v_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    mu_product_[i] = 1.0;
}

// NAdam update: compute the schedule-annealed momentum mu_t and the
// look-ahead momentum mu_{t+1}, advance mu_product_, then apply the
// two-term update that incorporates Nesterov acceleration.
void NAdam::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    if (i == 0)
        ++step_count_;
    const auto dt = p->dtype();
    const std::int64_t step = step_count_;
    // Schedule-annealed momentum coefficient for the current step.
    const double mu =
        beta1_ * (1.0 - 0.5 * std::pow(0.96, static_cast<double>(step) * momentum_decay_));
    // Look-ahead momentum coefficient for the next step.
    const double mu_next =
        beta1_ * (1.0 - 0.5 * std::pow(0.96, static_cast<double>(step + 1) * momentum_decay_));
    mu_product_[i] *= mu;
    const double mu_prod = mu_product_[i];
    const double mu_prod_next = mu_prod * mu_next;
    const double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(step));

    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& mg = gpu_get(m_[i]);
        auto& vg = gpu_get(v_[i]);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        auto new_m = ::mlx::core::add(::mlx::core::multiply(mlx_scalar(beta1_, dt), *mg.arr),
                                      ::mlx::core::multiply(mlx_scalar(1.0 - beta1_, dt), g));
        auto new_v = ::mlx::core::add(
            ::mlx::core::multiply(mlx_scalar(beta2_, dt), *vg.arr),
            ::mlx::core::multiply(mlx_scalar(1.0 - beta2_, dt), ::mlx::core::square(g)));
        gpu_replace(mg, ::mlx::core::array(new_m), dt);
        gpu_replace(vg, ::mlx::core::array(new_v), dt);
        auto denom = ::mlx::core::add(
            ::mlx::core::sqrt(::mlx::core::multiply(mlx_scalar(1.0 / bc2, dt), new_v)),
            mlx_scalar(eps_, dt));
        // term1 uses the raw gradient; term2 uses the bias-corrected first moment.
        auto term1 = ::mlx::core::multiply(mlx_scalar(lr_ * (1.0 - mu) / (1.0 - mu_prod), dt),
                                           ::mlx::core::divide(g, denom));
        auto term2 = ::mlx::core::multiply(mlx_scalar(lr_ * mu_next / (1.0 - mu_prod_next), dt),
                                           ::mlx::core::divide(new_m, denom));
        auto new_p = ::mlx::core::subtract(::mlx::core::subtract(*pg.arr, term1), term2);
        gpu_replace(pg, std::move(new_p), dt);
        pg.bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* M = cpu_ptr<T>(m_[i]);
        T* V = cpu_ptr<T>(v_[i]);
        const T b1T = static_cast<T>(beta1_);
        const T b2T = static_cast<T>(beta2_);
        const T omb1 = static_cast<T>(1.0 - beta1_);
        const T omb2 = static_cast<T>(1.0 - beta2_);
        const T epsT = static_cast<T>(eps_);
        const T wdT = static_cast<T>(weight_decay_);
        const T inv_bc2 = static_cast<T>(1.0 / bc2);
        const T c1 = static_cast<T>(lr_ * (1.0 - mu) / (1.0 - mu_prod));
        const T c2 = static_cast<T>(lr_ * mu_next / (1.0 - mu_prod_next));
        for (std::size_t k = 0; k < n; ++k) {
            T g = G[k];
            if (weight_decay_ != 0.0)
                g += wdT * P[k];
            M[k] = b1T * M[k] + omb1 * g;
            V[k] = b2T * V[k] + omb2 * g * g;
            const T denom = std::sqrt(V[k] * inv_bc2) + epsT;
            P[k] -= c1 * (g / denom) + c2 * (M[k] / denom);
        }
    };
    if (dt == Dtype::F32)
        step_cpu(reinterpret_cast<float*>(p_cpu.ptr.get()), cpu_cptr<float>(grad));
    else if (dt == Dtype::F64)
        step_cpu(reinterpret_cast<double*>(p_cpu.ptr.get()), cpu_cptr<double>(grad));
    else
        ErrorBuilder("NAdam").not_implemented("dtype not supported");
    p_cpu.bump_version();
}

RAdam::RAdam(std::vector<std::shared_ptr<TensorImpl>> p,
             double lr,
             double b1,
             double b2,
             double eps,
             double wd)
    : Optimizer(std::move(p)),
      lr_(lr),
      beta1_(b1),
      beta2_(b2),
      eps_(eps),
      weight_decay_(wd),
      step_count_(0) {}

// Allocate zero-initialized m and v buffers for this slot.
void RAdam::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (m_.size() < params_.size())
        m_.resize(params_.size());
    if (v_.size() < params_.size())
        v_.resize(params_.size());
    m_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    v_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
}

// RAdam update: compute rho_inf, rho_t, and the rectification factor r_t.
// When rho_t > 5 the variance estimate is stable and the adaptive step
// is used; otherwise fall back to a bias-corrected SGD-like step.
void RAdam::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    if (i == 0)
        ++step_count_;
    const auto dt = p->dtype();
    const std::int64_t step = step_count_;
    const double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(step));
    const double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(step));
    const double bc2_sqrt = std::sqrt(bc2);
    // Maximum SMA (simple moving average) achievable in the limit.
    const double rho_inf = 2.0 / (1.0 - beta2_) - 1.0;
    // Approximated SMA at the current step.
    const double rho_t = rho_inf - 2.0 * step * std::pow(beta2_, static_cast<double>(step)) / bc2;
    const bool use_rect = rho_t > 5.0;
    double r_t = 0.0;
    if (use_rect) {
        // Rectification factor: ratio of the variance of the adaptive
        // learning rate to the maximum achievable variance.
        r_t = std::sqrt((rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                        ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t));
    }

    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& mg = gpu_get(m_[i]);
        auto& vg = gpu_get(v_[i]);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        auto new_m = ::mlx::core::add(::mlx::core::multiply(mlx_scalar(beta1_, dt), *mg.arr),
                                      ::mlx::core::multiply(mlx_scalar(1.0 - beta1_, dt), g));
        auto new_v = ::mlx::core::add(
            ::mlx::core::multiply(mlx_scalar(beta2_, dt), *vg.arr),
            ::mlx::core::multiply(mlx_scalar(1.0 - beta2_, dt), ::mlx::core::square(g)));
        gpu_replace(mg, ::mlx::core::array(new_m), dt);
        gpu_replace(vg, ::mlx::core::array(new_v), dt);
        auto m_hat = ::mlx::core::multiply(mlx_scalar(1.0 / bc1, dt), new_m);
        if (use_rect) {
            auto adaptive = ::mlx::core::divide(
                mlx_scalar(bc2_sqrt, dt),
                ::mlx::core::add(::mlx::core::sqrt(new_v), mlx_scalar(eps_, dt)));
            auto new_p = ::mlx::core::subtract(
                *pg.arr, ::mlx::core::multiply(mlx_scalar(lr_ * r_t, dt),
                                               ::mlx::core::multiply(m_hat, adaptive)));
            gpu_replace(pg, std::move(new_p), dt);
        } else {
            auto new_p =
                ::mlx::core::subtract(*pg.arr, ::mlx::core::multiply(mlx_scalar(lr_, dt), m_hat));
            gpu_replace(pg, std::move(new_p), dt);
        }
        pg.bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* M = cpu_ptr<T>(m_[i]);
        T* V = cpu_ptr<T>(v_[i]);
        const T lrT = static_cast<T>(lr_);
        const T b1T = static_cast<T>(beta1_);
        const T b2T = static_cast<T>(beta2_);
        const T omb1 = static_cast<T>(1.0 - beta1_);
        const T omb2 = static_cast<T>(1.0 - beta2_);
        const T epsT = static_cast<T>(eps_);
        const T wdT = static_cast<T>(weight_decay_);
        const T inv_bc1 = static_cast<T>(1.0 / bc1);
        const T bc2_sqT = static_cast<T>(bc2_sqrt);
        const T r_tT = static_cast<T>(r_t);
        for (std::size_t k = 0; k < n; ++k) {
            T g = G[k];
            if (weight_decay_ != 0.0)
                g += wdT * P[k];
            M[k] = b1T * M[k] + omb1 * g;
            V[k] = b2T * V[k] + omb2 * g * g;
            const T m_hat = M[k] * inv_bc1;
            if (use_rect) {
                const T adaptive = bc2_sqT / (std::sqrt(V[k]) + epsT);
                P[k] -= lrT * r_tT * m_hat * adaptive;
            } else {
                P[k] -= lrT * m_hat;
            }
        }
    };
    if (dt == Dtype::F32)
        step_cpu(reinterpret_cast<float*>(p_cpu.ptr.get()), cpu_cptr<float>(grad));
    else if (dt == Dtype::F64)
        step_cpu(reinterpret_cast<double*>(p_cpu.ptr.get()), cpu_cptr<double>(grad));
    else
        ErrorBuilder("RAdam").not_implemented("dtype not supported");
    p_cpu.bump_version();
}

}  // namespace lucid
