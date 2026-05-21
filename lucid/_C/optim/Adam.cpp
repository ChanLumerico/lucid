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

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../autograd/Helpers.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "_OptimDetail.h"

namespace lucid {

using namespace lucid::optim_detail;

// AdamScalarCache out-of-line dtor needed because unique_ptr<mlx::core::array>
// requires a complete type to destruct, and Adam.h only forward-declares
// mlx::core::array.
AdamScalarCache::AdamScalarCache() = default;
AdamScalarCache::~AdamScalarCache() = default;

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

// Rebuild the per-step scalar cache.  Called from Adam::update_one /
// AdamW::update_one at slot_idx == 0 (first parameter of a step) when
// the cache is invalid or the dtype changed.  All scalars are constant
// across all parameters in one step, so building once amortises the
// MLX array construction cost across the whole parameter list.
void refresh_adam_scalar_cache(AdamScalarCache& cache,
                               Dtype dt,
                               double lr,
                               double beta1,
                               double beta2,
                               double eps,
                               double weight_decay,
                               bool decoupled_wd,
                               std::int64_t step) {
    const auto mdt = gpu::to_mlx_dtype(dt);
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step));

    cache.b1 = std::make_unique<::mlx::core::array>(beta1, mdt);
    cache.omb1 = std::make_unique<::mlx::core::array>(1.0 - beta1, mdt);
    cache.b2 = std::make_unique<::mlx::core::array>(beta2, mdt);
    cache.omb2 = std::make_unique<::mlx::core::array>(1.0 - beta2, mdt);
    cache.eps_a = std::make_unique<::mlx::core::array>(eps, mdt);
    cache.lr_a = std::make_unique<::mlx::core::array>(lr, mdt);
    cache.inv_bc1 = std::make_unique<::mlx::core::array>(1.0 / bc1, mdt);
    cache.inv_bc2 = std::make_unique<::mlx::core::array>(1.0 / bc2, mdt);
    // 3.4+ Phase A.5: pre-folded bias-correction scalars for the
    // simplified update (see header).  lr_eff folds in sqrt(bc2)/bc1,
    // eps_eff folds in sqrt(bc2).
    const double sqrt_bc2 = std::sqrt(bc2);
    cache.lr_eff_a = std::make_unique<::mlx::core::array>(lr * sqrt_bc2 / bc1, mdt);
    cache.eps_eff_a = std::make_unique<::mlx::core::array>(eps * sqrt_bc2, mdt);
    if (decoupled_wd) {
        cache.wd_factor = std::make_unique<::mlx::core::array>(1.0 - lr * weight_decay, mdt);
        cache.wd_a.reset();
    } else if (weight_decay != 0.0) {
        cache.wd_a = std::make_unique<::mlx::core::array>(weight_decay, mdt);
        cache.wd_factor.reset();
    } else {
        cache.wd_a.reset();
        cache.wd_factor.reset();
    }
    cache.valid = true;
    cache.dt = dt;
    cache.for_step = step;
}

// MLX GPU path for Adam and AdamW using a pre-built scalar cache.
// All 8 broadcast scalars and the optional weight-decay scalar are
// taken from ``cache`` rather than freshly constructed — that saves
// ~520 ``mlx::core::array`` constructions per step on a 65-param
// ResNet-18 (8 × 65), dropping per-step Adam cost from ~5.3 ms to
// well under 2 ms on M4 Max.
void adam_step_gpu_cached(GpuStorage& param_g,
                          const GpuStorage& grad_g,
                          GpuStorage& m_g,
                          GpuStorage& v_g,
                          Dtype dt,
                          double weight_decay,
                          bool decoupled_wd,
                          const AdamScalarCache& cache) {
    if (!param_g.arr || !grad_g.arr || !m_g.arr || !v_g.arr) {
        ErrorBuilder("Adam GPU").fail("null array");
    }

    auto g = *grad_g.arr;
    if (decoupled_wd) {
        auto new_param = ::mlx::core::multiply(*param_g.arr, *cache.wd_factor);
        param_g.arr = gpu::wrap_mlx_array(std::move(new_param), dt).arr;
    } else if (weight_decay != 0.0) {
        g = ::mlx::core::add(g, ::mlx::core::multiply(*cache.wd_a, *param_g.arr));
    }

    auto m_new = ::mlx::core::add(::mlx::core::multiply(*cache.b1, *m_g.arr),
                                  ::mlx::core::multiply(*cache.omb1, g));
    auto v_new = ::mlx::core::add(::mlx::core::multiply(*cache.b2, *v_g.arr),
                                  ::mlx::core::multiply(*cache.omb2, ::mlx::core::square(g)));
    m_g.arr = gpu::wrap_mlx_array(::mlx::core::array(m_new), dt).arr;
    v_g.arr = gpu::wrap_mlx_array(::mlx::core::array(v_new), dt).arr;

    // 3.4+ Phase A.5: simplified update folding bias corrections into
    // pre-computed scalars (lr_eff_a, eps_eff_a).  Equivalent to the prior
    //   m_hat = m_new / bc1
    //   v_hat = v_new / bc2
    //   step  = lr * m_hat / (sqrt(v_hat) + eps)
    // by the algebra
    //   step = (lr * sqrt(bc2) / bc1) * m_new / (sqrt(v_new) + eps * sqrt(bc2))
    // and saves the two ``m_hat`` / ``v_hat`` materialisations (2 full-
    // tensor multiplies) per parameter.
    auto denom = ::mlx::core::add(::mlx::core::sqrt(v_new), *cache.eps_eff_a);
    auto step_arr =
        ::mlx::core::multiply(*cache.lr_eff_a, ::mlx::core::divide(m_new, denom));
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
        // 3.4 perf: refresh the scalar cache once per step.  All eight
        // broadcast scalars (beta1, omb1, beta2, omb2, eps, lr, inv_bc1,
        // inv_bc2) are constant across the parameter loop, so building
        // them ~65 times wastes ~3 ms; with the cache we build them
        // once and reuse for every parameter in the step.  Invalidates
        // when the param dtype changes (mixed-precision case).
        if (!scalar_cache_.valid || scalar_cache_.for_step != step_count_ ||
            scalar_cache_.dt != param->dtype()) {
            refresh_adam_scalar_cache(scalar_cache_, param->dtype(), lr_, beta1_, beta2_, eps_,
                                      weight_decay_, false, step_count_);
        }
        adam_step_gpu_cached(pg, gg, mg, vg, param->dtype(), weight_decay_, false, scalar_cache_);
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

std::vector<Optimizer::NamedBuffers> Adam::state_buffers() const {
    std::vector<NamedBuffers> out;
    if (m_.empty())
        return out;
    std::vector<std::shared_ptr<TensorImpl>> ms;
    std::vector<std::shared_ptr<TensorImpl>> vs;
    ms.reserve(params_.size());
    vs.reserve(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (i >= state_initialized_.size() || !state_initialized_[i] || !params_[i]) {
            ms.push_back(nullptr);
            vs.push_back(nullptr);
            continue;
        }
        const auto& p = params_[i];
        ms.push_back(clone_state_storage(m_[i], p->shape(), p->dtype(), p->device()));
        vs.push_back(clone_state_storage(v_[i], p->shape(), p->dtype(), p->device()));
    }
    out.emplace_back("exp_avg", std::move(ms));
    out.emplace_back("exp_avg_sq", std::move(vs));
    return out;
}

void Adam::load_state_buffers(const std::vector<NamedBuffers>& bufs) {
    if (m_.size() != params_.size())
        m_.resize(params_.size());
    if (v_.size() != params_.size())
        v_.resize(params_.size());
    if (state_initialized_.size() != params_.size())
        state_initialized_.assign(params_.size(), false);
    for (const auto& [name, tensors] : bufs) {
        std::vector<Storage>* dst = nullptr;
        if (name == "exp_avg")
            dst = &m_;
        else if (name == "exp_avg_sq")
            dst = &v_;
        else
            continue;
        for (std::size_t i = 0; i < tensors.size() && i < params_.size(); ++i) {
            if (!tensors[i] || !params_[i])
                continue;
            if (!state_initialized_[i]) {
                init_state_slot(i, params_[i]);
                state_initialized_[i] = true;
            }
            overwrite_state_storage((*dst)[i], tensors[i]->storage());
        }
    }
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
        // 3.4 perf: same scalar-cache optimization as Adam — see comment
        // there.  AdamW uses ``decoupled_wd = true``, which selects the
        // ``wd_factor`` (= 1 - lr*wd) branch of the cache.
        if (!scalar_cache_.valid || scalar_cache_.for_step != step_count_ ||
            scalar_cache_.dt != param->dtype()) {
            refresh_adam_scalar_cache(scalar_cache_, param->dtype(), lr_, beta1_, beta2_, eps_,
                                      weight_decay_, true, step_count_);
        }
        adam_step_gpu_cached(pg, gg, mg, vg, param->dtype(), weight_decay_, true, scalar_cache_);
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

std::vector<Optimizer::NamedBuffers> AdamW::state_buffers() const {
    std::vector<NamedBuffers> out;
    if (m_.empty())
        return out;
    std::vector<std::shared_ptr<TensorImpl>> ms;
    std::vector<std::shared_ptr<TensorImpl>> vs;
    ms.reserve(params_.size());
    vs.reserve(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (i >= state_initialized_.size() || !state_initialized_[i] || !params_[i]) {
            ms.push_back(nullptr);
            vs.push_back(nullptr);
            continue;
        }
        const auto& p = params_[i];
        ms.push_back(clone_state_storage(m_[i], p->shape(), p->dtype(), p->device()));
        vs.push_back(clone_state_storage(v_[i], p->shape(), p->dtype(), p->device()));
    }
    out.emplace_back("exp_avg", std::move(ms));
    out.emplace_back("exp_avg_sq", std::move(vs));
    return out;
}

void AdamW::load_state_buffers(const std::vector<NamedBuffers>& bufs) {
    if (m_.size() != params_.size())
        m_.resize(params_.size());
    if (v_.size() != params_.size())
        v_.resize(params_.size());
    if (state_initialized_.size() != params_.size())
        state_initialized_.assign(params_.size(), false);
    for (const auto& [name, tensors] : bufs) {
        std::vector<Storage>* dst = nullptr;
        if (name == "exp_avg")
            dst = &m_;
        else if (name == "exp_avg_sq")
            dst = &v_;
        else
            continue;
        for (std::size_t i = 0; i < tensors.size() && i < params_.size(); ++i) {
            if (!tensors[i] || !params_[i])
                continue;
            if (!state_initialized_[i]) {
                init_state_slot(i, params_[i]);
                state_initialized_[i] = true;
            }
            overwrite_state_storage((*dst)[i], tensors[i]->storage());
        }
    }
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
