#include "SGD.h"

#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../autograd/Helpers.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Exceptions.h"
#include "../core/TensorImpl.h"
#include "_OptimDetail.h"

using namespace lucid::optim_detail;

namespace lucid {

SGD::SGD(std::vector<std::shared_ptr<TensorImpl>> params,
         double lr, double momentum, double dampening, double weight_decay,
         bool nesterov)
    : Optimizer(std::move(params)),
      lr_(lr), momentum_(momentum), dampening_(dampening),
      weight_decay_(weight_decay), nesterov_(nesterov) {
    if (lr_ < 0.0) throw LucidError("SGD: lr must be >= 0");
    if (momentum_ < 0.0) throw LucidError("SGD: momentum must be >= 0");
    if (weight_decay_ < 0.0) throw LucidError("SGD: weight_decay must be >= 0");
    if (nesterov_ && (momentum_ <= 0.0 || dampening_ != 0.0)) {
        throw LucidError("SGD: nesterov requires momentum > 0 and dampening = 0");
    }
}

void SGD::init_state_slot(std::size_t slot_idx,
                          const std::shared_ptr<TensorImpl>& param) {
    if (moment_.size() < params_.size()) {
        moment_.resize(params_.size());
    }
    if (momentum_ != 0.0) {
        moment_[slot_idx] = make_zero_storage(param->shape_, param->dtype_,
                                              param->device_);
    }
}

namespace {

// CPU SGD step (single buffer of length numel). All math in T.
template <typename T>
void sgd_step_cpu(T* param, const T* grad,
                  T* moment_buf,
                  std::size_t numel,
                  double lr, double momentum, double dampening,
                  double weight_decay, bool nesterov) {
    const T lrT = static_cast<T>(lr);
    const T mT = static_cast<T>(momentum);
    const T dampT = static_cast<T>(1.0 - dampening);
    const T wdT = static_cast<T>(weight_decay);
    if (momentum != 0.0) {
        for (std::size_t i = 0; i < numel; ++i) {
            T g = grad[i];
            if (weight_decay != 0.0) g += wdT * param[i];
            T buf = mT * moment_buf[i] + dampT * g;
            moment_buf[i] = buf;
            const T eff_g = nesterov ? (g + mT * buf) : buf;
            param[i] -= lrT * eff_g;
        }
    } else {
        for (std::size_t i = 0; i < numel; ++i) {
            T g = grad[i];
            if (weight_decay != 0.0) g += wdT * param[i];
            param[i] -= lrT * g;
        }
    }
}

// GPU SGD step using MLX functional ops. Replaces param.storage_ and
// moment.arr with new arrays each call (functional MLX semantics).
void sgd_step_gpu(GpuStorage& param_g, const GpuStorage& grad_g,
                  GpuStorage& moment_g,
                  Dtype dt,
                  double lr, double momentum, double dampening,
                  double weight_decay, bool nesterov) {
    if (!param_g.arr || !grad_g.arr) {
        throw LucidError("SGD GPU: null array");
    }
    const auto mdt = gpu::to_mlx_dtype(dt);
    auto g = *grad_g.arr;
    if (weight_decay != 0.0) {
        ::mlx::core::array wd_arr(weight_decay, mdt);
        g = ::mlx::core::add(g, ::mlx::core::multiply(wd_arr, *param_g.arr));
    }
    ::mlx::core::array lr_arr(lr, mdt);
    if (momentum != 0.0) {
        if (!moment_g.arr) {
            throw LucidError("SGD GPU: null momentum array");
        }
        ::mlx::core::array m_arr(momentum, mdt);
        ::mlx::core::array dampening_arr(1.0 - dampening, mdt);
        // buf ← momentum · buf + (1-damp) · g
        auto new_buf = ::mlx::core::add(
            ::mlx::core::multiply(m_arr, *moment_g.arr),
            ::mlx::core::multiply(dampening_arr, g));
        // Swap moment storage with new buffer.
        moment_g.arr =
            gpu::wrap_mlx_array(::mlx::core::array(new_buf), dt).arr;

        ::mlx::core::array eff_g = nesterov
            ? ::mlx::core::add(g, ::mlx::core::multiply(m_arr, new_buf))
            : new_buf;
        auto new_param =
            ::mlx::core::subtract(*param_g.arr,
                                  ::mlx::core::multiply(lr_arr, eff_g));
        param_g.arr =
            gpu::wrap_mlx_array(std::move(new_param), dt).arr;
    } else {
        auto new_param =
            ::mlx::core::subtract(*param_g.arr,
                                  ::mlx::core::multiply(lr_arr, g));
        param_g.arr =
            gpu::wrap_mlx_array(std::move(new_param), dt).arr;
    }
}

}  // namespace

void SGD::update_one(std::size_t slot_idx,
                     std::shared_ptr<TensorImpl>& param,
                     const Storage& grad) {
    if (param->device_ == Device::GPU) {
        auto& param_g = std::get<GpuStorage>(param->storage_);
        const auto& grad_g = std::get<GpuStorage>(grad);
        GpuStorage* moment_g = nullptr;
        if (momentum_ != 0.0) {
            moment_g = &std::get<GpuStorage>(moment_[slot_idx]);
        }
        // The function expects a non-null reference even when momentum=0;
        // gate at the kernel level instead.
        GpuStorage dummy_moment;
        sgd_step_gpu(param_g, grad_g,
                     moment_g ? *moment_g : dummy_moment,
                     param->dtype_,
                     lr_, momentum_, dampening_, weight_decay_, nesterov_);
        return;
    }

    auto& param_cpu = std::get<CpuStorage>(param->storage_);
    const auto& grad_cpu = std::get<CpuStorage>(grad);
    CpuStorage* moment_cpu = nullptr;
    if (momentum_ != 0.0) {
        moment_cpu = &std::get<CpuStorage>(moment_[slot_idx]);
    }
    const std::size_t numel = param_cpu.nbytes / dtype_size(param->dtype_);

    switch (param->dtype_) {
        case Dtype::F32:
            sgd_step_cpu<float>(
                reinterpret_cast<float*>(param_cpu.ptr.get()),
                reinterpret_cast<const float*>(grad_cpu.ptr.get()),
                moment_cpu ? reinterpret_cast<float*>(moment_cpu->ptr.get()) : nullptr,
                numel, lr_, momentum_, dampening_, weight_decay_, nesterov_);
            break;
        case Dtype::F64:
            sgd_step_cpu<double>(
                reinterpret_cast<double*>(param_cpu.ptr.get()),
                reinterpret_cast<const double*>(grad_cpu.ptr.get()),
                moment_cpu ? reinterpret_cast<double*>(moment_cpu->ptr.get()) : nullptr,
                numel, lr_, momentum_, dampening_, weight_decay_, nesterov_);
            break;
        default:
            throw NotImplementedError(
                "SGD: dtype not supported (F32/F64)");
    }
}

// =====================================================================
// ASGD
// =====================================================================

ASGD::ASGD(std::vector<std::shared_ptr<TensorImpl>> p,
           double lr, double mom, double wd, double alpha, double t0,
           double lambd)
    : Optimizer(std::move(p)),
      lr_(lr), momentum_(mom), weight_decay_(wd),
      alpha_(alpha), t0_(t0), lambd_(lambd) {
    if (lr_ < 0.0) throw LucidError("ASGD: lr must be >= 0");
}

void ASGD::init_state_slot(std::size_t i,
                           const std::shared_ptr<TensorImpl>& p) {
    if (moment_.size() < params_.size()) moment_.resize(params_.size());
    if (ax_.size()     < params_.size()) ax_.resize(params_.size());
    if (step_.size()   < params_.size()) step_.resize(params_.size(), 0);
    if (momentum_ != 0.0) {
        moment_[i] = make_zero_storage(p->shape_, p->dtype_, p->device_);
    }
    // ax_ initialized to a clone of param.
    ax_[i] = make_zero_storage(p->shape_, p->dtype_, p->device_);
    if (p->device_ == Device::GPU) {
        const auto& gp = gpu_get(p->storage_);
        gpu_replace(gpu_get(ax_[i]), ::mlx::core::copy(*gp.arr), p->dtype_);
    } else {
        const auto& pc = std::get<CpuStorage>(p->storage_);
        auto& ac = std::get<CpuStorage>(ax_[i]);
        std::memcpy(ac.ptr.get(), pc.ptr.get(), pc.nbytes);
    }
}

void ASGD::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p,
                      const Storage& grad) {
    step_[i] += 1;
    const auto dt = p->dtype_;
    if (p->device_ == Device::GPU) {
        auto& pg = gpu_get(p->storage_);
        const auto& gg = gpu_get(grad);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(
                mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        if (momentum_ != 0.0) {
            auto& mg = gpu_get(moment_[i]);
            auto new_m = ::mlx::core::add(
                ::mlx::core::multiply(mlx_scalar(momentum_, dt), *mg.arr), g);
            gpu_replace(mg, ::mlx::core::array(new_m), dt);
            g = new_m;
        }
        auto new_p = ::mlx::core::subtract(
            *pg.arr, ::mlx::core::multiply(mlx_scalar(lr_, dt), g));
        gpu_replace(pg, std::move(new_p), dt);

        if (step_[i] >= static_cast<std::int64_t>(t0_)) {
            const double coef = 1.0 / (alpha_ * step_[i] + 1.0);
            auto& ag = gpu_get(ax_[i]);
            // ax = (1 − coef)·ax + coef·param − lambd·ax
            auto new_ax = ::mlx::core::subtract(
                ::mlx::core::add(
                    ::mlx::core::multiply(mlx_scalar(1.0 - coef, dt), *ag.arr),
                    ::mlx::core::multiply(mlx_scalar(coef, dt), *pg.arr)),
                ::mlx::core::multiply(mlx_scalar(lambd_, dt), *ag.arr));
            gpu_replace(ag, std::move(new_ax), dt);
        }
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* M = (momentum_ != 0.0) ? cpu_ptr<T>(moment_[i]) : nullptr;
        T* A = cpu_ptr<T>(ax_[i]);
        const T lrT = static_cast<T>(lr_);
        const T mT = static_cast<T>(momentum_);
        const T wdT = static_cast<T>(weight_decay_);
        const T coefT = static_cast<T>(1.0 / (alpha_ * step_[i] + 1.0));
        const T lambdT = static_cast<T>(lambd_);
        const bool do_avg = step_[i] >= static_cast<std::int64_t>(t0_);
        for (std::size_t k = 0; k < n; ++k) {
            T g = G[k];
            if (weight_decay_ != 0.0) g += wdT * P[k];
            if (M) { M[k] = mT * M[k] + g; g = M[k]; }
            P[k] -= lrT * g;
            if (do_avg) {
                A[k] = (T{1} - coefT) * A[k] + coefT * P[k] - lambdT * A[k];
            }
        }
    };
    if (dt == Dtype::F32) step_cpu(cpu_ptr<float>(p->storage_),
                                    cpu_cptr<float>(grad));
    else if (dt == Dtype::F64) step_cpu(cpu_ptr<double>(p->storage_),
                                          cpu_cptr<double>(grad));
    else throw NotImplementedError("ASGD: dtype not supported");
}

}  // namespace lucid
