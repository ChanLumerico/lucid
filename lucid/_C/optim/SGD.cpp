// lucid/_C/optim/SGD.cpp
//
// CPU and GPU implementations of Stochastic Gradient Descent (SGD)
// and Averaged SGD (ASGD). The CPU path is a typed scalar loop
// operating directly on raw buffer pointers; the GPU path builds an
// MLX expression graph that is evaluated lazily by the MLX runtime.

#include "SGD.h"

#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../autograd/Helpers.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "_OptimDetail.h"

using namespace lucid::optim_detail;

namespace lucid {

SGD::SGD(std::vector<std::shared_ptr<TensorImpl>> params,
         double lr,
         double momentum,
         double dampening,
         double weight_decay,
         bool nesterov)
    : Optimizer(std::move(params)),
      lr_(lr),
      momentum_(momentum),
      dampening_(dampening),
      weight_decay_(weight_decay),
      nesterov_(nesterov) {
    if (lr_ < 0.0)
        ErrorBuilder("SGD").fail("lr must be >= 0");
    if (momentum_ < 0.0)
        ErrorBuilder("SGD").fail("momentum must be >= 0");
    if (weight_decay_ < 0.0)
        ErrorBuilder("SGD").fail("weight_decay must be >= 0");
    // Nesterov momentum requires a pure momentum term (no dampening) so
    // that the gradient look-ahead is well-defined.
    if (nesterov_ && (momentum_ <= 0.0 || dampening_ != 0.0)) {
        ErrorBuilder("SGD").fail("nesterov requires momentum > 0 and dampening = 0");
    }
}

// Grow moment_ to cover all parameter slots if needed, then
// allocate a zero velocity buffer only when momentum is active.
void SGD::init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) {
    if (moment_.size() < params_.size()) {
        moment_.resize(params_.size());
    }
    if (momentum_ != 0.0) {
        moment_[slot_idx] = make_zero_storage(param->shape(), param->dtype(), param->device());
    }
}

namespace {

// Scalar CPU loop for SGD. Supports full feature set: weight decay,
// momentum, dampening, and Nesterov acceleration. When momentum == 0
// the moment_buf pointer is null and the code takes the simpler branch.
template <typename T>
void sgd_step_cpu(T* param,
                  const T* grad,
                  T* moment_buf,
                  std::size_t numel,
                  double lr,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  bool nesterov) {
    const T lrT = static_cast<T>(lr);
    const T mT = static_cast<T>(momentum);
    // (1 - dampening) is the scale applied to the gradient contribution
    // when updating the velocity buffer.
    const T dampT = static_cast<T>(1.0 - dampening);
    const T wdT = static_cast<T>(weight_decay);
    if (momentum != 0.0) {
        for (std::size_t i = 0; i < numel; ++i) {
            T g = grad[i];
            if (weight_decay != 0.0)
                g += wdT * param[i];
            T buf = mT * moment_buf[i] + dampT * g;
            moment_buf[i] = buf;
            // Nesterov: look one step ahead by adding m * new_buf to the
            // gradient; classical: use the buffer directly.
            const T eff_g = nesterov ? (g + mT * buf) : buf;
            param[i] -= lrT * eff_g;
        }
    } else {
        for (std::size_t i = 0; i < numel; ++i) {
            T g = grad[i];
            if (weight_decay != 0.0)
                g += wdT * param[i];
            param[i] -= lrT * g;
        }
    }
}

// MLX-based GPU path for SGD.
//
// All arithmetic is expressed as MLX lazy-evaluated array operations.
// The momentum buffer and parameter array are replaced atomically via
// gpu_replace() so the shared_ptr inside GpuStorage always points to
// the latest computed result.
void sgd_step_gpu(GpuStorage& param_g,
                  const GpuStorage& grad_g,
                  GpuStorage& moment_g,
                  Dtype dt,
                  double lr,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  bool nesterov) {
    if (!param_g.arr || !grad_g.arr) {
        ErrorBuilder("SGD GPU").fail("null array");
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
            ErrorBuilder("SGD GPU").fail("null momentum array");
        }
        ::mlx::core::array m_arr(momentum, mdt);
        ::mlx::core::array dampening_arr(1.0 - dampening, mdt);

        auto new_buf = ::mlx::core::add(::mlx::core::multiply(m_arr, *moment_g.arr),
                                        ::mlx::core::multiply(dampening_arr, g));

        moment_g.arr = gpu::wrap_mlx_array(::mlx::core::array(new_buf), dt).arr;

        ::mlx::core::array eff_g =
            nesterov ? ::mlx::core::add(g, ::mlx::core::multiply(m_arr, new_buf)) : new_buf;
        auto new_param = ::mlx::core::subtract(*param_g.arr, ::mlx::core::multiply(lr_arr, eff_g));
        param_g.arr = gpu::wrap_mlx_array(std::move(new_param), dt).arr;
    } else {
        auto new_param = ::mlx::core::subtract(*param_g.arr, ::mlx::core::multiply(lr_arr, g));
        param_g.arr = gpu::wrap_mlx_array(std::move(new_param), dt).arr;
    }
}

}  // namespace

// Dispatch SGD update to the GPU path or the typed CPU scalar loop.
// Only F32 and F64 are supported on CPU; other dtypes raise.
void SGD::update_one(std::size_t slot_idx,
                     std::shared_ptr<TensorImpl>& param,
                     const Storage& grad) {
    if (param->device() == Device::GPU) {
        auto& param_g = storage_gpu(param->mutable_storage());
        const auto& grad_g = storage_gpu(grad);
        GpuStorage* moment_g = nullptr;
        if (momentum_ != 0.0) {
            moment_g = &storage_gpu(moment_[slot_idx]);
        }

        GpuStorage dummy_moment;
        sgd_step_gpu(param_g, grad_g, moment_g ? *moment_g : dummy_moment, param->dtype(), lr_,
                     momentum_, dampening_, weight_decay_, nesterov_);
        param_g.bump_version();
        return;
    }

    auto& param_cpu = storage_cpu(param->mutable_storage());
    const auto& grad_cpu = storage_cpu(grad);
    CpuStorage* moment_cpu = nullptr;
    if (momentum_ != 0.0) {
        moment_cpu = &storage_cpu(moment_[slot_idx]);
    }
    const std::size_t numel = param_cpu.nbytes / dtype_size(param->dtype());

    switch (param->dtype()) {
    case Dtype::F32:
        sgd_step_cpu<float>(reinterpret_cast<float*>(param_cpu.ptr.get()),
                            reinterpret_cast<const float*>(grad_cpu.ptr.get()),
                            moment_cpu ? reinterpret_cast<float*>(moment_cpu->ptr.get()) : nullptr,
                            numel, lr_, momentum_, dampening_, weight_decay_, nesterov_);
        break;
    case Dtype::F64:
        sgd_step_cpu<double>(reinterpret_cast<double*>(param_cpu.ptr.get()),
                             reinterpret_cast<const double*>(grad_cpu.ptr.get()),
                             moment_cpu ? reinterpret_cast<double*>(moment_cpu->ptr.get())
                                        : nullptr,
                             numel, lr_, momentum_, dampening_, weight_decay_, nesterov_);
        break;
    default:
        ErrorBuilder("SGD").not_implemented("dtype not supported (F32/F64)");
    }
    param_cpu.bump_version();
}

ASGD::ASGD(std::vector<std::shared_ptr<TensorImpl>> p,
           double lr,
           double mom,
           double wd,
           double alpha,
           double t0,
           double lambd)
    : Optimizer(std::move(p)),
      lr_(lr),
      momentum_(mom),
      weight_decay_(wd),
      alpha_(alpha),
      t0_(t0),
      lambd_(lambd) {
    if (lr_ < 0.0)
        ErrorBuilder("ASGD").fail("lr must be >= 0");
}

// Initialize velocity and running-average buffers for this slot.
// The running average ax_ is seeded with a copy of the current
// parameter value so the average starts from a meaningful point.
void ASGD::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (moment_.size() < params_.size())
        moment_.resize(params_.size());
    if (ax_.size() < params_.size())
        ax_.resize(params_.size());
    if (step_.size() < params_.size())
        step_.resize(params_.size(), 0);
    if (momentum_ != 0.0) {
        moment_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    }

    ax_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    if (p->device() == Device::GPU) {
        const auto& gp = gpu_get(p->storage());
        gpu_replace(gpu_get(ax_[i]), ::mlx::core::copy(*gp.arr), p->dtype());
    } else {
        const auto& pc = storage_cpu(p->mutable_storage());
        auto& ac = storage_cpu(ax_[i]);
        std::memcpy(ac.ptr.get(), pc.ptr.get(), pc.nbytes);
    }
}

// Apply one ASGD step: standard SGD update followed by a running
// average update of ax_ once the step counter reaches t0_.
void ASGD::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    step_[i] += 1;
    const auto dt = p->dtype();
    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        if (momentum_ != 0.0) {
            auto& mg = gpu_get(moment_[i]);
            auto new_m =
                ::mlx::core::add(::mlx::core::multiply(mlx_scalar(momentum_, dt), *mg.arr), g);
            gpu_replace(mg, ::mlx::core::array(new_m), dt);
            g = new_m;
        }
        auto new_p = ::mlx::core::subtract(*pg.arr, ::mlx::core::multiply(mlx_scalar(lr_, dt), g));
        gpu_replace(pg, std::move(new_p), dt);

        if (step_[i] >= static_cast<std::int64_t>(t0_)) {
            // Exponentially decaying coefficient for the running average.
            const double coef = 1.0 / (alpha_ * step_[i] + 1.0);
            auto& ag = gpu_get(ax_[i]);

            auto new_ax = ::mlx::core::subtract(
                ::mlx::core::add(::mlx::core::multiply(mlx_scalar(1.0 - coef, dt), *ag.arr),
                                 ::mlx::core::multiply(mlx_scalar(coef, dt), *pg.arr)),
                ::mlx::core::multiply(mlx_scalar(lambd_, dt), *ag.arr));
            gpu_replace(ag, std::move(new_ax), dt);
        }
        gpu_get(p->mutable_storage()).bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
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
            if (weight_decay_ != 0.0)
                g += wdT * P[k];
            if (M) {
                M[k] = mT * M[k] + g;
                g = M[k];
            }
            P[k] -= lrT * g;
            if (do_avg) {
                A[k] = (T{1} - coefT) * A[k] + coefT * P[k] - lambdT * A[k];
            }
        }
    };
    if (dt == Dtype::F32)
        step_cpu(reinterpret_cast<float*>(p_cpu.ptr.get()), cpu_cptr<float>(grad));
    else if (dt == Dtype::F64)
        step_cpu(reinterpret_cast<double*>(p_cpu.ptr.get()), cpu_cptr<double>(grad));
    else
        ErrorBuilder("ASGD").not_implemented("dtype not supported");
    p_cpu.bump_version();
}

}  // namespace lucid
