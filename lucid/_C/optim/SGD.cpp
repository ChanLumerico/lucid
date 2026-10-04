// lucid/_C/optim/SGD.cpp
//
// CPU and GPU implementations of Stochastic Gradient Descent (SGD)
// and Averaged SGD (ASGD). The CPU path is a typed scalar loop
// operating directly on raw buffer pointers; the GPU path builds an
// MLX expression graph that is evaluated lazily by the MLX runtime.

#include "SGD.h"

#include <algorithm>
#include <cmath>
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
        ErrorBuilder("SGD").invalid_argument("lr must be >= 0");
    if (momentum_ < 0.0)
        ErrorBuilder("SGD").invalid_argument("momentum must be >= 0");
    if (weight_decay_ < 0.0)
        ErrorBuilder("SGD").invalid_argument("weight_decay must be >= 0");
    // Nesterov momentum requires a pure momentum term (no dampening) so
    // that the gradient look-ahead is well-defined.
    if (nesterov_ && (momentum_ <= 0.0 || dampening_ != 0.0)) {
        ErrorBuilder("SGD").invalid_argument("nesterov requires momentum > 0 and dampening = 0");
    }
}

// Grow moment_ to cover all parameter slots.  The buffer itself is made by
// the slot's first momentum step.
void SGD::init_state_slot(std::size_t slot_idx, const std::shared_ptr<TensorImpl>& param) {
    (void)slot_idx;
    (void)param;
    if (moment_.size() < params_.size())
        moment_.resize(params_.size());
}

namespace {

// Scalar CPU loop for SGD. Supports full feature set: weight decay,
// momentum, dampening, and Nesterov acceleration. When momentum == 0
// the moment_buf pointer is null and the code takes the simpler branch.
// ``first_momentum_step`` starts the buffer at the gradient, undamped —
// the reference framework's ``buf = clone(grad)`` for a parameter that has
// no buffer yet.
template <typename T>
void sgd_step_cpu(T* param,
                  const T* grad,
                  T* moment_buf,
                  bool first_momentum_step,
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
            T buf = first_momentum_step ? g : mT * moment_buf[i] + dampT * g;
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
// the latest computed result.  ``moment`` is null without momentum; a
// slot that holds no buffer yet gets one here, started at the gradient
// (see ``sgd_step_cpu``).
void sgd_step_gpu(GpuStorage& param_g,
                  const GpuStorage& grad_g,
                  Storage* moment,
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
    if (moment != nullptr) {
        ::mlx::core::array m_arr(momentum, mdt);
        ::mlx::core::array dampening_arr(1.0 - dampening, mdt);

        // MLX arrays are immutable, so the buffer may share the gradient's.
        const bool first_momentum_step = !holds_buffer(*moment);
        auto new_buf = first_momentum_step
                           ? g
                           : ::mlx::core::add(::mlx::core::multiply(m_arr, *gpu_get(*moment).arr),
                                              ::mlx::core::multiply(dampening_arr, g));
        if (first_momentum_step)
            *moment = gpu::wrap_mlx_array(::mlx::core::array(new_buf), dt);
        else
            gpu_get(*moment).arr = gpu::wrap_mlx_array(::mlx::core::array(new_buf), dt).arr;

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
    const bool use_momentum = momentum_ != 0.0;
    if (param->device() == Device::GPU) {
        auto& param_g = storage_gpu(param->mutable_storage());
        const auto& grad_g = storage_gpu(grad);
        sgd_step_gpu(param_g, grad_g, use_momentum ? &moment_[slot_idx] : nullptr, param->dtype(),
                     lr_, momentum_, dampening_, weight_decay_, nesterov_);
        param_g.bump_version();
        return;
    }

    auto& param_cpu = storage_cpu(param->mutable_storage());
    const auto& grad_cpu = storage_cpu(grad);
    CpuStorage* moment_cpu = nullptr;
    // A slot without a buffer is at its first momentum step: the loop
    // writes the gradient into a fresh one.
    const bool first_momentum_step = use_momentum && !holds_buffer(moment_[slot_idx]);
    if (first_momentum_step)
        moment_[slot_idx] = make_zero_storage(param->shape(), param->dtype(), param->device());
    if (use_momentum) {
        moment_cpu = &storage_cpu(moment_[slot_idx]);
    }
    const std::size_t numel = param_cpu.nbytes / dtype_size(param->dtype());

    switch (param->dtype()) {
    case Dtype::F32:
        sgd_step_cpu<float>(reinterpret_cast<float*>(param_cpu.ptr.get()),
                            reinterpret_cast<const float*>(grad_cpu.ptr.get()),
                            moment_cpu ? reinterpret_cast<float*>(moment_cpu->ptr.get()) : nullptr,
                            first_momentum_step, numel, lr_, momentum_, dampening_, weight_decay_,
                            nesterov_);
        break;
    case Dtype::F64:
        sgd_step_cpu<double>(
            reinterpret_cast<double*>(param_cpu.ptr.get()),
            reinterpret_cast<const double*>(grad_cpu.ptr.get()),
            moment_cpu ? reinterpret_cast<double*>(moment_cpu->ptr.get()) : nullptr,
            first_momentum_step, numel, lr_, momentum_, dampening_, weight_decay_, nesterov_);
        break;
    default:
        ErrorBuilder("SGD").not_implemented("dtype not supported (F32/F64)");
    }
    param_cpu.bump_version();
}

// Every buffer a slot holds is exported, including one kept after momentum
// was set to zero; slots without one contribute null.
std::vector<Optimizer::NamedBuffers> SGD::state_buffers() const {
    std::vector<std::shared_ptr<TensorImpl>> mom(params_.size());
    bool any = false;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (!slot_has_state(i) || i >= moment_.size() || !holds_buffer(moment_[i]))
            continue;
        const auto& p = params_[i];
        mom[i] = clone_state_storage(moment_[i], p->shape(), p->dtype(), p->device());
        any = true;
    }
    if (!any)
        return {};
    std::vector<NamedBuffers> out;
    out.emplace_back("momentum_buffer", std::move(mom));
    return out;
}

void SGD::load_state_buffers(const std::vector<NamedBuffers>& bufs) {
    for (const auto& [name, tensors] : bufs) {
        if (name == "momentum_buffer")
            load_state_slots(moment_, tensors);
    }
}

ASGD::ASGD(std::vector<std::shared_ptr<TensorImpl>> p,
           double lr,
           double lambd,
           double alpha,
           double t0,
           double wd)
    : Optimizer(std::move(p)), lr_(lr), lambd_(lambd), alpha_(alpha), t0_(t0), weight_decay_(wd) {
    if (lr_ < 0.0)
        ErrorBuilder("ASGD").invalid_argument("lr must be >= 0");
}

// ax starts at zero — the first update has mu = 1 and copies the
// parameter in — eta at the current learning rate, mu at one.
void ASGD::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (ax_.size() < params_.size())
        ax_.resize(params_.size());
    if (eta_.size() < params_.size())
        eta_.resize(params_.size(), 0.0);
    if (mu_.size() < params_.size())
        mu_.resize(params_.size(), 1.0);
    ax_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    eta_[i] = round_to_state_scalar(lr_, p->dtype());
    mu_[i] = 1.0;
}

// One ASGD step with the slot's current eta and mu:
//   p  = p * (1 - lambd * eta) - eta * g      (g carries the weight decay)
//   ax = p                     when mu == 1   (before averaging starts)
//   ax = ax + (p - ax) * mu    otherwise
// then eta and mu advance for the slot's next step.  The decay and the
// step are two roundings, as the reference framework's two in-place ops
// are, and the averaging is too.
void ASGD::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    const auto dt = p->dtype();
    const double eta = eta_[i];
    const double mu = mu_[i];
    // mu is exactly 1 until averaging starts, and the average is then the
    // parameter itself.
    const bool track = (mu == 1.0);
    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& ag = gpu_get(ax_[i]);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        auto decayed = ::mlx::core::multiply(*pg.arr, mlx_scalar(1.0 - lambd_ * eta, dt));
        auto new_p = ::mlx::core::add(decayed, ::mlx::core::multiply(mlx_scalar(-eta, dt), g));
        ::mlx::core::array new_ax =
            track ? ::mlx::core::copy(new_p)
                  : ::mlx::core::add(*ag.arr,
                                     ::mlx::core::multiply(::mlx::core::subtract(new_p, *ag.arr),
                                                           mlx_scalar(mu, dt)));
        gpu_replace(ag, std::move(new_ax), dt);
        gpu_replace(pg, std::move(new_p), dt);
        pg.bump_version();
    } else {
        const std::size_t n = cpu_numel(*p);
        auto& p_cpu = storage_cpu(p->mutable_storage());
        auto step_cpu = [&](auto* P, const auto* G) {
            using T = std::remove_pointer_t<decltype(P)>;
            T* A = cpu_ptr<T>(ax_[i]);
            const T wdT = static_cast<T>(weight_decay_);
            const T decay = static_cast<T>(1.0 - lambd_ * eta);
            const T neg_eta = static_cast<T>(-eta);
            const T muT = static_cast<T>(mu);
            for (std::size_t k = 0; k < n; ++k) {
                T g = G[k];
                if (weight_decay_ != 0.0)
                    g += wdT * P[k];
                // Separate statements keep the decay and the step from
                // being contracted into one rounding.
                const T decayed = P[k] * decay;
                P[k] = decayed + neg_eta * g;
                if (track) {
                    A[k] = P[k];
                } else {
                    const T pull = (P[k] - A[k]) * muT;
                    A[k] += pull;
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

    // Advance the schedule with the slot's own step count and the current
    // learning rate.  Both are held at their checkpoint precision.
    const double t = static_cast<double>(steps_[i]);
    eta_[i] = round_to_state_scalar(lr_ / std::pow(1.0 + lambd_ * lr_ * t, alpha_), dt);
    mu_[i] = round_to_state_scalar(1.0 / std::max(1.0, t - t0_), dt);
}

std::vector<Optimizer::NamedBuffers> ASGD::state_buffers() const {
    std::vector<std::shared_ptr<TensorImpl>> eta(params_.size());
    std::vector<std::shared_ptr<TensorImpl>> mu(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (!slot_has_state(i) || i >= eta_.size() || i >= mu_.size())
            continue;
        const Dtype sdt = state_scalar_dtype(params_[i]->dtype());
        eta[i] = make_state_scalar(eta_[i], sdt);
        mu[i] = make_state_scalar(mu_[i], sdt);
    }
    std::vector<NamedBuffers> out;
    out.emplace_back("step", clone_step_slots());
    out.emplace_back("eta", std::move(eta));
    out.emplace_back("mu", std::move(mu));
    out.emplace_back("ax", clone_state_slots(ax_));
    return out;
}

void ASGD::load_state_buffers(const std::vector<NamedBuffers>& bufs) {
    // eta and mu are one number per parameter, read wherever they live.
    auto load_scalars = [this](std::vector<double>& dst,
                               const std::vector<std::shared_ptr<TensorImpl>>& saved) {
        for (std::size_t i = 0; i < saved.size() && i < params_.size(); ++i) {
            if (!saved[i] || !params_[i])
                continue;
            ensure_state_slot(i);
            dst[i] = round_to_state_scalar(read_state_scalar(*saved[i]), params_[i]->dtype());
        }
    };
    for (const auto& [name, tensors] : bufs) {
        if (name == "step")
            load_step_slots(tensors);
        else if (name == "eta")
            load_scalars(eta_, tensors);
        else if (name == "mu")
            load_scalars(mu_, tensors);
        else if (name == "ax")
            load_state_slots(ax_, tensors);
    }
}

}  // namespace lucid
