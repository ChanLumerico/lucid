// lucid/_C/optim/Prop.cpp
//
// CPU and GPU implementations of RMSprop and Rprop. RMSprop uses MLX
// array operations on GPU and a scalar loop on CPU. Rprop on GPU uses
// element-wise masking via mlx::core::where to avoid conditional branches
// across array elements.

#include "Prop.h"

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

// Move ``from`` toward ``to`` by weight ``w`` the way the reference
// framework's lerp does: ``from + w * (to - from)`` while |w| < 0.5,
// else ``to - (to - from) * (1 - w)``.  The two forms round differently,
// and the reference picks by the weight, so matching it bit for bit
// means picking the same way.
template <typename T>
inline T lerp_toward(T from, T to, T w, bool small_w) {
    return small_w ? from + w * (to - from) : to - (to - from) * (T{1} - w);
}

// MLX counterpart of the scalar ``lerp_toward`` above.
inline ::mlx::core::array
lerp_toward(const ::mlx::core::array& from, const ::mlx::core::array& to, double w, Dtype dt) {
    auto diff = ::mlx::core::subtract(to, from);
    if (std::abs(w) < 0.5)
        return ::mlx::core::add(from, ::mlx::core::multiply(mlx_scalar(w, dt), diff));
    return ::mlx::core::subtract(to, ::mlx::core::multiply(diff, mlx_scalar(1.0 - w, dt)));
}

}  // namespace

void RMSprop::check_hyperparams(
    double lr, double alpha, double eps, double weight_decay, double momentum) {
    require(lr >= 0.0, "RMSprop", "lr must be >= 0");
    require(eps >= 0.0, "RMSprop", "eps must be >= 0");
    require(momentum >= 0.0, "RMSprop", "momentum must be >= 0");
    require(weight_decay >= 0.0, "RMSprop", "weight_decay must be >= 0");
    require(alpha >= 0.0, "RMSprop", "alpha must be >= 0");
}

RMSprop::RMSprop(std::vector<std::shared_ptr<TensorImpl>> p,
                 double lr,
                 double alpha,
                 double eps,
                 double wd,
                 double momentum,
                 bool centered)
    : Optimizer(std::move(p)),
      lr_(lr),
      alpha_(alpha),
      eps_(eps),
      weight_decay_(wd),
      momentum_(momentum),
      centered_(centered) {
    check_hyperparams(lr_, alpha_, eps_, weight_decay_, momentum_);
}

void RMSprop::set_hyperparams(
    double lr, double alpha, double eps, double weight_decay, double momentum, bool centered) {
    check_hyperparams(lr, alpha, eps, weight_decay, momentum);
    if (centered && !centered_)
        ensure_buffers(grad_avg_);
    if (momentum != 0.0 && momentum_ == 0.0)
        ensure_buffers(moment_buf_);
    lr_ = lr;
    alpha_ = alpha;
    eps_ = eps;
    weight_decay_ = weight_decay;
    momentum_ = momentum;
    centered_ = centered;
}

// Conditionally allocate the three possible state buffers.
// grad_avg_ and moment_buf_ are only allocated when their corresponding
// features are enabled, avoiding unnecessary memory allocation.
void RMSprop::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (square_avg_.size() < params_.size())
        square_avg_.resize(params_.size());
    if (grad_avg_.size() < params_.size())
        grad_avg_.resize(params_.size());
    if (moment_buf_.size() < params_.size())
        moment_buf_.resize(params_.size());
    square_avg_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    if (centered_)
        grad_avg_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    if (momentum_ != 0.0)
        moment_buf_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
}

// Apply one RMSprop step. On CPU the lambda captures typed pointers
// and runs a scalar loop. On GPU each operation produces a new MLX
// array that replaces the previous one via gpu_replace().
void RMSprop::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    const auto dt = p->dtype();
    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& sq = gpu_get(square_avg_[i]);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        auto new_sq = ::mlx::core::add(
            ::mlx::core::multiply(mlx_scalar(alpha_, dt), *sq.arr),
            ::mlx::core::multiply(mlx_scalar(1.0 - alpha_, dt), ::mlx::core::square(g)));
        gpu_replace(sq, ::mlx::core::array(new_sq), dt);
        ::mlx::core::array avg = new_sq;
        if (centered_) {
            // Centered RMSprop: subtract the squared gradient mean to
            // estimate the variance rather than the raw second moment.
            // The mean advances as a lerp toward g by 1 - alpha.
            auto& ga = gpu_get(grad_avg_[i]);
            auto new_ga = lerp_toward(*ga.arr, g, 1.0 - alpha_, dt);
            gpu_replace(ga, ::mlx::core::array(new_ga), dt);
            avg = ::mlx::core::subtract(new_sq, ::mlx::core::square(new_ga));
        }

        auto denom = ::mlx::core::add(::mlx::core::sqrt(avg), mlx_scalar(eps_, dt));
        // Without momentum the step is (lr * g) / denom, in that order.
        ::mlx::core::array step_dir = g;
        if (momentum_ != 0.0) {
            auto& mb = gpu_get(moment_buf_[i]);
            auto new_mb =
                ::mlx::core::add(::mlx::core::multiply(mlx_scalar(momentum_, dt), *mb.arr),
                                 ::mlx::core::divide(g, denom));
            gpu_replace(mb, ::mlx::core::array(new_mb), dt);
            step_dir = ::mlx::core::multiply(mlx_scalar(lr_, dt), new_mb);
        } else {
            step_dir = ::mlx::core::divide(::mlx::core::multiply(mlx_scalar(lr_, dt), g), denom);
        }
        auto new_p = ::mlx::core::subtract(*pg.arr, step_dir);
        gpu_replace(pg, std::move(new_p), dt);
        pg.bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* SQ = cpu_ptr<T>(square_avg_[i]);
        // GA and MB are null when the corresponding features are disabled.
        T* GA = centered_ ? cpu_ptr<T>(grad_avg_[i]) : nullptr;
        T* MB = (momentum_ != 0.0) ? cpu_ptr<T>(moment_buf_[i]) : nullptr;
        const T lrT = static_cast<T>(lr_);
        const T aT = static_cast<T>(alpha_);
        const T omaT = static_cast<T>(1.0 - alpha_);
        const T epsT = static_cast<T>(eps_);
        const T wdT = static_cast<T>(weight_decay_);
        const T mT = static_cast<T>(momentum_);
        const bool small_w = std::abs(omaT) < T{0.5};
        for (std::size_t k = 0; k < n; ++k) {
            T g = G[k];
            if (weight_decay_ != 0.0)
                g += wdT * P[k];
            SQ[k] = aT * SQ[k] + omaT * g * g;
            T avg = SQ[k];
            if (GA) {
                GA[k] = lerp_toward(GA[k], g, omaT, small_w);
                avg = SQ[k] - GA[k] * GA[k];
            }

            const T denom = std::sqrt(avg) + epsT;
            if (MB) {
                MB[k] = mT * MB[k] + g / denom;
                P[k] -= lrT * MB[k];
            } else {
                P[k] -= lrT * g / denom;
            }
        }
    };
    if (dt == Dtype::F32)
        step_cpu(reinterpret_cast<float*>(p_cpu.ptr.get()), cpu_cptr<float>(grad));
    else if (dt == Dtype::F64)
        step_cpu(reinterpret_cast<double*>(p_cpu.ptr.get()), cpu_cptr<double>(grad));
    else
        ErrorBuilder("RMSprop").not_implemented("dtype not supported");
    p_cpu.bump_version();
}

std::vector<Optimizer::NamedBuffers> RMSprop::state_buffers() const {
    std::vector<NamedBuffers> out;
    out.emplace_back("step", clone_step_slots());
    out.emplace_back("square_avg", clone_state_slots(square_avg_));
    if (momentum_ != 0.0)
        out.emplace_back("momentum_buffer", clone_state_slots(moment_buf_));
    if (centered_)
        out.emplace_back("grad_avg", clone_state_slots(grad_avg_));
    return out;
}

void RMSprop::load_state_buffers(const std::vector<NamedBuffers>& bufs) {
    for (const auto& [name, tensors] : bufs) {
        if (name == "step")
            load_step_slots(tensors);
        else if (name == "square_avg")
            load_state_slots(square_avg_, tensors);
        else if (name == "momentum_buffer" && momentum_ != 0.0)
            load_state_slots(moment_buf_, tensors);
        else if (name == "grad_avg" && centered_)
            load_state_slots(grad_avg_, tensors);
    }
}

Rprop::Rprop(std::vector<std::shared_ptr<TensorImpl>> p,
             double lr,
             double eta_minus,
             double eta_plus,
             double step_min,
             double step_max)
    : Optimizer(std::move(p)),
      lr_(lr),
      eta_minus_(eta_minus),
      eta_plus_(eta_plus),
      step_min_(step_min),
      step_max_(step_max) {
    check_hyperparams(lr_, eta_minus_, eta_plus_);
}

void Rprop::check_hyperparams(double lr, double eta_minus, double eta_plus) {
    require(lr >= 0.0, "Rprop", "lr must be >= 0");
    require(eta_minus > 0.0 && eta_minus < 1.0 && eta_plus > 1.0, "Rprop",
            "etas must satisfy 0 < eta_minus < 1 < eta_plus");
}

void Rprop::set_hyperparams(
    double lr, double eta_minus, double eta_plus, double step_min, double step_max) {
    check_hyperparams(lr, eta_minus, eta_plus);
    lr_ = lr;
    eta_minus_ = eta_minus;
    eta_plus_ = eta_plus;
    step_min_ = step_min;
    step_max_ = step_max;
}

// Allocate previous-gradient buffer (zero) and step-size buffer.
// step_size_ is set to lr_ on every element rather than 1.0 so the
// very first update uses a sensible absolute step magnitude.
void Rprop::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (prev_grad_.size() < params_.size())
        prev_grad_.resize(params_.size());
    if (step_size_.size() < params_.size())
        step_size_.resize(params_.size());
    prev_grad_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());

    step_size_[i] = make_ones_storage(p->shape(), p->dtype(), p->device());
    if (p->device() == Device::GPU) {
        auto& s = gpu_get(step_size_[i]);
        auto scaled = ::mlx::core::multiply(mlx_scalar(lr_, p->dtype()), *s.arr);
        gpu_replace(s, std::move(scaled), p->dtype());
    } else {
        const std::size_t n = cpu_numel(*p);
        if (p->dtype() == Dtype::F32) {
            auto* q = cpu_ptr<float>(step_size_[i]);
            const float lrf = static_cast<float>(lr_);
            for (std::size_t k = 0; k < n; ++k)
                q[k] = lrf;
        } else if (p->dtype() == Dtype::F64) {
            auto* q = cpu_ptr<double>(step_size_[i]);
            for (std::size_t k = 0; k < n; ++k)
                q[k] = lr_;
        }
    }
}

// Apply one Rprop step. On GPU sign agreement is detected with
// element-wise greater/less comparisons, and where() selects between
// the scaled and unscaled step for each element without branching.
// When the gradient reverses sign, the previous gradient is zeroed so
// the next step does not trigger another sign-change event.
void Rprop::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    const auto dt = p->dtype();
    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& pv = gpu_get(prev_grad_[i]);
        auto& ss = gpu_get(step_size_[i]);

        ::mlx::core::array sign_change = ::mlx::core::multiply(*gg.arr, *pv.arr);
        ::mlx::core::array zero_arr = mlx_scalar(0.0, dt);

        // Increase step size where gradient maintained its sign.
        auto pos_mask = ::mlx::core::greater(sign_change, zero_arr);
        auto inc = ::mlx::core::multiply(mlx_scalar(eta_plus_, dt), *ss.arr);
        auto new_ss = ::mlx::core::where(pos_mask, inc, *ss.arr);

        // Decrease step size where gradient reversed sign.
        auto neg_mask = ::mlx::core::less(sign_change, zero_arr);
        auto dec = ::mlx::core::multiply(mlx_scalar(eta_minus_, dt), new_ss);
        new_ss = ::mlx::core::where(neg_mask, dec, new_ss);
        new_ss = ::mlx::core::clip(new_ss, mlx_scalar(step_min_, dt), mlx_scalar(step_max_, dt));
        gpu_replace(ss, ::mlx::core::array(new_ss), dt);

        // Zero out the gradient for sign-reversal elements so the next
        // step does not see a spurious sign agreement.
        auto eff_g = ::mlx::core::where(neg_mask, zero_arr, *gg.arr);
        gpu_replace(pv, ::mlx::core::array(eff_g), dt);
        auto new_p =
            ::mlx::core::subtract(*pg.arr, ::mlx::core::multiply(::mlx::core::sign(eff_g), new_ss));
        gpu_replace(pg, std::move(new_p), dt);
        pg.bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* PV = cpu_ptr<T>(prev_grad_[i]);
        T* SS = cpu_ptr<T>(step_size_[i]);
        const T epT = static_cast<T>(eta_plus_);
        const T emT = static_cast<T>(eta_minus_);
        const T smin = static_cast<T>(step_min_);
        const T smax = static_cast<T>(step_max_);
        for (std::size_t k = 0; k < n; ++k) {
            const T sc = G[k] * PV[k];  // Positive: same sign; negative: reversed.
            T s = SS[k];
            if (sc > T{0})
                s *= epT;
            else if (sc < T{0})
                s *= emT;
            if (s < smin)
                s = smin;
            if (s > smax)
                s = smax;
            SS[k] = s;
            // Zero the gradient for sign-reversal elements; this also
            // makes PV[k] = 0 so the next step sees no sign agreement.
            T g = (sc < T{0}) ? T{0} : G[k];
            PV[k] = g;
            const T sgn = (g > T{0}) ? T{1} : ((g < T{0}) ? T{-1} : T{0});
            P[k] -= sgn * s;
        }
    };
    if (dt == Dtype::F32)
        step_cpu(reinterpret_cast<float*>(p_cpu.ptr.get()), cpu_cptr<float>(grad));
    else if (dt == Dtype::F64)
        step_cpu(reinterpret_cast<double*>(p_cpu.ptr.get()), cpu_cptr<double>(grad));
    else
        ErrorBuilder("Rprop").not_implemented("dtype not supported");
    p_cpu.bump_version();
}

std::vector<Optimizer::NamedBuffers> Rprop::state_buffers() const {
    std::vector<NamedBuffers> out;
    out.emplace_back("step", clone_step_slots());
    out.emplace_back("prev", clone_state_slots(prev_grad_));
    out.emplace_back("step_size", clone_state_slots(step_size_));
    return out;
}

void Rprop::load_state_buffers(const std::vector<NamedBuffers>& bufs) {
    for (const auto& [name, tensors] : bufs) {
        if (name == "step")
            load_step_slots(tensors);
        else if (name == "prev")
            load_state_slots(prev_grad_, tensors);
        else if (name == "step_size")
            load_state_slots(step_size_, tensors);
    }
}

}  // namespace lucid
