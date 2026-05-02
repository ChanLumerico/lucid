#include "Prop.h"

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
      centered_(centered) {}

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
            auto& ga = gpu_get(grad_avg_[i]);
            auto new_ga = ::mlx::core::add(::mlx::core::multiply(mlx_scalar(alpha_, dt), *ga.arr),
                                           ::mlx::core::multiply(mlx_scalar(1.0 - alpha_, dt), g));
            gpu_replace(ga, ::mlx::core::array(new_ga), dt);
            avg = ::mlx::core::subtract(new_sq, ::mlx::core::square(new_ga));
        }

        auto denom = ::mlx::core::add(::mlx::core::sqrt(avg), mlx_scalar(eps_, dt));
        ::mlx::core::array update = ::mlx::core::divide(g, denom);
        if (momentum_ != 0.0) {
            auto& mb = gpu_get(moment_buf_[i]);
            auto new_mb =
                ::mlx::core::add(::mlx::core::multiply(mlx_scalar(momentum_, dt), *mb.arr), update);
            gpu_replace(mb, ::mlx::core::array(new_mb), dt);
            update = new_mb;
        }
        auto new_p =
            ::mlx::core::subtract(*pg.arr, ::mlx::core::multiply(mlx_scalar(lr_, dt), update));
        gpu_replace(pg, std::move(new_p), dt);
        pg.bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* SQ = cpu_ptr<T>(square_avg_[i]);
        T* GA = centered_ ? cpu_ptr<T>(grad_avg_[i]) : nullptr;
        T* MB = (momentum_ != 0.0) ? cpu_ptr<T>(moment_buf_[i]) : nullptr;
        const T lrT = static_cast<T>(lr_);
        const T aT = static_cast<T>(alpha_);
        const T omaT = static_cast<T>(1.0 - alpha_);
        const T epsT = static_cast<T>(eps_);
        const T wdT = static_cast<T>(weight_decay_);
        const T mT = static_cast<T>(momentum_);
        for (std::size_t k = 0; k < n; ++k) {
            T g = G[k];
            if (weight_decay_ != 0.0)
                g += wdT * P[k];
            SQ[k] = aT * SQ[k] + omaT * g * g;
            T avg = SQ[k];
            if (GA) {
                GA[k] = aT * GA[k] + omaT * g;
                avg = SQ[k] - GA[k] * GA[k];
            }

            const T denom = std::sqrt(avg) + epsT;
            T update = g / denom;
            if (MB) {
                MB[k] = mT * MB[k] + update;
                update = MB[k];
            }
            P[k] -= lrT * update;
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
      step_max_(step_max) {}

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

void Rprop::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    const auto dt = p->dtype();
    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& pv = gpu_get(prev_grad_[i]);
        auto& ss = gpu_get(step_size_[i]);

        ::mlx::core::array sign_change = ::mlx::core::multiply(*gg.arr, *pv.arr);
        ::mlx::core::array zero_arr = mlx_scalar(0.0, dt);

        auto pos_mask = ::mlx::core::greater(sign_change, zero_arr);
        auto inc = ::mlx::core::multiply(mlx_scalar(eta_plus_, dt), *ss.arr);
        auto new_ss = ::mlx::core::where(pos_mask, inc, *ss.arr);

        auto neg_mask = ::mlx::core::less(sign_change, zero_arr);
        auto dec = ::mlx::core::multiply(mlx_scalar(eta_minus_, dt), new_ss);
        new_ss = ::mlx::core::where(neg_mask, dec, new_ss);
        new_ss = ::mlx::core::clip(new_ss, mlx_scalar(step_min_, dt), mlx_scalar(step_max_, dt));
        gpu_replace(ss, ::mlx::core::array(new_ss), dt);

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
            const T sc = G[k] * PV[k];
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

}  // namespace lucid
