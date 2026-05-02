#include "Ada.h"

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

// =====================================================================
// Adamax
// =====================================================================

Adamax::Adamax(std::vector<std::shared_ptr<TensorImpl>> p,
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

void Adamax::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (m_.size() < params_.size())
        m_.resize(params_.size());
    if (u_.size() < params_.size())
        u_.resize(params_.size());
    m_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    u_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
}

void Adamax::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    if (i == 0)
        ++step_count_;
    const auto dt = p->dtype();
    const std::int64_t step = step_count_;
    const double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(step));
    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& mg = gpu_get(m_[i]);
        auto& ug = gpu_get(u_[i]);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        auto new_m = ::mlx::core::add(::mlx::core::multiply(mlx_scalar(beta1_, dt), *mg.arr),
                                      ::mlx::core::multiply(mlx_scalar(1.0 - beta1_, dt), g));
        auto new_u = ::mlx::core::maximum(::mlx::core::multiply(mlx_scalar(beta2_, dt), *ug.arr),
                                          ::mlx::core::abs(g));
        gpu_replace(mg, ::mlx::core::array(new_m), dt);
        gpu_replace(ug, ::mlx::core::array(new_u), dt);
        auto step_size = mlx_scalar(lr_ / bc1, dt);
        auto denom = ::mlx::core::add(new_u, mlx_scalar(eps_, dt));
        auto update = ::mlx::core::multiply(step_size, ::mlx::core::divide(new_m, denom));
        gpu_replace(pg, ::mlx::core::subtract(*pg.arr, update), dt);
        pg.bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* M = cpu_ptr<T>(m_[i]);
        T* U = cpu_ptr<T>(u_[i]);
        const T b1T = static_cast<T>(beta1_);
        const T b2T = static_cast<T>(beta2_);
        const T omb1 = static_cast<T>(1.0 - beta1_);
        const T epsT = static_cast<T>(eps_);
        const T wdT = static_cast<T>(weight_decay_);
        const T step_size = static_cast<T>(lr_ / bc1);
        for (std::size_t k = 0; k < n; ++k) {
            T g = G[k];
            if (weight_decay_ != 0.0)
                g += wdT * P[k];
            M[k] = b1T * M[k] + omb1 * g;
            const T abs_g = std::abs(g);
            const T b2u = b2T * U[k];
            U[k] = (b2u > abs_g) ? b2u : abs_g;
            P[k] -= step_size * (M[k] / (U[k] + epsT));
        }
    };
    if (dt == Dtype::F32)
        step_cpu(reinterpret_cast<float*>(p_cpu.ptr.get()), cpu_cptr<float>(grad));
    else if (dt == Dtype::F64)
        step_cpu(reinterpret_cast<double*>(p_cpu.ptr.get()), cpu_cptr<double>(grad));
    else
        ErrorBuilder("Adamax").not_implemented("dtype not supported");
    p_cpu.bump_version();
}

// =====================================================================
// Adagrad
// =====================================================================

Adagrad::Adagrad(
    std::vector<std::shared_ptr<TensorImpl>> p, double lr, double eps, double wd, double init_acc)
    : Optimizer(std::move(p)),
      lr_(lr),
      eps_(eps),
      weight_decay_(wd),
      initial_accumulator_value_(init_acc) {}

void Adagrad::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (sum_sq_grad_.size() < params_.size())
        sum_sq_grad_.resize(params_.size());
    sum_sq_grad_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    if (initial_accumulator_value_ != 0.0) {
        if (p->device() == Device::GPU) {
            auto ones_arr =
                ::mlx::core::ones(gpu::to_mlx_shape(p->shape()), gpu::to_mlx_dtype(p->dtype()));
            auto init =
                ::mlx::core::multiply(mlx_scalar(initial_accumulator_value_, p->dtype()), ones_arr);
            gpu_replace(gpu_get(sum_sq_grad_[i]), std::move(init), p->dtype());
        } else {
            const std::size_t n = cpu_numel(*p);
            if (p->dtype() == Dtype::F32) {
                auto* q = cpu_ptr<float>(sum_sq_grad_[i]);
                const auto v = static_cast<float>(initial_accumulator_value_);
                for (std::size_t k = 0; k < n; ++k)
                    q[k] = v;
            } else if (p->dtype() == Dtype::F64) {
                auto* q = cpu_ptr<double>(sum_sq_grad_[i]);
                for (std::size_t k = 0; k < n; ++k)
                    q[k] = initial_accumulator_value_;
            }
        }
    }
}

void Adagrad::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    const auto dt = p->dtype();
    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& ss = gpu_get(sum_sq_grad_[i]);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        auto new_ss = ::mlx::core::add(*ss.arr, ::mlx::core::square(g));
        gpu_replace(ss, ::mlx::core::array(new_ss), dt);
        auto denom = ::mlx::core::add(::mlx::core::sqrt(new_ss), mlx_scalar(eps_, dt));
        auto update = ::mlx::core::multiply(mlx_scalar(lr_, dt), ::mlx::core::divide(g, denom));
        gpu_replace(pg, ::mlx::core::subtract(*pg.arr, update), dt);
        pg.bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* SS = cpu_ptr<T>(sum_sq_grad_[i]);
        const T lrT = static_cast<T>(lr_);
        const T epsT = static_cast<T>(eps_);
        const T wdT = static_cast<T>(weight_decay_);
        for (std::size_t k = 0; k < n; ++k) {
            T g = G[k];
            if (weight_decay_ != 0.0)
                g += wdT * P[k];
            SS[k] += g * g;
            P[k] -= lrT * g / (std::sqrt(SS[k]) + epsT);
        }
    };
    if (dt == Dtype::F32)
        step_cpu(reinterpret_cast<float*>(p_cpu.ptr.get()), cpu_cptr<float>(grad));
    else if (dt == Dtype::F64)
        step_cpu(reinterpret_cast<double*>(p_cpu.ptr.get()), cpu_cptr<double>(grad));
    else
        ErrorBuilder("Adagrad").not_implemented("dtype not supported");
    p_cpu.bump_version();
}

// =====================================================================
// Adadelta
// =====================================================================

Adadelta::Adadelta(
    std::vector<std::shared_ptr<TensorImpl>> p, double lr, double rho, double eps, double wd)
    : Optimizer(std::move(p)), lr_(lr), rho_(rho), eps_(eps), weight_decay_(wd) {}

void Adadelta::init_state_slot(std::size_t i, const std::shared_ptr<TensorImpl>& p) {
    if (sq_avg_.size() < params_.size())
        sq_avg_.resize(params_.size());
    if (accumulated_update_.size() < params_.size())
        accumulated_update_.resize(params_.size());
    sq_avg_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    accumulated_update_[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
}

void Adadelta::update_one(std::size_t i, std::shared_ptr<TensorImpl>& p, const Storage& grad) {
    const auto dt = p->dtype();
    if (p->device() == Device::GPU) {
        auto& pg = gpu_get(p->mutable_storage());
        const auto& gg = gpu_get(grad);
        auto& sq = gpu_get(sq_avg_[i]);
        auto& acc = gpu_get(accumulated_update_[i]);
        ::mlx::core::array g = *gg.arr;
        if (weight_decay_ != 0.0) {
            g = ::mlx::core::add(g, ::mlx::core::multiply(mlx_scalar(weight_decay_, dt), *pg.arr));
        }
        auto new_sq = ::mlx::core::add(
            ::mlx::core::multiply(mlx_scalar(rho_, dt), *sq.arr),
            ::mlx::core::multiply(mlx_scalar(1.0 - rho_, dt), ::mlx::core::square(g)));
        gpu_replace(sq, ::mlx::core::array(new_sq), dt);
        auto update = ::mlx::core::multiply(
            ::mlx::core::divide(::mlx::core::sqrt(::mlx::core::add(*acc.arr, mlx_scalar(eps_, dt))),
                                ::mlx::core::sqrt(::mlx::core::add(new_sq, mlx_scalar(eps_, dt)))),
            g);
        auto new_acc = ::mlx::core::add(
            ::mlx::core::multiply(mlx_scalar(rho_, dt), *acc.arr),
            ::mlx::core::multiply(mlx_scalar(1.0 - rho_, dt), ::mlx::core::square(update)));
        gpu_replace(acc, ::mlx::core::array(new_acc), dt);
        gpu_replace(
            pg, ::mlx::core::subtract(*pg.arr, ::mlx::core::multiply(mlx_scalar(lr_, dt), update)),
            dt);
        pg.bump_version();
        return;
    }
    const std::size_t n = cpu_numel(*p);
    auto& p_cpu = storage_cpu(p->mutable_storage());
    auto step_cpu = [&](auto* P, const auto* G) {
        using T = std::remove_pointer_t<decltype(P)>;
        T* SQ = cpu_ptr<T>(sq_avg_[i]);
        T* AC = cpu_ptr<T>(accumulated_update_[i]);
        const T lrT = static_cast<T>(lr_);
        const T rT = static_cast<T>(rho_);
        const T omrT = static_cast<T>(1.0 - rho_);
        const T epsT = static_cast<T>(eps_);
        const T wdT = static_cast<T>(weight_decay_);
        for (std::size_t k = 0; k < n; ++k) {
            T g = G[k];
            if (weight_decay_ != 0.0)
                g += wdT * P[k];
            SQ[k] = rT * SQ[k] + omrT * g * g;
            const T update = (std::sqrt(AC[k] + epsT) / std::sqrt(SQ[k] + epsT)) * g;
            AC[k] = rT * AC[k] + omrT * update * update;
            P[k] -= lrT * update;
        }
    };
    if (dt == Dtype::F32)
        step_cpu(reinterpret_cast<float*>(p_cpu.ptr.get()), cpu_cptr<float>(grad));
    else if (dt == Dtype::F64)
        step_cpu(reinterpret_cast<double*>(p_cpu.ptr.get()), cpu_cptr<double>(grad));
    else
        ErrorBuilder("Adadelta").not_implemented("dtype not supported");
    p_cpu.bump_version();
}

}  // namespace lucid
