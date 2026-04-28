#include "RMSNorm.h"

#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/cpu/Norm.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema RMSNormBackward::schema_v1{"rms_norm", 1, AmpPolicy::ForceFP32, true};

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

}  // namespace

TensorImplPtr RMSNormBackward::forward(const TensorImplPtr& x,
                                       const TensorImplPtr& gamma,
                                       double eps) {
    if (!x || !gamma)
        ErrorBuilder("rms_norm").fail("null input");
    if (x->dtype() != gamma->dtype())
        throw DtypeMismatch(std::string(dtype_name(x->dtype())),
                            std::string(dtype_name(gamma->dtype())), "rms_norm");
    if (x->device() != gamma->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "rms_norm");

    // γ shape must match trailing dims of x.
    if (gamma->shape().size() > x->shape().size())
        throw ShapeMismatch(x->shape(), gamma->shape(), "rms_norm: γ has more dims than x");
    const std::size_t Dn = gamma->shape().size();
    const std::size_t lead = x->shape().size() - Dn;
    for (std::size_t i = 0; i < Dn; ++i) {
        if (x->shape()[lead + i] != gamma->shape()[i]) {
            throw ShapeMismatch(x->shape(), gamma->shape(),
                                "rms_norm: γ must match trailing dims of x");
        }
    }
    std::size_t outer = 1, N = 1;
    for (std::size_t i = 0; i < lead; ++i)
        outer *= static_cast<std::size_t>(x->shape()[i]);
    for (std::size_t i = 0; i < Dn; ++i)
        N *= static_cast<std::size_t>(gamma->shape()[i]);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};

    Storage out_storage;
    Storage saved_rstd;

    if (x->device() == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage());
        const auto& gg = std::get<GpuStorage>(gamma->storage());
        if (!gx.arr || !gg.arr) {
            ErrorBuilder("rms_norm").fail("null GPU input");
        }
        Shape flatX{static_cast<std::int64_t>(outer), static_cast<std::int64_t>(N)};
        Shape flatG{1, static_cast<std::int64_t>(N)};
        auto x_2d = ::mlx::core::reshape(*gx.arr, gpu::to_mlx_shape(flatX));
        auto g_2d = ::mlx::core::reshape(*gg.arr, gpu::to_mlx_shape(flatG));
        auto sq = ::mlx::core::square(x_2d);
        auto ms = ::mlx::core::mean(sq, std::vector<int>{1}, /*keepdims=*/true);
        ::mlx::core::array eps_arr(eps, gpu::to_mlx_dtype(x->dtype()));
        auto rstd = ::mlx::core::rsqrt(::mlx::core::add(ms, eps_arr));
        auto xnorm = ::mlx::core::multiply(x_2d, rstd);
        auto y_2d = ::mlx::core::multiply(xnorm, g_2d);
        auto y = ::mlx::core::reshape(y_2d, gpu::to_mlx_shape(x->shape()));
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype())};
        saved_rstd = Storage{gpu::wrap_mlx_array(std::move(rstd), x->dtype())};
    } else {
        auto y_cpu = allocate_size(outer * N, x->dtype());
        auto rstd_cpu = allocate_size(outer, x->dtype());

        if (outer * N > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x->storage());
            const auto& g_cpu = std::get<CpuStorage>(gamma->storage());
            switch (x->dtype()) {
                case Dtype::F32:
                    backend::cpu::rms_norm_forward_f32(
                        reinterpret_cast<const float*>(x_cpu.ptr.get()),
                        reinterpret_cast<const float*>(g_cpu.ptr.get()),
                        reinterpret_cast<float*>(y_cpu.ptr.get()),
                        reinterpret_cast<float*>(rstd_cpu.ptr.get()), outer, N, eps);
                    break;
                case Dtype::F64:
                    backend::cpu::rms_norm_forward_f64(
                        reinterpret_cast<const double*>(x_cpu.ptr.get()),
                        reinterpret_cast<const double*>(g_cpu.ptr.get()),
                        reinterpret_cast<double*>(y_cpu.ptr.get()),
                        reinterpret_cast<double*>(rstd_cpu.ptr.get()), outer, N, eps);
                    break;
                default:
                    ErrorBuilder("rms_norm").not_implemented("dtype not supported (F32/F64)");
            }
        }
        out_storage = Storage{std::move(y_cpu)};
        saved_rstd = Storage{std::move(rstd_cpu)};
    }
    scope.set_flops(static_cast<std::int64_t>(outer * N) * 4);

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), x->shape(), x->dtype(), x->device(),
                                     /*requires_grad=*/false);

    const bool needs_grad =
        GradMode::is_enabled() && (x->requires_grad() || gamma->requires_grad());
    if (!needs_grad)
        return out;

    auto x_edge = detail::ensure_grad_fn(x);
    auto g_edge = detail::ensure_grad_fn(gamma);

    auto bwd = std::make_shared<RMSNormBackward>();
    bwd->input_shapes_ = {x->shape(), gamma->shape()};
    bwd->out_shape_ = out->shape();
    bwd->dtype_ = x->dtype();
    bwd->device_ = x->device();
    bwd->input_tensors_ = {x, gamma};
    bwd->saved_inputs_ = {x->storage(), gamma->storage()};
    bwd->saved_rstd_ = std::move(saved_rstd);
    bwd->outer_ = outer;
    bwd->N_ = N;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(g_edge, 0)});
    bwd->set_saved_versions({x->version(), gamma->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

std::vector<Storage> RMSNormBackward::apply(Storage grad_out) {
    if (device_ == Device::GPU) {
        // RMSNorm backward (no β, no mean subtraction):
        //   xnorm = x * rstd
        //   gx = g * gamma
        //   dgamma = sum(g * xnorm, axis=0)
        //   dx = rstd * (gx − xnorm · mean(gx · xnorm, axis=1, keepdims))
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gg = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gR = std::get<GpuStorage>(saved_rstd_);
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gx.arr || !gg.arr || !gR.arr || !gG.arr) {
            ErrorBuilder("rms_norm backward").fail("null GPU array");
        }
        Shape flatX{static_cast<std::int64_t>(outer_), static_cast<std::int64_t>(N_)};
        Shape flatG{1, static_cast<std::int64_t>(N_)};
        auto x_2d = ::mlx::core::reshape(*gx.arr, gpu::to_mlx_shape(flatX));
        auto g_2d = ::mlx::core::reshape(*gG.arr, gpu::to_mlx_shape(flatX));
        auto gamma_2d = ::mlx::core::reshape(*gg.arr, gpu::to_mlx_shape(flatG));
        auto xnorm = ::mlx::core::multiply(x_2d, *gR.arr);
        auto dgamma_2d = ::mlx::core::sum(::mlx::core::multiply(g_2d, xnorm), std::vector<int>{0},
                                          /*keepdims=*/false);
        auto gx_scaled = ::mlx::core::multiply(g_2d, gamma_2d);
        auto m = ::mlx::core::mean(::mlx::core::multiply(gx_scaled, xnorm), std::vector<int>{1},
                                   /*keepdims=*/true);
        auto dx_2d = ::mlx::core::multiply(
            *gR.arr, ::mlx::core::subtract(gx_scaled, ::mlx::core::multiply(xnorm, m)));
        auto dx = ::mlx::core::reshape(dx_2d, gpu::to_mlx_shape(input_shapes_[0]));
        auto dgamma = ::mlx::core::reshape(dgamma_2d, gpu::to_mlx_shape(input_shapes_[1]));
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dgamma), dtype_)}};
    }

    auto dx_cpu = allocate_size(outer_ * N_, dtype_);
    auto dgamma_cpu = allocate_size(N_, dtype_);

    if (outer_ * N_ > 0) {
        const auto& x_cpu = std::get<CpuStorage>(saved_inputs_[0]);
        const auto& gamma_cpu = std::get<CpuStorage>(saved_inputs_[1]);
        const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd_);
        const auto& g_cpu = std::get<CpuStorage>(grad_out);

        switch (dtype_) {
            case Dtype::F32:
                backend::cpu::rms_norm_backward_f32(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                    reinterpret_cast<float*>(dx_cpu.ptr.get()),
                    reinterpret_cast<float*>(dgamma_cpu.ptr.get()), outer_, N_);
                break;
            case Dtype::F64:
                backend::cpu::rms_norm_backward_f64(
                    reinterpret_cast<const double*>(x_cpu.ptr.get()),
                    reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const double*>(g_cpu.ptr.get()),
                    reinterpret_cast<double*>(dx_cpu.ptr.get()),
                    reinterpret_cast<double*>(dgamma_cpu.ptr.get()), outer_, N_);
                break;
            default:
                ErrorBuilder("rms_norm backward").not_implemented("dtype not supported");
        }
    } else {
        if (dx_cpu.nbytes)
            std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
        if (dgamma_cpu.nbytes)
            std::memset(dgamma_cpu.ptr.get(), 0, dgamma_cpu.nbytes);
    }

    return {Storage{std::move(dx_cpu)}, Storage{std::move(dgamma_cpu)}};
}

TensorImplPtr rms_norm_op(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps) {
    return RMSNormBackward::forward(x, gamma, eps);
}

LUCID_REGISTER_OP(RMSNormBackward)

}  // namespace lucid
