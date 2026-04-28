#include "Loss.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
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

namespace {

using gpu::mlx_scalar;

// ---------------- GPU helpers (shared across loss kernels) ----------------
//
// Stable BCE-with-logits primitive on MLX:
//   loss = max(x, 0) - x*y + log_weight * log1p(exp(-|x|))
// where log_weight = (pos_weight - 1) * y + 1.

::mlx::core::array mlx_apply_reduction(const ::mlx::core::array& l, Reduction red) {
    if (red == Reduction::None)
        return l;
    auto s = ::mlx::core::sum(l, /*keepdims=*/false);
    if (red == Reduction::Mean) {
        const double n = static_cast<double>(l.size());
        return ::mlx::core::divide(s, mlx_scalar(n, l.dtype()));
    }
    return s;  // Sum
}

::mlx::core::array mlx_grad_scale(const ::mlx::core::array& gout,
                                  Reduction red,
                                  std::size_t numel,
                                  ::mlx::core::Dtype dt) {
    if (red == Reduction::None)
        return gout;
    if (red == Reduction::Mean) {
        return ::mlx::core::divide(gout, mlx_scalar(static_cast<double>(numel), dt));
    }
    return gout;  // Sum
}

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

CpuStorage make_scalar(double val, Dtype dt) {
    auto s = allocate_size(1, dt);
    switch (dt) {
        case Dtype::F32:
            *reinterpret_cast<float*>(s.ptr.get()) = static_cast<float>(val);
            break;
        case Dtype::F64:
            *reinterpret_cast<double*>(s.ptr.get()) = val;
            break;
        default:
            ErrorBuilder("loss").not_implemented("dtype not supported");
    }
    return s;
}

template <typename T>
T accumulate(const T* p, std::size_t n) {
    T s = T{0};
    for (std::size_t i = 0; i < n; ++i)
        s += p[i];
    return s;
}

template <typename T>
Storage apply_reduction(const T* lp, std::size_t numel, Reduction red, Dtype dt) {
    if (red == Reduction::None) {
        auto out = allocate_size(numel, dt);
        std::memcpy(out.ptr.get(), lp, numel * dtype_size(dt));
        return Storage{std::move(out)};
    }
    T s = accumulate(lp, numel);
    if (red == Reduction::Mean)
        s /= static_cast<T>(numel);
    return Storage{make_scalar(static_cast<double>(s), dt)};
}

Shape reduced_shape(const Shape& in, Reduction red) {
    if (red == Reduction::None)
        return in;
    return Shape{};
}

// Read a target index value from any-int / float storage at flat position i.
inline std::int64_t read_target(const CpuStorage& ts, std::size_t i) {
    switch (ts.dtype) {
        case Dtype::I8:
            return static_cast<std::int64_t>(reinterpret_cast<const std::int8_t*>(ts.ptr.get())[i]);
        case Dtype::I16:
            return static_cast<std::int64_t>(
                reinterpret_cast<const std::int16_t*>(ts.ptr.get())[i]);
        case Dtype::I32:
            return static_cast<std::int64_t>(
                reinterpret_cast<const std::int32_t*>(ts.ptr.get())[i]);
        case Dtype::I64:
            return reinterpret_cast<const std::int64_t*>(ts.ptr.get())[i];
        case Dtype::F32:
            return static_cast<std::int64_t>(reinterpret_cast<const float*>(ts.ptr.get())[i]);
        case Dtype::F64:
            return static_cast<std::int64_t>(reinterpret_cast<const double*>(ts.ptr.get())[i]);
        default:
            ErrorBuilder("loss target").not_implemented("dtype not supported");
    }
}

}  // namespace

// ===================================================================
// MSE
// ===================================================================

const OpSchema MseLossBackward::schema_v1{"mse_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr MseLossBackward::forward(const TensorImplPtr& input,
                                       const TensorImplPtr& target,
                                       Reduction reduction) {
    if (!input || !target)
        ErrorBuilder("mse_loss").fail("null input");
    if (input->shape_ != target->shape_)
        throw ShapeMismatch(input->shape_, target->shape_, "mse_loss: input/target shape mismatch");
    if (input->dtype_ != target->dtype_)
        throw DtypeMismatch(std::string(dtype_name(input->dtype_)),
                            std::string(dtype_name(target->dtype_)), "mse_loss");

    const std::size_t numel = input->numel();
    OpScopeFull scope{schema_v1.name, input->device_, input->dtype_,
                      reduced_shape(input->shape_, reduction)};

    Storage out_storage;
    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto& gt = std::get<GpuStorage>(target->storage_);
        if (!gx.arr || !gt.arr)
            ErrorBuilder("mse_loss").fail("null GPU array");
        auto diff = ::mlx::core::subtract(*gx.arr, *gt.arr);
        auto sq = ::mlx::core::multiply(diff, diff);
        auto red = mlx_apply_reduction(sq, reduction);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(red), input->dtype_)};
    } else {
        auto loss_buf = allocate_size(numel, input->dtype_);
        const auto& xs = std::get<CpuStorage>(input->storage_);
        const auto& ts = std::get<CpuStorage>(target->storage_);
        switch (input->dtype_) {
            case Dtype::F32: {
                auto* xp = reinterpret_cast<const float*>(xs.ptr.get());
                auto* tp = reinterpret_cast<const float*>(ts.ptr.get());
                auto* lp = reinterpret_cast<float*>(loss_buf.ptr.get());
                for (std::size_t i = 0; i < numel; ++i) {
                    const float d = xp[i] - tp[i];
                    lp[i] = d * d;
                }
                out_storage = apply_reduction<float>(lp, numel, reduction, input->dtype_);
                break;
            }
            case Dtype::F64: {
                auto* xp = reinterpret_cast<const double*>(xs.ptr.get());
                auto* tp = reinterpret_cast<const double*>(ts.ptr.get());
                auto* lp = reinterpret_cast<double*>(loss_buf.ptr.get());
                for (std::size_t i = 0; i < numel; ++i) {
                    const double d = xp[i] - tp[i];
                    lp[i] = d * d;
                }
                out_storage = apply_reduction<double>(lp, numel, reduction, input->dtype_);
                break;
            }
            default:
                ErrorBuilder("mse_loss").not_implemented("dtype not supported");
        }
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape_, reduction), input->dtype_,
                                            input->device_, false);

    if (!GradMode::is_enabled() || !(input->requires_grad_ || target->requires_grad_))
        return out;

    auto x_edge = detail::ensure_grad_fn(input);
    auto t_edge = detail::ensure_grad_fn(target);
    auto bwd = std::make_shared<MseLossBackward>();
    bwd->input_shapes_ = {input->shape_, target->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = input->dtype_;
    bwd->device_ = input->device_;
    bwd->input_tensors_ = {input, target};
    bwd->saved_inputs_ = {input->storage_, target->storage_};
    bwd->reduction_ = reduction;
    bwd->orig_shape_ = input->shape_;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(t_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{static_cast<std::int64_t>(input->version_),
                                                      static_cast<std::int64_t>(target->version_)});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> MseLossBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(orig_shape_);

    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gt = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);
        auto scaled = mlx_grad_scale(*gg.arr, reduction_, numel, mlx_dt);
        if (reduction_ != Reduction::None)
            scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(orig_shape_));
        auto diff = ::mlx::core::subtract(*gx.arr, *gt.arr);
        auto two = mlx_scalar(2.0, mlx_dt);
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(two, diff), scaled);
        auto dt = ::mlx::core::negative(dx);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dt), dtype_)}};
    }

    auto dx_cpu = allocate_size(numel, dtype_);
    auto dt_cpu = allocate_size(numel, dtype_);
    const auto& xs = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& ts = std::get<CpuStorage>(saved_inputs_[1]);
    const auto& gs = std::get<CpuStorage>(grad_out);

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
        auto* dtp = reinterpret_cast<T*>(dt_cpu.ptr.get());
        const bool elem = (reduction_ == Reduction::None);
        const T mean_scale = static_cast<T>(numel);
        for (std::size_t i = 0; i < numel; ++i) {
            const T go = elem ? gp[i] : gp[0];
            const T scale = (reduction_ == Reduction::Mean) ? (go / mean_scale) : go;
            const T d = xp[i] - tp[i];
            dxp[i] = static_cast<T>(2) * d * scale;
            dtp[i] = -dxp[i];
        }
    };
    if (dtype_ == Dtype::F32)
        compute(float{});
    else if (dtype_ == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("mse_loss backward").not_implemented("dtype not supported");
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dt_cpu)}};
}

TensorImplPtr mse_loss_op(const TensorImplPtr& input, const TensorImplPtr& target, int reduction) {
    return MseLossBackward::forward(input, target, static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(MseLossBackward)

// ===================================================================
// BCE — input in [0, 1].  Slots: 0=input, 1=target, 2=weight (required).
// ===================================================================

const OpSchema BCELossBackward::schema_v1{"bce_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BCELossBackward::forward(const TensorImplPtr& input,
                                       const TensorImplPtr& target,
                                       const TensorImplPtr& weight,
                                       Reduction reduction,
                                       double eps) {
    if (!input || !target || !weight)
        ErrorBuilder("bce_loss")
            .fail("input/target/weight required (pass ones for weight if not used)");
    if (input->shape_ != target->shape_)
        throw ShapeMismatch(input->shape_, target->shape_, "bce_loss: input/target shape mismatch");

    const std::size_t numel = input->numel();
    OpScopeFull scope{schema_v1.name, input->device_, input->dtype_,
                      reduced_shape(input->shape_, reduction)};

    Storage out_storage;
    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto& gt = std::get<GpuStorage>(target->storage_);
        const auto& gw = std::get<GpuStorage>(weight->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype_);
        auto e_lo = mlx_scalar(eps, mlx_dt);
        auto one = mlx_scalar(1.0, mlx_dt);
        auto e_hi = ::mlx::core::subtract(one, e_lo);
        auto p = ::mlx::core::clip(*gx.arr, std::optional<::mlx::core::array>(e_lo),
                                   std::optional<::mlx::core::array>(e_hi));
        auto one_mt = ::mlx::core::subtract(one, *gt.arr);
        auto one_mp = ::mlx::core::subtract(one, p);
        auto term1 = ::mlx::core::multiply(*gt.arr, ::mlx::core::log(p));
        auto term2 = ::mlx::core::multiply(one_mt, ::mlx::core::log(one_mp));
        auto l = ::mlx::core::negative(::mlx::core::add(term1, term2));
        auto wl = ::mlx::core::multiply(*gw.arr, l);
        auto red = mlx_apply_reduction(wl, reduction);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(red), input->dtype_)};
    } else {
        auto loss_buf = allocate_size(numel, input->dtype_);
        const auto& xs = std::get<CpuStorage>(input->storage_);
        const auto& ts = std::get<CpuStorage>(target->storage_);
        const auto& ws = std::get<CpuStorage>(weight->storage_);

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
            auto* lp = reinterpret_cast<T*>(loss_buf.ptr.get());
            const T e = static_cast<T>(eps);
            for (std::size_t i = 0; i < numel; ++i) {
                T p = std::min(std::max(xp[i], e), static_cast<T>(1) - e);
                const T l = -(tp[i] * std::log(p) +
                              (static_cast<T>(1) - tp[i]) * std::log(static_cast<T>(1) - p));
                lp[i] = wp[i] * l;
            }
        };
        if (input->dtype_ == Dtype::F32)
            compute(float{});
        else if (input->dtype_ == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("bce_loss").not_implemented("dtype not supported");

        if (input->dtype_ == Dtype::F32) {
            out_storage = apply_reduction<float>(reinterpret_cast<float*>(loss_buf.ptr.get()),
                                                 numel, reduction, input->dtype_);
        } else {
            out_storage = apply_reduction<double>(reinterpret_cast<double*>(loss_buf.ptr.get()),
                                                  numel, reduction, input->dtype_);
        }
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape_, reduction), input->dtype_,
                                            input->device_, false);

    const bool any_grad = input->requires_grad_ || target->requires_grad_ || weight->requires_grad_;
    if (!GradMode::is_enabled() || !any_grad)
        return out;

    auto x_edge = detail::ensure_grad_fn(input);
    auto t_edge = detail::ensure_grad_fn(target);
    auto w_edge = detail::ensure_grad_fn(weight);
    auto bwd = std::make_shared<BCELossBackward>();
    bwd->input_shapes_ = {input->shape_, target->shape_, weight->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = input->dtype_;
    bwd->device_ = input->device_;
    bwd->input_tensors_ = {input, target, weight};
    bwd->saved_inputs_ = {input->storage_, target->storage_, weight->storage_};
    bwd->reduction_ = reduction;
    bwd->eps_ = eps;
    bwd->orig_shape_ = input->shape_;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(t_edge, 0), Edge(w_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{static_cast<std::int64_t>(input->version_),
                                                      static_cast<std::int64_t>(target->version_),
                                                      static_cast<std::int64_t>(weight->version_)});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> BCELossBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(orig_shape_);

    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gt = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gw = std::get<GpuStorage>(saved_inputs_[2]);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);
        auto e_lo = mlx_scalar(eps_, mlx_dt);
        auto one = mlx_scalar(1.0, mlx_dt);
        auto e_hi = ::mlx::core::subtract(one, e_lo);
        auto p = ::mlx::core::clip(*gx.arr, std::optional<::mlx::core::array>(e_lo),
                                   std::optional<::mlx::core::array>(e_hi));
        auto one_mp = ::mlx::core::subtract(one, p);
        auto one_mt = ::mlx::core::subtract(one, *gt.arr);
        auto log_p = ::mlx::core::log(p);
        auto log_1mp = ::mlx::core::log(one_mp);
        auto dlp = ::mlx::core::add(::mlx::core::negative(::mlx::core::divide(*gt.arr, p)),
                                    ::mlx::core::divide(one_mt, one_mp));
        auto dly = ::mlx::core::add(::mlx::core::negative(log_p), log_1mp);
        auto l = ::mlx::core::negative(::mlx::core::add(::mlx::core::multiply(*gt.arr, log_p),
                                                        ::mlx::core::multiply(one_mt, log_1mp)));

        auto scaled = mlx_grad_scale(*gg.arr, reduction_, numel, mlx_dt);
        if (reduction_ != Reduction::None)
            scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(orig_shape_));
        auto dx = ::mlx::core::multiply(*gw.arr, ::mlx::core::multiply(dlp, scaled));
        auto dt = ::mlx::core::multiply(*gw.arr, ::mlx::core::multiply(dly, scaled));
        auto dw = ::mlx::core::multiply(l, scaled);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dt), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dw), dtype_)}};
    }

    auto dx_cpu = allocate_size(numel, dtype_);
    auto dt_cpu = allocate_size(numel, dtype_);
    auto dw_cpu = allocate_size(numel, dtype_);
    const auto& xs = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& ts = std::get<CpuStorage>(saved_inputs_[1]);
    const auto& ws = std::get<CpuStorage>(saved_inputs_[2]);
    const auto& gs = std::get<CpuStorage>(grad_out);

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
        auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
        auto* dtp = reinterpret_cast<T*>(dt_cpu.ptr.get());
        auto* dwp = reinterpret_cast<T*>(dw_cpu.ptr.get());
        const T e = static_cast<T>(eps_);
        const bool elem = (reduction_ == Reduction::None);
        const T mean_scale = static_cast<T>(numel);
        for (std::size_t i = 0; i < numel; ++i) {
            const T go = elem ? gp[i] : gp[0];
            const T scale = (reduction_ == Reduction::Mean) ? (go / mean_scale) : go;
            T p = std::min(std::max(xp[i], e), static_cast<T>(1) - e);
            const T y = tp[i];
            const T w = wp[i];
            const T dlp = -y / p + (static_cast<T>(1) - y) / (static_cast<T>(1) - p);
            dxp[i] = w * dlp * scale;
            const T dly = -std::log(p) + std::log(static_cast<T>(1) - p);
            dtp[i] = w * dly * scale;
            const T l =
                -(y * std::log(p) + (static_cast<T>(1) - y) * std::log(static_cast<T>(1) - p));
            dwp[i] = l * scale;
        }
    };
    if (dtype_ == Dtype::F32)
        compute(float{});
    else if (dtype_ == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("bce_loss backward").not_implemented("dtype not supported");
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dt_cpu)}, Storage{std::move(dw_cpu)}};
}

TensorImplPtr bce_loss_op(const TensorImplPtr& input,
                          const TensorImplPtr& target,
                          const TensorImplPtr& weight,
                          int reduction,
                          double eps) {
    return BCELossBackward::forward(input, target, weight, static_cast<Reduction>(reduction), eps);
}
LUCID_REGISTER_OP(BCELossBackward)

// ===================================================================
// BCE with logits.  Slots: 0=input, 1=target, 2=weight, 3=pos_weight.
//   l_i = max(x, 0) − x·y + log_weight · log1p(exp(−|x|))
//   log_weight = (pos_weight − 1)·y + 1
//   dL/dx = log_weight · sigm(x) − y
//   dL/dy = −x + (pos_weight − 1) · log1p(exp(−|x|))
//   dL/dw = l        (rec)
//   dL/dpw = w · y · log1p(exp(−|x|))
// ===================================================================

const OpSchema BCEWithLogitsBackward::schema_v1{"bce_with_logits", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BCEWithLogitsBackward::forward(const TensorImplPtr& input,
                                             const TensorImplPtr& target,
                                             const TensorImplPtr& weight,
                                             const TensorImplPtr& pos_weight,
                                             Reduction reduction) {
    if (!input || !target || !weight || !pos_weight)
        ErrorBuilder("bce_with_logits").fail("input/target/weight/pos_weight required");
    if (input->shape_ != target->shape_)
        throw ShapeMismatch(input->shape_, target->shape_,
                            "bce_with_logits: input/target shape mismatch");

    const std::size_t numel = input->numel();
    OpScopeFull scope{schema_v1.name, input->device_, input->dtype_,
                      reduced_shape(input->shape_, reduction)};

    Storage out_storage;
    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto& gt = std::get<GpuStorage>(target->storage_);
        const auto& gw = std::get<GpuStorage>(weight->storage_);
        const auto& gpw = std::get<GpuStorage>(pos_weight->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype_);
        auto one = mlx_scalar(1.0, mlx_dt);
        auto zero = mlx_scalar(0.0, mlx_dt);
        auto pw_m1 = ::mlx::core::subtract(*gpw.arr, one);
        auto log_weight = ::mlx::core::add(::mlx::core::multiply(pw_m1, *gt.arr), one);
        auto neg_abs_x = ::mlx::core::negative(::mlx::core::abs(*gx.arr));
        auto log1pexp = ::mlx::core::log1p(::mlx::core::exp(neg_abs_x));
        auto max0 = ::mlx::core::maximum(*gx.arr, zero);
        auto term = ::mlx::core::multiply(log_weight, log1pexp);
        auto xy = ::mlx::core::multiply(*gx.arr, *gt.arr);
        auto l = ::mlx::core::add(::mlx::core::subtract(max0, xy), term);
        auto wl = ::mlx::core::multiply(*gw.arr, l);
        auto red = mlx_apply_reduction(wl, reduction);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(red), input->dtype_)};
    } else {
        auto loss_buf = allocate_size(numel, input->dtype_);
        const auto& xs = std::get<CpuStorage>(input->storage_);
        const auto& ts = std::get<CpuStorage>(target->storage_);
        const auto& ws_raw = std::get<CpuStorage>(weight->storage_);
        const auto& pws_raw = std::get<CpuStorage>(pos_weight->storage_);

        // Broadcast weight and pos_weight to input shape so per-element
        // indexing in the loop is well-defined (PyTorch broadcasts both).
        CpuStorage ws_buf, pws_buf;
        const CpuStorage* ws_p = &ws_raw;
        const CpuStorage* pws_p = &pws_raw;
        if (weight->shape_ != input->shape_) {
            ws_buf = ::lucid::detail::broadcast_cpu(ws_raw, weight->shape_, input->shape_,
                                                    input->dtype_);
            ws_p = &ws_buf;
        }
        if (pos_weight->shape_ != input->shape_) {
            pws_buf = ::lucid::detail::broadcast_cpu(pws_raw, pos_weight->shape_, input->shape_,
                                                     input->dtype_);
            pws_p = &pws_buf;
        }
        // The broadcast_cpu helper lives in lucid::detail (defined in
        // ops/bfunc/_BinaryOp.h) so the explicit `::lucid::detail::` qualifier
        // suffices given the include above.

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            auto* wp = reinterpret_cast<const T*>(ws_p->ptr.get());
            auto* pwp = reinterpret_cast<const T*>(pws_p->ptr.get());
            auto* lp = reinterpret_cast<T*>(loss_buf.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) {
                const T x = xp[i];
                const T y = tp[i];
                const T pw = pwp[i];
                const T log_weight = (pw - static_cast<T>(1)) * y + static_cast<T>(1);
                const T log1pexp = std::log1p(std::exp(-std::abs(x)));
                T l = std::max(x, static_cast<T>(0)) - x * y + log_weight * log1pexp;
                lp[i] = wp[i] * l;
            }
        };
        if (input->dtype_ == Dtype::F32)
            compute(float{});
        else if (input->dtype_ == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("bce_with_logits").not_implemented("dtype not supported");

        if (input->dtype_ == Dtype::F32) {
            out_storage = apply_reduction<float>(reinterpret_cast<float*>(loss_buf.ptr.get()),
                                                 numel, reduction, input->dtype_);
        } else {
            out_storage = apply_reduction<double>(reinterpret_cast<double*>(loss_buf.ptr.get()),
                                                  numel, reduction, input->dtype_);
        }
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape_, reduction), input->dtype_,
                                            input->device_, false);

    const bool any_grad = input->requires_grad_ || target->requires_grad_ ||
                          weight->requires_grad_ || pos_weight->requires_grad_;
    if (!GradMode::is_enabled() || !any_grad)
        return out;

    auto x_edge = detail::ensure_grad_fn(input);
    auto t_edge = detail::ensure_grad_fn(target);
    auto w_edge = detail::ensure_grad_fn(weight);
    auto pw_edge = detail::ensure_grad_fn(pos_weight);
    auto bwd = std::make_shared<BCEWithLogitsBackward>();
    bwd->input_shapes_ = {input->shape_, target->shape_, weight->shape_, pos_weight->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = input->dtype_;
    bwd->device_ = input->device_;
    bwd->input_tensors_ = {input, target, weight, pos_weight};
    bwd->saved_inputs_ = {input->storage_, target->storage_, weight->storage_,
                          pos_weight->storage_};
    bwd->reduction_ = reduction;
    bwd->orig_shape_ = input->shape_;
    bwd->set_next_edges(
        std::vector<Edge>{Edge(x_edge, 0), Edge(t_edge, 0), Edge(w_edge, 0), Edge(pw_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{
        static_cast<std::int64_t>(input->version_), static_cast<std::int64_t>(target->version_),
        static_cast<std::int64_t>(weight->version_),
        static_cast<std::int64_t>(pos_weight->version_)});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> BCEWithLogitsBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(orig_shape_);

    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gt = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gw = std::get<GpuStorage>(saved_inputs_[2]);
        const auto& gpw = std::get<GpuStorage>(saved_inputs_[3]);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);
        auto one = mlx_scalar(1.0, mlx_dt);
        auto zero = mlx_scalar(0.0, mlx_dt);
        auto pw_m1 = ::mlx::core::subtract(*gpw.arr, one);
        auto log_weight = ::mlx::core::add(::mlx::core::multiply(pw_m1, *gt.arr), one);
        auto sigm = ::mlx::core::sigmoid(*gx.arr);
        auto neg_abs_x = ::mlx::core::negative(::mlx::core::abs(*gx.arr));
        auto log1pexp = ::mlx::core::log1p(::mlx::core::exp(neg_abs_x));
        auto dlx = ::mlx::core::subtract(::mlx::core::multiply(log_weight, sigm), *gt.arr);
        auto dly = ::mlx::core::add(::mlx::core::negative(*gx.arr),
                                    ::mlx::core::multiply(pw_m1, log1pexp));
        auto max0 = ::mlx::core::maximum(*gx.arr, zero);
        auto xy = ::mlx::core::multiply(*gx.arr, *gt.arr);
        auto l = ::mlx::core::add(::mlx::core::subtract(max0, xy),
                                  ::mlx::core::multiply(log_weight, log1pexp));
        auto dpw = ::mlx::core::multiply(*gw.arr, ::mlx::core::multiply(*gt.arr, log1pexp));

        auto scaled = mlx_grad_scale(*gg.arr, reduction_, numel, mlx_dt);
        if (reduction_ != Reduction::None)
            scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(orig_shape_));
        auto dx = ::mlx::core::multiply(*gw.arr, ::mlx::core::multiply(dlx, scaled));
        auto dt = ::mlx::core::multiply(*gw.arr, ::mlx::core::multiply(dly, scaled));
        auto dw_ = ::mlx::core::multiply(l, scaled);
        auto dpw_scaled = ::mlx::core::multiply(dpw, scaled);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dt), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dw_), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dpw_scaled), dtype_)}};
    }

    auto dx_cpu = allocate_size(numel, dtype_);
    auto dt_cpu = allocate_size(numel, dtype_);
    auto dw_cpu = allocate_size(numel, dtype_);
    auto dpw_cpu = allocate_size(numel, dtype_);

    const auto& xs = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& ts = std::get<CpuStorage>(saved_inputs_[1]);
    const auto& ws = std::get<CpuStorage>(saved_inputs_[2]);
    const auto& pws = std::get<CpuStorage>(saved_inputs_[3]);
    const auto& gs = std::get<CpuStorage>(grad_out);

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
        auto* pwp = reinterpret_cast<const T*>(pws.ptr.get());
        auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
        auto* dtp = reinterpret_cast<T*>(dt_cpu.ptr.get());
        auto* dwp = reinterpret_cast<T*>(dw_cpu.ptr.get());
        auto* dpwp = reinterpret_cast<T*>(dpw_cpu.ptr.get());
        const bool elem = (reduction_ == Reduction::None);
        const T mean_scale = static_cast<T>(numel);
        for (std::size_t i = 0; i < numel; ++i) {
            const T go = elem ? gp[i] : gp[0];
            const T scale = (reduction_ == Reduction::Mean) ? (go / mean_scale) : go;
            const T x = xp[i];
            const T y = tp[i];
            const T pw = pwp[i];
            const T sigm = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
            const T log_weight = (pw - static_cast<T>(1)) * y + static_cast<T>(1);
            const T log1pexp = std::log1p(std::exp(-std::abs(x)));
            const T l = std::max(x, static_cast<T>(0)) - x * y + log_weight * log1pexp;
            const T w = wp[i];

            const T dlx = log_weight * sigm - y;
            dxp[i] = w * dlx * scale;
            const T dly = -x + (pw - static_cast<T>(1)) * log1pexp;
            dtp[i] = w * dly * scale;
            dwp[i] = l * scale;
            dpwp[i] = w * y * log1pexp * scale;
        }
    };
    if (dtype_ == Dtype::F32)
        compute(float{});
    else if (dtype_ == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("bce_with_logits backward").not_implemented("dtype not supported");
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dt_cpu)}, Storage{std::move(dw_cpu)},
            Storage{std::move(dpw_cpu)}};
}

TensorImplPtr bce_with_logits_op(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& pos_weight,
                                 int reduction) {
    return BCEWithLogitsBackward::forward(input, target, weight, pos_weight,
                                          static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(BCEWithLogitsBackward)

// ===================================================================
// Cross-entropy = LogSoftmax + NLL fused. Only `input` is in autograd.
// ===================================================================

const OpSchema CrossEntropyBackward::schema_v1{"cross_entropy_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr CrossEntropyBackward::forward(const TensorImplPtr& input,
                                            const TensorImplPtr& target,
                                            const TensorImplPtr& weight_or_null,
                                            Reduction reduction,
                                            double eps,
                                            int ignore_index) {
    if (!input || !target)
        ErrorBuilder("cross_entropy").fail("null input");
    if (input->shape_.size() < 2)
        throw ShapeMismatch(input->shape_, Shape{}, "cross_entropy: input must be (N, C, ...)");
    if (input->device_ != target->device_)
        throw DeviceMismatch(std::string(device_name(input->device_)),
                             std::string(device_name(target->device_)),
                             "cross_entropy: input/target");
    if (weight_or_null && weight_or_null->device_ != input->device_)
        throw DeviceMismatch(std::string(device_name(input->device_)),
                             std::string(device_name(weight_or_null->device_)),
                             "cross_entropy: input/weight");

    const int N = static_cast<int>(input->shape_[0]);
    const int C = static_cast<int>(input->shape_[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < input->shape_.size(); ++i)
        spatial *= static_cast<int>(input->shape_[i]);
    const std::size_t total_samples = static_cast<std::size_t>(N) * spatial;

    Shape per_sample_shape;
    per_sample_shape.push_back(static_cast<std::int64_t>(N));
    if (input->shape_.size() > 2) {
        for (std::size_t i = 2; i < input->shape_.size(); ++i)
            per_sample_shape.push_back(input->shape_[i]);
    }
    OpScopeFull scope{schema_v1.name, input->device_, input->dtype_,
                      reduced_shape(per_sample_shape, reduction)};

    // ------------------------------------------------------------------
    // GPU branch — fused log_softmax + NLL via MLX.
    // ------------------------------------------------------------------
    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto& gt = std::get<GpuStorage>(target->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype_);

        // softmax along axis=1 → use precise=true for stability.
        auto softmax = ::mlx::core::softmax(*gx.arr, std::vector<int>{1}, /*precise=*/true);

        // Build target_idx of shape [N, 1, *spatial] and dtype int64 for take_along_axis.
        auto t_idx = ::mlx::core::astype(*gt.arr, ::mlx::core::int64);
        // target shape is [N, *spatial] (no C axis). Insert axis=1.
        ::mlx::core::Shape t_shape_with_axis = gpu::to_mlx_shape(target->shape_);
        t_shape_with_axis.insert(t_shape_with_axis.begin() + 1, 1);
        t_idx = ::mlx::core::reshape(t_idx, t_shape_with_axis);

        // Replace ignore-index entries with 0 so take_along_axis gives a defined value;
        // we mask them out below.
        auto ig_arr = ::mlx::core::astype(::mlx::core::array(ignore_index), ::mlx::core::int64);
        auto ig_mask = ::mlx::core::not_equal(t_idx, ig_arr);  // [N, 1, *spatial], bool
        auto safe_t = ::mlx::core::where(
            ig_mask, t_idx, ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64));

        // Gather softmax[n, target[n, ...], ...] → shape [N, 1, *spatial].
        auto pred = ::mlx::core::take_along_axis(softmax, safe_t, /*axis=*/1);
        auto eps_arr = mlx_scalar(eps, mlx_dt);
        auto neg_log_pred =
            ::mlx::core::negative(::mlx::core::log(::mlx::core::add(pred, eps_arr)));

        // weight[target] gather → [N, 1, *spatial].
        ::mlx::core::array w_gather = mlx_scalar(1.0, mlx_dt);
        if (weight_or_null) {
            const auto& gw = std::get<GpuStorage>(weight_or_null->storage_);
            // mlx::core::take(weight[C], indices) → indices shape, here [N, 1, *spatial].
            w_gather = ::mlx::core::take(*gw.arr, safe_t);
        } else {
            w_gather = ::mlx::core::broadcast_to(w_gather, neg_log_pred.shape());
        }

        // Apply weight, then ignore mask.
        auto loss = ::mlx::core::multiply(w_gather, neg_log_pred);
        auto ig_mask_dt = ::mlx::core::astype(ig_mask, mlx_dt);
        loss = ::mlx::core::multiply(loss, ig_mask_dt);
        // Squeeze axis=1.
        auto loss_squeezed = ::mlx::core::squeeze(loss, std::vector<int>{1});

        // Valid count = sum(ig_mask). For Mean reduction we divide by it.
        auto valid_count = ::mlx::core::sum(ig_mask_dt, /*keepdims=*/false);
        // Avoid divide-by-zero.
        auto one = mlx_scalar(1.0, mlx_dt);
        auto vc_safe = ::mlx::core::maximum(valid_count, one);

        Storage out_storage;
        Shape out_shape_local = (reduction == Reduction::None) ? per_sample_shape : Shape{};
        if (reduction == Reduction::None) {
            out_storage = Storage{gpu::wrap_mlx_array(std::move(loss_squeezed), input->dtype_)};
        } else if (reduction == Reduction::Sum) {
            auto s = ::mlx::core::sum(loss_squeezed, /*keepdims=*/false);
            out_storage = Storage{gpu::wrap_mlx_array(std::move(s), input->dtype_)};
        } else {  // Mean
            auto s = ::mlx::core::sum(loss_squeezed, /*keepdims=*/false);
            auto m = ::mlx::core::divide(s, vc_safe);
            out_storage = Storage{gpu::wrap_mlx_array(std::move(m), input->dtype_)};
        }

        auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape_local,
                                                input->dtype_, input->device_, false);

        if (!GradMode::is_enabled() || !input->requires_grad_)
            return out;

        auto x_edge = detail::ensure_grad_fn(input);
        auto bwd = std::make_shared<CrossEntropyBackward>();
        bwd->input_shapes_ = {input->shape_};
        bwd->out_shape_ = out_shape_local;
        bwd->dtype_ = input->dtype_;
        bwd->device_ = input->device_;
        bwd->input_tensors_ = {input};
        bwd->saved_inputs_ = {input->storage_};
        bwd->reduction_ = reduction;
        bwd->eps_ = eps;
        bwd->ignore_index_ = ignore_index;
        bwd->orig_input_shape_ = input->shape_;
        bwd->has_weight_ = (weight_or_null != nullptr);
        // Save softmax (large but unavoidable for fused backward).
        bwd->saved_softmax_ = Storage{gpu::wrap_mlx_array(std::move(softmax), input->dtype_)};
        bwd->saved_target_ = target->storage_;
        if (weight_or_null)
            bwd->saved_weight_ = weight_or_null->storage_;
        // Save valid_count as a 1-elem GPU array.
        bwd->saved_valid_count_ = Storage{gpu::wrap_mlx_array(std::move(vc_safe), input->dtype_)};
        bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
        bwd->set_saved_versions(
            std::vector<std::int64_t>{static_cast<std::int64_t>(input->version_)});
        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
        return out;
    }

    auto softmax_buf = allocate_size(static_cast<std::size_t>(N) * C * spatial, input->dtype_);
    auto loss_buf = allocate_size(total_samples, input->dtype_);

    const auto& xs = std::get<CpuStorage>(input->storage_);
    const auto& ts = std::get<CpuStorage>(target->storage_);
    const CpuStorage* ws =
        weight_or_null ? &std::get<CpuStorage>(weight_or_null->storage_) : nullptr;

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        auto* sp = reinterpret_cast<T*>(softmax_buf.ptr.get());
        auto* lp = reinterpret_cast<T*>(loss_buf.ptr.get());
        const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                T mx = -std::numeric_limits<T>::infinity();
                for (int c = 0; c < C; ++c) {
                    const T v = xp[(n * C + c) * spatial + s];
                    if (v > mx)
                        mx = v;
                }
                T sum = T{0};
                for (int c = 0; c < C; ++c) {
                    const T e = std::exp(xp[(n * C + c) * spatial + s] - mx);
                    sp[(n * C + c) * spatial + s] = e;
                    sum += e;
                }
                const T inv = T{1} / sum;
                for (int c = 0; c < C; ++c) {
                    sp[(n * C + c) * spatial + s] *= inv;
                }
                const std::int64_t y = read_target(ts, n * spatial + s);
                if (static_cast<int>(y) == ignore_index) {
                    lp[n * spatial + s] = T{0};
                    continue;
                }
                const T pred = sp[(n * C + static_cast<int>(y)) * spatial + s];
                const T w = wp ? wp[static_cast<int>(y)] : static_cast<T>(1);
                lp[n * spatial + s] = -w * std::log(pred + static_cast<T>(eps));
            }
        }
    };
    if (input->dtype_ == Dtype::F32)
        compute(float{});
    else if (input->dtype_ == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cross_entropy").not_implemented("dtype not supported");

    std::size_t valid_count = 0;
    for (std::size_t i = 0; i < total_samples; ++i) {
        if (read_target(ts, i) != ignore_index)
            ++valid_count;
    }
    if (valid_count == 0)
        valid_count = 1;

    Shape out_shape = (reduction == Reduction::None) ? per_sample_shape : Shape{};
    Storage out_storage;
    if (input->dtype_ == Dtype::F32) {
        auto* lp = reinterpret_cast<float*>(loss_buf.ptr.get());
        if (reduction == Reduction::None) {
            auto out = allocate_size(total_samples, input->dtype_);
            std::memcpy(out.ptr.get(), lp, total_samples * dtype_size(input->dtype_));
            out_storage = Storage{std::move(out)};
        } else {
            float s = accumulate(lp, total_samples);
            if (reduction == Reduction::Mean)
                s /= static_cast<float>(valid_count);
            out_storage = Storage{make_scalar(static_cast<double>(s), input->dtype_)};
        }
    } else {
        auto* lp = reinterpret_cast<double*>(loss_buf.ptr.get());
        if (reduction == Reduction::None) {
            auto out = allocate_size(total_samples, input->dtype_);
            std::memcpy(out.ptr.get(), lp, total_samples * dtype_size(input->dtype_));
            out_storage = Storage{std::move(out)};
        } else {
            double s = accumulate(lp, total_samples);
            if (reduction == Reduction::Mean)
                s /= static_cast<double>(valid_count);
            out_storage = Storage{make_scalar(s, input->dtype_)};
        }
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype_,
                                            input->device_, false);

    if (!GradMode::is_enabled() || !input->requires_grad_)
        return out;

    auto x_edge = detail::ensure_grad_fn(input);
    auto bwd = std::make_shared<CrossEntropyBackward>();
    bwd->input_shapes_ = {input->shape_};
    bwd->out_shape_ = out_shape;
    bwd->dtype_ = input->dtype_;
    bwd->device_ = input->device_;
    bwd->input_tensors_ = {input};
    bwd->saved_inputs_ = {input->storage_};
    bwd->reduction_ = reduction;
    bwd->eps_ = eps;
    bwd->ignore_index_ = ignore_index;
    bwd->orig_input_shape_ = input->shape_;
    bwd->has_weight_ = (weight_or_null != nullptr);
    bwd->saved_softmax_ = Storage{std::move(softmax_buf)};
    bwd->saved_target_ = target->storage_;
    if (weight_or_null)
        bwd->saved_weight_ = weight_or_null->storage_;
    bwd->saved_valid_count_ = Storage{make_scalar(static_cast<double>(valid_count), input->dtype_)};
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{static_cast<std::int64_t>(input->version_)});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> CrossEntropyBackward::apply(Storage grad_out) {
    const Shape& xs = orig_input_shape_;
    const int N = static_cast<int>(xs[0]);
    const int C = static_cast<int>(xs[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < xs.size(); ++i)
        spatial *= static_cast<int>(xs[i]);

    if (device_ == Device::GPU) {
        const auto& sm = std::get<GpuStorage>(saved_softmax_);
        const auto& tgt = std::get<GpuStorage>(saved_target_);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);

        // Build target index reshaped to [N, 1, *spatial].
        ::mlx::core::Shape t_shape_with_axis = sm.arr->shape();
        t_shape_with_axis[1] = 1;
        auto t_idx = ::mlx::core::reshape(::mlx::core::astype(*tgt.arr, ::mlx::core::int64),
                                          t_shape_with_axis);
        auto ig_arr = ::mlx::core::astype(::mlx::core::array(ignore_index_), ::mlx::core::int64);
        auto ig_mask = ::mlx::core::not_equal(t_idx, ig_arr);  // [N, 1, *spatial]
        auto safe_t = ::mlx::core::where(
            ig_mask, t_idx, ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64));

        // one_hot[n, c, ...] = (c == target[n, ...]). Build via arange compare.
        auto c_range = ::mlx::core::arange(0, C, 1);
        c_range = ::mlx::core::astype(c_range, ::mlx::core::int64);
        ::mlx::core::Shape c_shape(sm.arr->shape().size(), 1);
        c_shape[1] = C;
        auto c_idx = ::mlx::core::reshape(c_range, c_shape);  // [1, C, 1, ...]
        auto onehot = ::mlx::core::astype(::mlx::core::equal(c_idx, t_idx), mlx_dt);
        // (softmax - one_hot)
        auto base = ::mlx::core::subtract(*sm.arr, onehot);

        // weight[target] gather, broadcast to [N, C, *spatial] by broadcasting [N, 1, *spatial].
        ::mlx::core::array w_gather = mlx_scalar(1.0, mlx_dt);
        if (has_weight_) {
            const auto& gw = std::get<GpuStorage>(saved_weight_);
            w_gather = ::mlx::core::take(*gw.arr, safe_t);  // [N, 1, *spatial]
        } else {
            w_gather = ::mlx::core::broadcast_to(w_gather, t_shape_with_axis);
        }
        // Mask out ignored positions in weight.
        auto ig_mask_dt = ::mlx::core::astype(ig_mask, mlx_dt);
        w_gather = ::mlx::core::multiply(w_gather, ig_mask_dt);
        // Broadcast [N, 1, *spatial] to [N, C, *spatial].
        auto w_full = ::mlx::core::broadcast_to(w_gather, sm.arr->shape());

        // grad scale: per-element if reduction==None; otherwise gout / valid (Mean) or gout (Sum).
        ::mlx::core::array scaled = *gg.arr;
        if (reduction_ == Reduction::Mean) {
            const auto& vc = std::get<GpuStorage>(saved_valid_count_);
            scaled = ::mlx::core::divide(scaled, *vc.arr);
        }
        if (reduction_ != Reduction::None) {
            scaled = ::mlx::core::broadcast_to(scaled, t_shape_with_axis);
        } else {
            // None: gout is [N, *spatial]; reshape to [N, 1, *spatial].
            ::mlx::core::Shape gs_shape = scaled.shape();
            gs_shape.insert(gs_shape.begin() + 1, 1);
            scaled = ::mlx::core::reshape(scaled, gs_shape);
        }
        // Now `scaled` is [N, 1, *spatial]. Broadcast to [N, C, *spatial].
        auto scaled_full = ::mlx::core::broadcast_to(scaled, sm.arr->shape());

        auto dx = ::mlx::core::multiply(::mlx::core::multiply(base, w_full), scaled_full);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)}};
    }

    auto dx_cpu = allocate_size(static_cast<std::size_t>(N) * C * spatial, dtype_);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    const auto& sm_cpu = std::get<CpuStorage>(saved_softmax_);
    const auto& tgt = std::get<CpuStorage>(saved_target_);
    const auto& gs = std::get<CpuStorage>(grad_out);
    const CpuStorage* ws = has_weight_ ? &std::get<CpuStorage>(saved_weight_) : nullptr;

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto* sp = reinterpret_cast<const T*>(sm_cpu.ptr.get());
        auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
        const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
        const T valid =
            *reinterpret_cast<const T*>(std::get<CpuStorage>(saved_valid_count_).ptr.get());
        const bool elem = (reduction_ == Reduction::None);

        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const std::int64_t y = read_target(tgt, n * spatial + s);
                if (static_cast<int>(y) == ignore_index_)
                    continue;
                const T go = elem ? gp[n * spatial + s] : gp[0];
                const T scale = (reduction_ == Reduction::Mean) ? (go / valid) : go;
                const T w = wp ? wp[static_cast<int>(y)] : static_cast<T>(1);
                for (int c = 0; c < C; ++c) {
                    T v = sp[(n * C + c) * spatial + s];
                    if (c == static_cast<int>(y))
                        v -= static_cast<T>(1);
                    dxp[(n * C + c) * spatial + s] = w * v * scale;
                }
            }
        }
    };
    if (dtype_ == Dtype::F32)
        compute(float{});
    else if (dtype_ == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cross_entropy backward").not_implemented("dtype not supported");
    return {Storage{std::move(dx_cpu)}};
}

TensorImplPtr cross_entropy_op(const TensorImplPtr& input,
                               const TensorImplPtr& target,
                               const TensorImplPtr& weight_or_null,
                               int reduction,
                               double eps,
                               int ignore_index) {
    return CrossEntropyBackward::forward(input, target, weight_or_null,
                                         static_cast<Reduction>(reduction), eps, ignore_index);
}
LUCID_REGISTER_OP(CrossEntropyBackward)

// ===================================================================
// NLL — input is log-probabilities. Only `input` is differentiable.
// ===================================================================

const OpSchema NLLLossBackward::schema_v1{"nll_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr NLLLossBackward::forward(const TensorImplPtr& input,
                                       const TensorImplPtr& target,
                                       const TensorImplPtr& weight_or_null,
                                       Reduction reduction,
                                       int ignore_index) {
    if (!input || !target)
        ErrorBuilder("nll_loss").fail("null input");
    if (input->shape_.size() < 2)
        throw ShapeMismatch(input->shape_, Shape{}, "nll_loss: input must be (N, C, ...)");
    if (input->device_ != target->device_)
        throw DeviceMismatch(std::string(device_name(input->device_)),
                             std::string(device_name(target->device_)), "nll_loss: input/target");
    if (weight_or_null && weight_or_null->device_ != input->device_)
        throw DeviceMismatch(std::string(device_name(input->device_)),
                             std::string(device_name(weight_or_null->device_)),
                             "nll_loss: input/weight");

    const int N = static_cast<int>(input->shape_[0]);
    const int C = static_cast<int>(input->shape_[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < input->shape_.size(); ++i)
        spatial *= static_cast<int>(input->shape_[i]);
    const std::size_t total = static_cast<std::size_t>(N) * spatial;

    Shape per_sample_shape;
    per_sample_shape.push_back(static_cast<std::int64_t>(N));
    if (input->shape_.size() > 2) {
        for (std::size_t i = 2; i < input->shape_.size(); ++i)
            per_sample_shape.push_back(input->shape_[i]);
    }
    OpScopeFull scope{schema_v1.name, input->device_, input->dtype_,
                      reduced_shape(per_sample_shape, reduction)};

    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto& gt = std::get<GpuStorage>(target->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype_);

        ::mlx::core::Shape t_shape_with_axis = gpu::to_mlx_shape(target->shape_);
        t_shape_with_axis.insert(t_shape_with_axis.begin() + 1, 1);
        auto t_idx = ::mlx::core::reshape(::mlx::core::astype(*gt.arr, ::mlx::core::int64),
                                          t_shape_with_axis);
        auto ig_arr = ::mlx::core::astype(::mlx::core::array(ignore_index), ::mlx::core::int64);
        auto ig_mask = ::mlx::core::not_equal(t_idx, ig_arr);
        auto safe_t = ::mlx::core::where(
            ig_mask, t_idx, ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64));
        auto pred = ::mlx::core::take_along_axis(*gx.arr, safe_t, 1);
        auto neg = ::mlx::core::negative(pred);

        ::mlx::core::array w_gather = mlx_scalar(1.0, mlx_dt);
        if (weight_or_null) {
            const auto& gw = std::get<GpuStorage>(weight_or_null->storage_);
            w_gather = ::mlx::core::take(*gw.arr, safe_t);
        } else {
            w_gather = ::mlx::core::broadcast_to(w_gather, neg.shape());
        }
        auto loss = ::mlx::core::multiply(w_gather, neg);
        auto ig_mask_dt = ::mlx::core::astype(ig_mask, mlx_dt);
        loss = ::mlx::core::multiply(loss, ig_mask_dt);
        auto loss_squeezed = ::mlx::core::squeeze(loss, std::vector<int>{1});

        auto valid_count = ::mlx::core::sum(ig_mask_dt, /*keepdims=*/false);
        auto one = mlx_scalar(1.0, mlx_dt);
        auto vc_safe = ::mlx::core::maximum(valid_count, one);

        Storage out_storage;
        Shape out_shape_local = (reduction == Reduction::None) ? per_sample_shape : Shape{};
        if (reduction == Reduction::None) {
            out_storage = Storage{gpu::wrap_mlx_array(std::move(loss_squeezed), input->dtype_)};
        } else if (reduction == Reduction::Sum) {
            auto s = ::mlx::core::sum(loss_squeezed, /*keepdims=*/false);
            out_storage = Storage{gpu::wrap_mlx_array(std::move(s), input->dtype_)};
        } else {
            auto s = ::mlx::core::sum(loss_squeezed, /*keepdims=*/false);
            auto m = ::mlx::core::divide(s, vc_safe);
            out_storage = Storage{gpu::wrap_mlx_array(std::move(m), input->dtype_)};
        }
        auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape_local,
                                                input->dtype_, input->device_, false);

        if (!GradMode::is_enabled() || !input->requires_grad_)
            return out;

        auto x_edge = detail::ensure_grad_fn(input);
        auto bwd = std::make_shared<NLLLossBackward>();
        bwd->input_shapes_ = {input->shape_};
        bwd->out_shape_ = out_shape_local;
        bwd->dtype_ = input->dtype_;
        bwd->device_ = input->device_;
        bwd->input_tensors_ = {input};
        bwd->saved_inputs_ = {input->storage_};
        bwd->reduction_ = reduction;
        bwd->ignore_index_ = ignore_index;
        bwd->orig_input_shape_ = input->shape_;
        bwd->has_weight_ = (weight_or_null != nullptr);
        bwd->saved_target_ = target->storage_;
        if (weight_or_null)
            bwd->saved_weight_ = weight_or_null->storage_;
        bwd->saved_valid_count_ = Storage{gpu::wrap_mlx_array(std::move(vc_safe), input->dtype_)};
        bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
        bwd->set_saved_versions(
            std::vector<std::int64_t>{static_cast<std::int64_t>(input->version_)});
        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
        return out;
    }

    auto loss_buf = allocate_size(total, input->dtype_);
    const auto& xs = std::get<CpuStorage>(input->storage_);
    const auto& ts = std::get<CpuStorage>(target->storage_);
    const CpuStorage* ws =
        weight_or_null ? &std::get<CpuStorage>(weight_or_null->storage_) : nullptr;

    std::size_t valid_count = 0;
    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        auto* lp = reinterpret_cast<T*>(loss_buf.ptr.get());
        const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const std::int64_t y = read_target(ts, n * spatial + s);
                if (static_cast<int>(y) == ignore_index) {
                    lp[n * spatial + s] = T{0};
                } else {
                    const T w = wp ? wp[static_cast<int>(y)] : static_cast<T>(1);
                    lp[n * spatial + s] = -w * xp[(n * C + static_cast<int>(y)) * spatial + s];
                    ++valid_count;
                }
            }
        }
    };
    if (input->dtype_ == Dtype::F32)
        compute(float{});
    else if (input->dtype_ == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("nll_loss").not_implemented("dtype not supported");
    if (valid_count == 0)
        valid_count = 1;

    Shape out_shape = (reduction == Reduction::None) ? per_sample_shape : Shape{};
    Storage out_storage;
    if (input->dtype_ == Dtype::F32) {
        auto* lp = reinterpret_cast<float*>(loss_buf.ptr.get());
        if (reduction == Reduction::None) {
            auto out = allocate_size(total, input->dtype_);
            std::memcpy(out.ptr.get(), lp, total * dtype_size(input->dtype_));
            out_storage = Storage{std::move(out)};
        } else {
            float s = accumulate(lp, total);
            if (reduction == Reduction::Mean)
                s /= static_cast<float>(valid_count);
            out_storage = Storage{make_scalar(static_cast<double>(s), input->dtype_)};
        }
    } else {
        auto* lp = reinterpret_cast<double*>(loss_buf.ptr.get());
        if (reduction == Reduction::None) {
            auto out = allocate_size(total, input->dtype_);
            std::memcpy(out.ptr.get(), lp, total * dtype_size(input->dtype_));
            out_storage = Storage{std::move(out)};
        } else {
            double s = accumulate(lp, total);
            if (reduction == Reduction::Mean)
                s /= static_cast<double>(valid_count);
            out_storage = Storage{make_scalar(s, input->dtype_)};
        }
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype_,
                                            input->device_, false);

    if (!GradMode::is_enabled() || !input->requires_grad_)
        return out;

    auto x_edge = detail::ensure_grad_fn(input);
    auto bwd = std::make_shared<NLLLossBackward>();
    bwd->input_shapes_ = {input->shape_};
    bwd->out_shape_ = out_shape;
    bwd->dtype_ = input->dtype_;
    bwd->device_ = input->device_;
    bwd->input_tensors_ = {input};
    bwd->saved_inputs_ = {input->storage_};
    bwd->reduction_ = reduction;
    bwd->ignore_index_ = ignore_index;
    bwd->orig_input_shape_ = input->shape_;
    bwd->has_weight_ = (weight_or_null != nullptr);
    bwd->saved_target_ = target->storage_;
    if (weight_or_null)
        bwd->saved_weight_ = weight_or_null->storage_;
    bwd->saved_valid_count_ = Storage{make_scalar(static_cast<double>(valid_count), input->dtype_)};
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{static_cast<std::int64_t>(input->version_)});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> NLLLossBackward::apply(Storage grad_out) {
    const Shape& xs = orig_input_shape_;
    const int N = static_cast<int>(xs[0]);
    const int C = static_cast<int>(xs[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < xs.size(); ++i)
        spatial *= static_cast<int>(xs[i]);

    if (device_ == Device::GPU) {
        const auto& tgt = std::get<GpuStorage>(saved_target_);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);

        ::mlx::core::Shape full_shape = gpu::to_mlx_shape(xs);
        ::mlx::core::Shape t_shape_with_axis = full_shape;
        t_shape_with_axis[1] = 1;
        auto t_idx = ::mlx::core::reshape(::mlx::core::astype(*tgt.arr, ::mlx::core::int64),
                                          t_shape_with_axis);
        auto ig_arr = ::mlx::core::astype(::mlx::core::array(ignore_index_), ::mlx::core::int64);
        auto ig_mask = ::mlx::core::not_equal(t_idx, ig_arr);
        auto safe_t = ::mlx::core::where(
            ig_mask, t_idx, ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64));

        // one_hot(target) along axis=1, full shape.
        auto c_range = ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int64);
        ::mlx::core::Shape c_shape(full_shape.size(), 1);
        c_shape[1] = C;
        auto c_idx = ::mlx::core::reshape(c_range, c_shape);
        auto onehot = ::mlx::core::astype(::mlx::core::equal(c_idx, t_idx), mlx_dt);
        auto neg_onehot = ::mlx::core::negative(onehot);

        ::mlx::core::array w_gather = mlx_scalar(1.0, mlx_dt);
        if (has_weight_) {
            const auto& gw = std::get<GpuStorage>(saved_weight_);
            w_gather = ::mlx::core::take(*gw.arr, safe_t);
        } else {
            w_gather = ::mlx::core::broadcast_to(w_gather, t_shape_with_axis);
        }
        auto ig_mask_dt = ::mlx::core::astype(ig_mask, mlx_dt);
        w_gather = ::mlx::core::multiply(w_gather, ig_mask_dt);
        auto w_full = ::mlx::core::broadcast_to(w_gather, full_shape);

        ::mlx::core::array scaled = *gg.arr;
        if (reduction_ == Reduction::Mean) {
            const auto& vc = std::get<GpuStorage>(saved_valid_count_);
            scaled = ::mlx::core::divide(scaled, *vc.arr);
        }
        if (reduction_ != Reduction::None) {
            scaled = ::mlx::core::broadcast_to(scaled, t_shape_with_axis);
        } else {
            ::mlx::core::Shape gs_shape = scaled.shape();
            gs_shape.insert(gs_shape.begin() + 1, 1);
            scaled = ::mlx::core::reshape(scaled, gs_shape);
        }
        auto scaled_full = ::mlx::core::broadcast_to(scaled, full_shape);
        auto dx = ::mlx::core::multiply(::mlx::core::multiply(neg_onehot, w_full), scaled_full);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)}};
    }

    auto dx_cpu = allocate_size(static_cast<std::size_t>(N) * C * spatial, dtype_);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    const auto& tgt = std::get<CpuStorage>(saved_target_);
    const auto& gs = std::get<CpuStorage>(grad_out);
    const CpuStorage* ws = has_weight_ ? &std::get<CpuStorage>(saved_weight_) : nullptr;

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
        const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
        const T valid =
            *reinterpret_cast<const T*>(std::get<CpuStorage>(saved_valid_count_).ptr.get());
        const bool elem = (reduction_ == Reduction::None);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const std::int64_t y = read_target(tgt, n * spatial + s);
                if (static_cast<int>(y) == ignore_index_)
                    continue;
                const T go = elem ? gp[n * spatial + s] : gp[0];
                const T scale = (reduction_ == Reduction::Mean) ? (go / valid) : go;
                const T w = wp ? wp[static_cast<int>(y)] : static_cast<T>(1);
                dxp[(n * C + static_cast<int>(y)) * spatial + s] = -w * scale;
            }
        }
    };
    if (dtype_ == Dtype::F32)
        compute(float{});
    else if (dtype_ == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("nll_loss backward").not_implemented("dtype not supported");
    return {Storage{std::move(dx_cpu)}};
}

TensorImplPtr nll_loss_op(const TensorImplPtr& input,
                          const TensorImplPtr& target,
                          const TensorImplPtr& weight_or_null,
                          int reduction,
                          int ignore_index) {
    return NLLLossBackward::forward(input, target, weight_or_null,
                                    static_cast<Reduction>(reduction), ignore_index);
}
LUCID_REGISTER_OP(NLLLossBackward)

// ===================================================================
// Huber loss
// ===================================================================

const OpSchema HuberLossBackward::schema_v1{"huber_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr HuberLossBackward::forward(const TensorImplPtr& input,
                                         const TensorImplPtr& target,
                                         double delta,
                                         Reduction reduction) {
    if (!input || !target)
        ErrorBuilder("huber_loss").fail("null input");
    if (input->shape_ != target->shape_)
        throw ShapeMismatch(input->shape_, target->shape_,
                            "huber_loss: input/target shape mismatch");
    if (delta <= 0.0)
        ErrorBuilder("huber_loss").fail("delta must be positive");

    const std::size_t numel = input->numel();
    OpScopeFull scope{schema_v1.name, input->device_, input->dtype_,
                      reduced_shape(input->shape_, reduction)};

    Storage out_storage;
    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto& gt = std::get<GpuStorage>(target->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype_);
        auto d = mlx_scalar(delta, mlx_dt);
        auto half_d_sq = mlx_scalar(0.5 * delta * delta, mlx_dt);
        auto half = mlx_scalar(0.5, mlx_dt);
        auto r = ::mlx::core::subtract(*gx.arr, *gt.arr);
        auto ar = ::mlx::core::abs(r);
        auto sq_term = ::mlx::core::multiply(half, ::mlx::core::multiply(r, r));
        auto lin_term = ::mlx::core::subtract(::mlx::core::multiply(d, ar), half_d_sq);
        auto cond = ::mlx::core::less(ar, d);
        // less(ar, d) is true when |r| < d. We want quadratic when |r| <= d,
        // and linear otherwise. The boundary point matches both branches, so
        // we use less here for safety with NaN propagation.
        auto l = ::mlx::core::where(cond, sq_term, lin_term);
        auto red = mlx_apply_reduction(l, reduction);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(red), input->dtype_)};
    } else {
        auto loss_buf = allocate_size(numel, input->dtype_);
        const auto& xs = std::get<CpuStorage>(input->storage_);
        const auto& ts = std::get<CpuStorage>(target->storage_);

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            auto* lp = reinterpret_cast<T*>(loss_buf.ptr.get());
            const T d = static_cast<T>(delta);
            for (std::size_t i = 0; i < numel; ++i) {
                const T r = xp[i] - tp[i];
                const T ar = std::abs(r);
                lp[i] = (ar <= d) ? T{0.5} * r * r : d * (ar - T{0.5} * d);
            }
        };
        if (input->dtype_ == Dtype::F32)
            compute(float{});
        else if (input->dtype_ == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("huber_loss").not_implemented("dtype not supported");

        if (input->dtype_ == Dtype::F32) {
            out_storage = apply_reduction<float>(reinterpret_cast<float*>(loss_buf.ptr.get()),
                                                 numel, reduction, input->dtype_);
        } else {
            out_storage = apply_reduction<double>(reinterpret_cast<double*>(loss_buf.ptr.get()),
                                                  numel, reduction, input->dtype_);
        }
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape_, reduction), input->dtype_,
                                            input->device_, false);

    if (!GradMode::is_enabled() || !(input->requires_grad_ || target->requires_grad_))
        return out;

    auto x_edge = detail::ensure_grad_fn(input);
    auto t_edge = detail::ensure_grad_fn(target);
    auto bwd = std::make_shared<HuberLossBackward>();
    bwd->input_shapes_ = {input->shape_, target->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = input->dtype_;
    bwd->device_ = input->device_;
    bwd->input_tensors_ = {input, target};
    bwd->saved_inputs_ = {input->storage_, target->storage_};
    bwd->reduction_ = reduction;
    bwd->delta_ = delta;
    bwd->orig_shape_ = input->shape_;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(t_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{static_cast<std::int64_t>(input->version_),
                                                      static_cast<std::int64_t>(target->version_)});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> HuberLossBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(orig_shape_);

    if (device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gt = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);
        auto d = mlx_scalar(delta_, mlx_dt);
        auto neg_d = mlx_scalar(-delta_, mlx_dt);
        auto r = ::mlx::core::subtract(*gx.arr, *gt.arr);
        auto ar = ::mlx::core::abs(r);
        auto cond = ::mlx::core::less(ar, d);
        auto sgn_d = ::mlx::core::where(::mlx::core::greater(r, mlx_scalar(0.0, mlx_dt)), d, neg_d);
        auto dr = ::mlx::core::where(cond, r, sgn_d);

        auto scaled = mlx_grad_scale(*gg.arr, reduction_, numel, mlx_dt);
        if (reduction_ != Reduction::None)
            scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(orig_shape_));
        auto dx = ::mlx::core::multiply(dr, scaled);
        auto dt = ::mlx::core::negative(dx);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dt), dtype_)}};
    }

    auto dx_cpu = allocate_size(numel, dtype_);
    auto dt_cpu = allocate_size(numel, dtype_);
    const auto& xs = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& ts = std::get<CpuStorage>(saved_inputs_[1]);
    const auto& gs = std::get<CpuStorage>(grad_out);

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
        auto* dtp = reinterpret_cast<T*>(dt_cpu.ptr.get());
        const T d = static_cast<T>(delta_);
        const bool elem = (reduction_ == Reduction::None);
        const T mean_scale = static_cast<T>(numel);
        for (std::size_t i = 0; i < numel; ++i) {
            const T go = elem ? gp[i] : gp[0];
            const T scale = (reduction_ == Reduction::Mean) ? (go / mean_scale) : go;
            const T r = xp[i] - tp[i];
            T dr;
            if (std::abs(r) <= d)
                dr = r;
            else
                dr = (r > T{0}) ? d : -d;
            dxp[i] = dr * scale;
            dtp[i] = -dxp[i];
        }
    };
    if (dtype_ == Dtype::F32)
        compute(float{});
    else if (dtype_ == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("huber_loss backward").not_implemented("dtype not supported");
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dt_cpu)}};
}

TensorImplPtr huber_loss_op(const TensorImplPtr& input,
                            const TensorImplPtr& target,
                            double delta,
                            int reduction) {
    return HuberLossBackward::forward(input, target, delta, static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(HuberLossBackward)

}  // namespace lucid
