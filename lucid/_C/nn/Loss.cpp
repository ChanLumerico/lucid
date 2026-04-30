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
#include "../backend/Dispatcher.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

namespace {

using gpu::mlx_scalar;

// ---------------- GPU helpers (shared across loss kernels) ----------------
//
// Stable BCE-with-logits primitive on MLX:
//   loss = max(x, 0) - x*y + log_weight * log1p(exp(-|x|))
// where log_weight = (pos_weight - 1) * y + 1.

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
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "mse_loss: input/target shape mismatch");
    if (input->dtype() != target->dtype())
        throw DtypeMismatch(std::string(dtype_name(input->dtype())),
                            std::string(dtype_name(target->dtype())), "mse_loss");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};

    Storage out_storage =
        backend::Dispatcher::for_device(input->device())
            .mse_loss(input->storage(), target->storage(), input->shape(), input->dtype(),
                      static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<MseLossBackward>();
        bwd->reduction_ = reduction;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<MseLossBackward, 2>::wire_autograd(std::move(bwd), {input, target}, out);
    }
    return out;
}

std::vector<Storage> MseLossBackward::apply(Storage grad_out) {
    auto grads = backend::Dispatcher::for_device(device_)
                     .mse_loss_backward(saved_inputs_[0], saved_inputs_[1], grad_out,
                                        orig_shape_, dtype_, static_cast<int>(reduction_));
    return {std::move(grads.first), std::move(grads.second)};
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
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "bce_loss: input/target shape mismatch");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};

    Storage out_storage =
        backend::Dispatcher::for_device(input->device())
            .bce_loss(input->storage(), target->storage(), weight->storage(), input->shape(),
                      input->dtype(), eps, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<BCELossBackward>();
        bwd->reduction_ = reduction;
        bwd->eps_ = eps;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<BCELossBackward, 3>::wire_autograd(std::move(bwd),
                                                              {input, target, weight}, out);
    }
    return out;
}

std::vector<Storage> BCELossBackward::apply(Storage grad_out) {
    return backend::Dispatcher::for_device(device_)
        .bce_loss_backward(saved_inputs_[0], saved_inputs_[1], saved_inputs_[2], grad_out,
                           orig_shape_, dtype_, eps_, static_cast<int>(reduction_));
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
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "bce_with_logits: input/target shape mismatch");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};

    Storage out_storage =
        backend::Dispatcher::for_device(input->device())
            .bce_with_logits_loss(input->storage(), target->storage(), weight->storage(),
                                  pos_weight->storage(), input->shape(), weight->shape(),
                                  pos_weight->shape(), input->dtype(), static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<BCEWithLogitsBackward>();
        bwd->reduction_ = reduction;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<BCEWithLogitsBackward, 4>::wire_autograd(
            std::move(bwd), {input, target, weight, pos_weight}, out);
    }
    return out;
}

std::vector<Storage> BCEWithLogitsBackward::apply(Storage grad_out) {
    return backend::Dispatcher::for_device(device_)
        .bce_with_logits_backward(saved_inputs_[0], saved_inputs_[1], saved_inputs_[2],
                                  saved_inputs_[3], grad_out, orig_shape_, dtype_,
                                  static_cast<int>(reduction_));
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
    if (input->shape().size() < 2)
        throw ShapeMismatch(input->shape(), Shape{}, "cross_entropy: input must be (N, C, ...)");
    if (input->device() != target->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(target->device())),
                             "cross_entropy: input/target");
    if (weight_or_null && weight_or_null->device() != input->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(weight_or_null->device())),
                             "cross_entropy: input/weight");

    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < input->shape().size(); ++i)
        spatial *= static_cast<int>(input->shape()[i]);
    const std::size_t total_samples = static_cast<std::size_t>(N) * spatial;

    Shape per_sample_shape;
    per_sample_shape.push_back(static_cast<std::int64_t>(N));
    if (input->shape().size() > 2) {
        for (std::size_t i = 2; i < input->shape().size(); ++i)
            per_sample_shape.push_back(input->shape()[i]);
    }
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(per_sample_shape, reduction)};

    // ------------------------------------------------------------------
    // GPU branch — fused log_softmax + NLL via MLX.
    // ------------------------------------------------------------------
    if (input->device() == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage());
        const auto& gt = std::get<GpuStorage>(target->storage());
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype());

        // softmax along axis=1 → use precise=true for stability.
        auto softmax = ::mlx::core::softmax(*gx.arr, std::vector<int>{1}, /*precise=*/true);

        // Build target_idx of shape [N, 1, *spatial] and dtype int64 for take_along_axis.
        auto t_idx = ::mlx::core::astype(*gt.arr, ::mlx::core::int64);
        // target shape is [N, *spatial] (no C axis). Insert axis=1.
        ::mlx::core::Shape t_shape_with_axis = gpu::to_mlx_shape(target->shape());
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
            const auto& gw = std::get<GpuStorage>(weight_or_null->storage());
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
            out_storage = Storage{gpu::wrap_mlx_array(std::move(loss_squeezed), input->dtype())};
        } else if (reduction == Reduction::Sum) {
            auto s = ::mlx::core::sum(loss_squeezed, /*keepdims=*/false);
            out_storage = Storage{gpu::wrap_mlx_array(std::move(s), input->dtype())};
        } else {  // Mean
            auto s = ::mlx::core::sum(loss_squeezed, /*keepdims=*/false);
            auto m = ::mlx::core::divide(s, vc_safe);
            out_storage = Storage{gpu::wrap_mlx_array(std::move(m), input->dtype())};
        }

        auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape_local,
                                                input->dtype(), input->device(), false);

        {
            auto bwd = std::make_shared<CrossEntropyBackward>();
            bwd->reduction_ = reduction;
            bwd->eps_ = eps;
            bwd->ignore_index_ = ignore_index;
            bwd->orig_input_shape_ = input->shape();
            bwd->has_weight_ = (weight_or_null != nullptr);
            // Save softmax (large but unavoidable for fused backward).
            bwd->saved_softmax_ = Storage{gpu::wrap_mlx_array(std::move(softmax), input->dtype())};
            bwd->saved_target_ = target->storage();
            if (weight_or_null)
                bwd->saved_weight_ = weight_or_null->storage();
            // Save valid_count as a 1-elem GPU array.
            bwd->saved_valid_count_ =
                Storage{gpu::wrap_mlx_array(std::move(vc_safe), input->dtype())};
            kernel::NaryKernel<CrossEntropyBackward, 1>::wire_autograd(std::move(bwd), {input}, out,
                                                                       /*save_ins=*/false);
        }
        return out;
    }

    auto softmax_buf = allocate_size(static_cast<std::size_t>(N) * C * spatial, input->dtype());
    auto loss_buf = allocate_size(total_samples, input->dtype());

    const auto& xs = std::get<CpuStorage>(input->storage());
    const auto& ts = std::get<CpuStorage>(target->storage());
    const CpuStorage* ws =
        weight_or_null ? &std::get<CpuStorage>(weight_or_null->storage()) : nullptr;

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
    if (input->dtype() == Dtype::F32)
        compute(float{});
    else if (input->dtype() == Dtype::F64)
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
    if (input->dtype() == Dtype::F32) {
        auto* lp = reinterpret_cast<float*>(loss_buf.ptr.get());
        if (reduction == Reduction::None) {
            auto out = allocate_size(total_samples, input->dtype());
            std::memcpy(out.ptr.get(), lp, total_samples * dtype_size(input->dtype()));
            out_storage = Storage{std::move(out)};
        } else {
            float s = accumulate(lp, total_samples);
            if (reduction == Reduction::Mean)
                s /= static_cast<float>(valid_count);
            out_storage = Storage{make_scalar(static_cast<double>(s), input->dtype())};
        }
    } else {
        auto* lp = reinterpret_cast<double*>(loss_buf.ptr.get());
        if (reduction == Reduction::None) {
            auto out = allocate_size(total_samples, input->dtype());
            std::memcpy(out.ptr.get(), lp, total_samples * dtype_size(input->dtype()));
            out_storage = Storage{std::move(out)};
        } else {
            double s = accumulate(lp, total_samples);
            if (reduction == Reduction::Mean)
                s /= static_cast<double>(valid_count);
            out_storage = Storage{make_scalar(s, input->dtype())};
        }
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);

    {
        auto bwd = std::make_shared<CrossEntropyBackward>();
        bwd->reduction_ = reduction;
        bwd->eps_ = eps;
        bwd->ignore_index_ = ignore_index;
        bwd->orig_input_shape_ = input->shape();
        bwd->has_weight_ = (weight_or_null != nullptr);
        bwd->saved_softmax_ = Storage{std::move(softmax_buf)};
        bwd->saved_target_ = target->storage();
        if (weight_or_null)
            bwd->saved_weight_ = weight_or_null->storage();
        bwd->saved_valid_count_ =
            Storage{make_scalar(static_cast<double>(valid_count), input->dtype())};
        kernel::NaryKernel<CrossEntropyBackward, 1>::wire_autograd(std::move(bwd), {input}, out,
                                                                   /*save_ins=*/false);
    }
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
    if (input->shape().size() < 2)
        throw ShapeMismatch(input->shape(), Shape{}, "nll_loss: input must be (N, C, ...)");
    if (input->device() != target->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(target->device())), "nll_loss: input/target");
    if (weight_or_null && weight_or_null->device() != input->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(weight_or_null->device())),
                             "nll_loss: input/weight");

    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < input->shape().size(); ++i)
        spatial *= static_cast<int>(input->shape()[i]);
    const std::size_t total = static_cast<std::size_t>(N) * spatial;

    Shape per_sample_shape;
    per_sample_shape.push_back(static_cast<std::int64_t>(N));
    if (input->shape().size() > 2) {
        for (std::size_t i = 2; i < input->shape().size(); ++i)
            per_sample_shape.push_back(input->shape()[i]);
    }
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(per_sample_shape, reduction)};

    if (input->device() == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage());
        const auto& gt = std::get<GpuStorage>(target->storage());
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype());

        ::mlx::core::Shape t_shape_with_axis = gpu::to_mlx_shape(target->shape());
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
            const auto& gw = std::get<GpuStorage>(weight_or_null->storage());
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
            out_storage = Storage{gpu::wrap_mlx_array(std::move(loss_squeezed), input->dtype())};
        } else if (reduction == Reduction::Sum) {
            auto s = ::mlx::core::sum(loss_squeezed, /*keepdims=*/false);
            out_storage = Storage{gpu::wrap_mlx_array(std::move(s), input->dtype())};
        } else {
            auto s = ::mlx::core::sum(loss_squeezed, /*keepdims=*/false);
            auto m = ::mlx::core::divide(s, vc_safe);
            out_storage = Storage{gpu::wrap_mlx_array(std::move(m), input->dtype())};
        }
        auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape_local,
                                                input->dtype(), input->device(), false);

        {
            auto bwd = std::make_shared<NLLLossBackward>();
            bwd->reduction_ = reduction;
            bwd->ignore_index_ = ignore_index;
            bwd->orig_input_shape_ = input->shape();
            bwd->has_weight_ = (weight_or_null != nullptr);
            bwd->saved_target_ = target->storage();
            if (weight_or_null)
                bwd->saved_weight_ = weight_or_null->storage();
            bwd->saved_valid_count_ =
                Storage{gpu::wrap_mlx_array(std::move(vc_safe), input->dtype())};
            kernel::NaryKernel<NLLLossBackward, 1>::wire_autograd(std::move(bwd), {input}, out,
                                                                  /*save_ins=*/false);
        }
        return out;
    }

    auto loss_buf = allocate_size(total, input->dtype());
    const auto& xs = std::get<CpuStorage>(input->storage());
    const auto& ts = std::get<CpuStorage>(target->storage());
    const CpuStorage* ws =
        weight_or_null ? &std::get<CpuStorage>(weight_or_null->storage()) : nullptr;

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
    if (input->dtype() == Dtype::F32)
        compute(float{});
    else if (input->dtype() == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("nll_loss").not_implemented("dtype not supported");
    if (valid_count == 0)
        valid_count = 1;

    Shape out_shape = (reduction == Reduction::None) ? per_sample_shape : Shape{};
    Storage out_storage;
    if (input->dtype() == Dtype::F32) {
        auto* lp = reinterpret_cast<float*>(loss_buf.ptr.get());
        if (reduction == Reduction::None) {
            auto out = allocate_size(total, input->dtype());
            std::memcpy(out.ptr.get(), lp, total * dtype_size(input->dtype()));
            out_storage = Storage{std::move(out)};
        } else {
            float s = accumulate(lp, total);
            if (reduction == Reduction::Mean)
                s /= static_cast<float>(valid_count);
            out_storage = Storage{make_scalar(static_cast<double>(s), input->dtype())};
        }
    } else {
        auto* lp = reinterpret_cast<double*>(loss_buf.ptr.get());
        if (reduction == Reduction::None) {
            auto out = allocate_size(total, input->dtype());
            std::memcpy(out.ptr.get(), lp, total * dtype_size(input->dtype()));
            out_storage = Storage{std::move(out)};
        } else {
            double s = accumulate(lp, total);
            if (reduction == Reduction::Mean)
                s /= static_cast<double>(valid_count);
            out_storage = Storage{make_scalar(s, input->dtype())};
        }
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);

    {
        auto bwd = std::make_shared<NLLLossBackward>();
        bwd->reduction_ = reduction;
        bwd->ignore_index_ = ignore_index;
        bwd->orig_input_shape_ = input->shape();
        bwd->has_weight_ = (weight_or_null != nullptr);
        bwd->saved_target_ = target->storage();
        if (weight_or_null)
            bwd->saved_weight_ = weight_or_null->storage();
        bwd->saved_valid_count_ =
            Storage{make_scalar(static_cast<double>(valid_count), input->dtype())};
        kernel::NaryKernel<NLLLossBackward, 1>::wire_autograd(std::move(bwd), {input}, out,
                                                              /*save_ins=*/false);
    }
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
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "huber_loss: input/target shape mismatch");
    if (delta <= 0.0)
        ErrorBuilder("huber_loss").fail("delta must be positive");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};

    Storage out_storage =
        backend::Dispatcher::for_device(input->device())
            .huber_loss(input->storage(), target->storage(), input->shape(), input->dtype(), delta,
                        static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<HuberLossBackward>();
        bwd->reduction_ = reduction;
        bwd->delta_ = delta;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<HuberLossBackward, 2>::wire_autograd(std::move(bwd), {input, target},
                                                                out);
    }
    return out;
}

std::vector<Storage> HuberLossBackward::apply(Storage grad_out) {
    auto grads = backend::Dispatcher::for_device(device_)
                     .huber_loss_backward(saved_inputs_[0], saved_inputs_[1], grad_out,
                                          orig_shape_, dtype_, delta_,
                                          static_cast<int>(reduction_));
    return {std::move(grads.first), std::move(grads.second)};
}

TensorImplPtr huber_loss_op(const TensorImplPtr& input,
                            const TensorImplPtr& target,
                            double delta,
                            int reduction) {
    return HuberLossBackward::forward(input, target, delta, static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(HuberLossBackward)

}  // namespace lucid
