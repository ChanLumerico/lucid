// lucid/_C/backend/cpu/methods/Loss.cpp
// Out-of-line CpuBackend method definitions for the Loss family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::ctc_loss_forward(const Storage& log_probs,
                                     const Storage& targets,
                                     const Storage& input_lengths,
                                     const Storage& target_lengths,
                                     const Shape& lp_shape,
                                     int blank,
                                     bool zero_infinity,
                                     Dtype dt) {
    const std::size_t out_nb = static_cast<std::size_t>(lp_shape[1]) * dtype_size(dt);
    auto out_ptr = allocate_aligned_bytes(out_nb, Device::CPU);
    auto lp = [dt](const CpuStorage& clp, std::size_t idx) -> double {
        if (dt == Dtype::F32)
            return static_cast<double>(reinterpret_cast<const float*>(clp.ptr.get())[idx]);
        return reinterpret_cast<const double*>(clp.ptr.get())[idx];
    };
    auto store = [&](std::size_t b, double v) {
        if (dt == Dtype::F32)
            reinterpret_cast<float*>(out_ptr.get())[b] = static_cast<float>(v);
        else
            reinterpret_cast<double*>(out_ptr.get())[b] = v;
    };
    return cpu::ctc::forward(log_probs, targets, input_lengths, target_lengths, lp_shape, blank,
                             zero_infinity, dt, CpuStorage{out_ptr, out_nb, dt}, lp, store);
}

Storage CpuBackend::ctc_loss_backward(const Storage& log_probs,
                                      const Storage& targets,
                                      const Storage& input_lengths,
                                      const Storage& target_lengths,
                                      const Storage& grad_out,
                                      const Shape& lp_shape,
                                      int blank,
                                      bool zero_infinity,
                                      Dtype dt) {
    const std::size_t nb = static_cast<std::size_t>(lp_shape[0]) *
                           static_cast<std::size_t>(lp_shape[1]) *
                           static_cast<std::size_t>(lp_shape[2]) * dtype_size(dt);
    auto out_ptr = allocate_aligned_bytes(nb, Device::CPU);
    std::memset(out_ptr.get(), 0, nb);
    auto read = [dt](const CpuStorage& s, std::size_t i) -> double {
        if (dt == Dtype::F32)
            return static_cast<double>(reinterpret_cast<const float*>(s.ptr.get())[i]);
        return reinterpret_cast<const double*>(s.ptr.get())[i];
    };
    auto put = [&](std::size_t i, double v) {
        if (dt == Dtype::F32)
            reinterpret_cast<float*>(out_ptr.get())[i] = static_cast<float>(v);
        else
            reinterpret_cast<double*>(out_ptr.get())[i] = v;
    };
    return cpu::ctc::backward(log_probs, targets, input_lengths, target_lengths, grad_out, lp_shape,
                              blank, zero_infinity, dt, CpuStorage{out_ptr, nb, dt}, read, put);
}

Storage CpuBackend::mse_loss(
    const Storage& input, const Storage& target, const Shape& shape, Dtype dt, int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            mse_loss(detail::as_f32(input), detail::as_f32(target), shape, Dtype::F32, reduction),
            dt);
    const std::size_t n = shape_numel(shape);
    if (reduction == 0) {
        const std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        compute_mse_loss_values(input, target, ptr.get(), n, dt);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    auto scalar = allocate_aligned_bytes(dtype_size(dt), Device::CPU);
    if (dt == Dtype::F32) {
        float sum = mse_loss_sum<float>(input, target, n);
        if (reduction == 1)
            sum /= static_cast<float>(n);
        *reinterpret_cast<float*>(scalar.get()) = sum;
    } else if (dt == Dtype::F64) {
        double sum = mse_loss_sum<double>(input, target, n);
        if (reduction == 1)
            sum /= static_cast<double>(n);
        *reinterpret_cast<double*>(scalar.get()) = sum;
    } else {
        ErrorBuilder("cpu_backend::mse_loss").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{scalar, dtype_size(dt), dt}};
}

std::pair<Storage, Storage> CpuBackend::mse_loss_backward(const Storage& input,
                                                          const Storage& target,
                                                          const Storage& grad,
                                                          const Shape& shape,
                                                          Dtype dt,
                                                          int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(mse_loss_backward(detail::as_f32(input), detail::as_f32(target),
                                                     detail::as_f32(grad), shape, Dtype::F32,
                                                     reduction),
                                   dt);
    const std::size_t n = shape_numel(shape);
    const std::size_t nb = n * dtype_size(dt);
    auto dx = allocate_aligned_bytes(nb, Device::CPU);
    auto dtarget = allocate_aligned_bytes(nb, Device::CPU);
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& gs = std::get<CpuStorage>(grad);

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx.get());
        auto* dtp = reinterpret_cast<T*>(dtarget.get());
        const bool elem = reduction == 0;
        const T mean_scale = static_cast<T>(n);
        for (std::size_t i = 0; i < n; ++i) {
            const T go = elem ? gp[i] : gp[0];
            const T scale = (reduction == 1) ? (go / mean_scale) : go;
            const T d = xp[i] - tp[i];
            dxp[i] = static_cast<T>(2) * d * scale;
            dtp[i] = -dxp[i];
        }
    };

    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::mse_loss_backward").not_implemented("dtype not supported");

    return {Storage{CpuStorage{dx, nb, dt}}, Storage{CpuStorage{dtarget, nb, dt}}};
}

Storage CpuBackend::huber_loss(const Storage& input,
                               const Storage& target,
                               const Shape& shape,
                               Dtype dt,
                               double delta,
                               int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(huber_loss(detail::as_f32(input), detail::as_f32(target), shape,
                                              Dtype::F32, delta, reduction),
                                   dt);
    const std::size_t n = shape_numel(shape);
    if (reduction == 0) {
        const std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        compute_huber_loss_values(input, target, ptr.get(), n, dt, delta);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    auto scalar = allocate_aligned_bytes(dtype_size(dt), Device::CPU);
    if (dt == Dtype::F32) {
        float sum = huber_loss_sum<float>(input, target, n, static_cast<float>(delta));
        if (reduction == 1)
            sum /= static_cast<float>(n);
        *reinterpret_cast<float*>(scalar.get()) = sum;
    } else if (dt == Dtype::F64) {
        double sum = huber_loss_sum<double>(input, target, n, delta);
        if (reduction == 1)
            sum /= static_cast<double>(n);
        *reinterpret_cast<double*>(scalar.get()) = sum;
    } else {
        ErrorBuilder("cpu_backend::huber_loss").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{scalar, dtype_size(dt), dt}};
}

std::pair<Storage, Storage> CpuBackend::huber_loss_backward(const Storage& input,
                                                            const Storage& target,
                                                            const Storage& grad,
                                                            const Shape& shape,
                                                            Dtype dt,
                                                            double delta,
                                                            int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(huber_loss_backward(detail::as_f32(input),
                                                       detail::as_f32(target), detail::as_f32(grad),
                                                       shape, Dtype::F32, delta, reduction),
                                   dt);
    const std::size_t n = shape_numel(shape);
    const std::size_t nb = n * dtype_size(dt);
    auto dx = allocate_aligned_bytes(nb, Device::CPU);
    auto dtarget = allocate_aligned_bytes(nb, Device::CPU);
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& gs = std::get<CpuStorage>(grad);

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx.get());
        auto* dtp = reinterpret_cast<T*>(dtarget.get());
        const T d = static_cast<T>(delta);
        const bool elem = reduction == 0;
        const T mean_scale = static_cast<T>(n);
        for (std::size_t i = 0; i < n; ++i) {
            const T go = elem ? gp[i] : gp[0];
            const T scale = (reduction == 1) ? (go / mean_scale) : go;
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

    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::huber_loss_backward").not_implemented("dtype not supported");

    return {Storage{CpuStorage{dx, nb, dt}}, Storage{CpuStorage{dtarget, nb, dt}}};
}

Storage CpuBackend::bce_loss(const Storage& input,
                             const Storage& target,
                             const Storage& weight,
                             const Shape& shape,
                             Dtype dt,
                             double eps,
                             int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(bce_loss(detail::as_f32(input), detail::as_f32(target),
                                            detail::as_f32(weight), shape, Dtype::F32, eps,
                                            reduction),
                                   dt);
    const std::size_t n = shape_numel(shape);
    if (reduction == 0) {
        const std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        compute_bce_loss_values(input, target, weight, ptr.get(), n, dt, eps);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    auto scalar = allocate_aligned_bytes(dtype_size(dt), Device::CPU);
    if (dt == Dtype::F32) {
        float sum = bce_loss_sum<float>(input, target, weight, n, static_cast<float>(eps));
        if (reduction == 1)
            sum /= static_cast<float>(n);
        *reinterpret_cast<float*>(scalar.get()) = sum;
    } else if (dt == Dtype::F64) {
        double sum = bce_loss_sum<double>(input, target, weight, n, eps);
        if (reduction == 1)
            sum /= static_cast<double>(n);
        *reinterpret_cast<double*>(scalar.get()) = sum;
    } else {
        ErrorBuilder("cpu_backend::bce_loss").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{scalar, dtype_size(dt), dt}};
}

std::vector<Storage> CpuBackend::bce_loss_backward(const Storage& input,
                                                   const Storage& target,
                                                   const Storage& weight,
                                                   const Storage& grad,
                                                   const Shape& shape,
                                                   Dtype dt,
                                                   double eps,
                                                   int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(bce_loss_backward(detail::as_f32(input), detail::as_f32(target),
                                                     detail::as_f32(weight), detail::as_f32(grad),
                                                     shape, Dtype::F32, eps, reduction),
                                   dt);
    const std::size_t n = shape_numel(shape);
    const std::size_t nb = n * dtype_size(dt);
    auto dx = allocate_aligned_bytes(nb, Device::CPU);
    auto dtarget = allocate_aligned_bytes(nb, Device::CPU);
    auto dweight = allocate_aligned_bytes(nb, Device::CPU);
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& ws = std::get<CpuStorage>(weight);
    const auto& gs = std::get<CpuStorage>(grad);

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
        const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx.get());
        auto* dtp = reinterpret_cast<T*>(dtarget.get());
        auto* dwp = reinterpret_cast<T*>(dweight.get());
        const T e = static_cast<T>(eps);
        const bool elem = reduction == 0;
        const T mean_scale = static_cast<T>(n);
        for (std::size_t i = 0; i < n; ++i) {
            const T go = elem ? gp[i] : gp[0];
            const T scale = (reduction == 1) ? (go / mean_scale) : go;
            const T p = std::min(std::max(xp[i], e), static_cast<T>(1) - e);
            const T y = tp[i];
            const T w = wp[i];
            const T one = static_cast<T>(1);
            const T dlp = -y / p + (one - y) / (one - p);
            dxp[i] = w * dlp * scale;
            const T dly = -std::log(p) + std::log(one - p);
            dtp[i] = w * dly * scale;
            const T l = -(y * std::log(p) + (one - y) * std::log(one - p));
            dwp[i] = l * scale;
        }
    };

    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::bce_loss_backward").not_implemented("dtype not supported");

    return {Storage{CpuStorage{dx, nb, dt}}, Storage{CpuStorage{dtarget, nb, dt}},
            Storage{CpuStorage{dweight, nb, dt}}};
}

Storage CpuBackend::bce_with_logits_loss(const Storage& input,
                                         const Storage& target,
                                         const Storage& weight,
                                         const Storage& pos_weight,
                                         const Shape& shape,
                                         const Shape& weight_shape,
                                         const Shape& pos_weight_shape,
                                         Dtype dt,
                                         int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            bce_with_logits_loss(detail::as_f32(input), detail::as_f32(target),
                                 detail::as_f32(weight), detail::as_f32(pos_weight), shape,
                                 weight_shape, pos_weight_shape, Dtype::F32, reduction),
            dt);
    const std::size_t n = shape_numel(shape);
    Storage weight_storage =
        weight_shape == shape ? weight : broadcast(weight, weight_shape, shape, dt);
    Storage pos_weight_storage =
        pos_weight_shape == shape ? pos_weight : broadcast(pos_weight, pos_weight_shape, shape, dt);
    if (reduction == 0) {
        const std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        compute_bce_logits_values(input, target, weight_storage, pos_weight_storage, ptr.get(), n,
                                  dt);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    auto scalar = allocate_aligned_bytes(dtype_size(dt), Device::CPU);
    if (dt == Dtype::F32) {
        float sum = bce_logits_sum<float>(input, target, weight_storage, pos_weight_storage, n);
        if (reduction == 1)
            sum /= static_cast<float>(n);
        *reinterpret_cast<float*>(scalar.get()) = sum;
    } else if (dt == Dtype::F64) {
        double sum = bce_logits_sum<double>(input, target, weight_storage, pos_weight_storage, n);
        if (reduction == 1)
            sum /= static_cast<double>(n);
        *reinterpret_cast<double*>(scalar.get()) = sum;
    } else {
        ErrorBuilder("cpu_backend::bce_with_logits_loss").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{scalar, dtype_size(dt), dt}};
}

std::vector<Storage> CpuBackend::bce_with_logits_backward(const Storage& input,
                                                          const Storage& target,
                                                          const Storage& weight,
                                                          const Storage& pos_weight,
                                                          const Storage& grad,
                                                          const Shape& shape,
                                                          Dtype dt,
                                                          int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            bce_with_logits_backward(detail::as_f32(input), detail::as_f32(target),
                                     detail::as_f32(weight), detail::as_f32(pos_weight),
                                     detail::as_f32(grad), shape, Dtype::F32, reduction),
            dt);
    const std::size_t n = shape_numel(shape);
    const std::size_t nb = n * dtype_size(dt);
    auto dx = allocate_aligned_bytes(nb, Device::CPU);
    auto dtarget = allocate_aligned_bytes(nb, Device::CPU);
    auto dweight = allocate_aligned_bytes(nb, Device::CPU);
    auto dpos_weight = allocate_aligned_bytes(nb, Device::CPU);
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& ws = std::get<CpuStorage>(weight);
    const auto& pws = std::get<CpuStorage>(pos_weight);
    const auto& gs = std::get<CpuStorage>(grad);

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
        const auto* pwp = reinterpret_cast<const T*>(pws.ptr.get());
        const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx.get());
        auto* dtp = reinterpret_cast<T*>(dtarget.get());
        auto* dwp = reinterpret_cast<T*>(dweight.get());
        auto* dpwp = reinterpret_cast<T*>(dpos_weight.get());
        const bool elem = reduction == 0;
        const T mean_scale = static_cast<T>(n);
        const T one = static_cast<T>(1);
        for (std::size_t i = 0; i < n; ++i) {
            const T go = elem ? gp[i] : gp[0];
            const T scale = (reduction == 1) ? (go / mean_scale) : go;
            const T x = xp[i];
            const T y = tp[i];
            const T pw = pwp[i];
            const T sigm = one / (one + std::exp(-x));
            const T log_weight = (pw - one) * y + one;
            const T log1pexp = std::log1p(std::exp(-std::abs(x)));
            const T loss = std::max(x, T{0}) - x * y + log_weight * log1pexp;
            const T w = wp[i];
            dxp[i] = w * (log_weight * sigm - y) * scale;
            dtp[i] = w * (-x + (pw - one) * log1pexp) * scale;
            dwp[i] = loss * scale;
            dpwp[i] = w * y * log1pexp * scale;
        }
    };

    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::bce_with_logits_backward")
            .not_implemented("dtype not supported");

    return {Storage{CpuStorage{dx, nb, dt}}, Storage{CpuStorage{dtarget, nb, dt}},
            Storage{CpuStorage{dweight, nb, dt}}, Storage{CpuStorage{dpos_weight, nb, dt}}};
}

ClassLossForwardResult CpuBackend::cross_entropy_loss(const Storage& input,
                                                      const Storage& target,
                                                      const Storage* weight,
                                                      const Shape& input_shape,
                                                      const Shape&,
                                                      Dtype dt,
                                                      double eps,
                                                      int ignore_index,
                                                      int reduction) {
    const int n_batch = static_cast<int>(input_shape[0]);
    const int channels = static_cast<int>(input_shape[1]);
    const int spatial = class_loss_spatial(input_shape);
    const std::size_t samples = static_cast<std::size_t>(n_batch) * spatial;
    const std::size_t input_numel = samples * static_cast<std::size_t>(channels);
    CpuStorage softmax{allocate_aligned_bytes(input_numel * dtype_size(dt), Device::CPU),
                       input_numel * dtype_size(dt), dt};
    CpuStorage losses{allocate_aligned_bytes(samples * dtype_size(dt), Device::CPU),
                      samples * dtype_size(dt), dt};
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const CpuStorage* ws = weight ? &std::get<CpuStorage>(*weight) : nullptr;
    std::size_t valid_count = 0;

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        auto* sp = reinterpret_cast<T*>(softmax.ptr.get());
        auto* lp = reinterpret_cast<T*>(losses.ptr.get());
        const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
        for (int n = 0; n < n_batch; ++n) {
            for (int s = 0; s < spatial; ++s) {
                T mx = -std::numeric_limits<T>::infinity();
                for (int c = 0; c < channels; ++c)
                    mx = std::max(mx, xp[(n * channels + c) * spatial + s]);
                T sum = T{0};
                for (int c = 0; c < channels; ++c) {
                    const T e = std::exp(xp[(n * channels + c) * spatial + s] - mx);
                    sp[(n * channels + c) * spatial + s] = e;
                    sum += e;
                }
                const T inv = T{1} / sum;
                for (int c = 0; c < channels; ++c)
                    sp[(n * channels + c) * spatial + s] *= inv;
                const std::int64_t y = class_target("cpu_backend::cross_entropy_loss", ts,
                                                    n * spatial + s, ignore_index, channels);
                if (y < 0) {
                    lp[n * spatial + s] = T{0};
                    continue;
                }
                ++valid_count;
                const T pred = sp[(n * channels + static_cast<int>(y)) * spatial + s];
                const T w = wp ? wp[static_cast<int>(y)] : T{1};
                lp[n * spatial + s] = -w * std::log(pred + static_cast<T>(eps));
            }
        }
    };
    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::cross_entropy_loss").not_implemented("dtype not supported");
    if (valid_count == 0)
        valid_count = 1;

    return {reduce_class_losses(losses, samples, valid_count, dt, reduction),
            Storage{std::move(softmax)},
            Storage{make_cpu_scalar(static_cast<double>(valid_count), dt)}};
}

Storage CpuBackend::cross_entropy_backward(const Storage& saved_softmax,
                                           const Storage& target,
                                           const Storage* weight,
                                           const Storage& valid_count,
                                           const Storage& grad,
                                           const Shape& input_shape,
                                           Dtype dt,
                                           int ignore_index,
                                           int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        std::optional<Storage> weight32;
        return detail::back_to_f16(
            cross_entropy_backward(detail::as_f32(saved_softmax), detail::as_f32(target),
                                   detail::as_f32(weight, weight32), detail::as_f32(valid_count),
                                   detail::as_f32(grad), input_shape, Dtype::F32, ignore_index,
                                   reduction),
            dt);
    }
    const int n_batch = static_cast<int>(input_shape[0]);
    const int channels = static_cast<int>(input_shape[1]);
    const int spatial = class_loss_spatial(input_shape);
    const std::size_t numel = static_cast<std::size_t>(n_batch) * channels * spatial;
    CpuStorage dx{allocate_aligned_bytes(numel * dtype_size(dt), Device::CPU),
                  numel * dtype_size(dt), dt};
    if (dx.nbytes)
        std::memset(dx.ptr.get(), 0, dx.nbytes);
    const auto& sm = std::get<CpuStorage>(saved_softmax);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& gs = std::get<CpuStorage>(grad);
    const auto& vc = std::get<CpuStorage>(valid_count);
    const CpuStorage* ws = weight ? &std::get<CpuStorage>(*weight) : nullptr;

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* sp = reinterpret_cast<const T*>(sm.ptr.get());
        const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx.ptr.get());
        const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
        const T valid = *reinterpret_cast<const T*>(vc.ptr.get());
        const bool elem = reduction == 0;
        for (int n = 0; n < n_batch; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const std::int64_t y = class_target("cpu_backend::cross_entropy_backward", ts,
                                                    n * spatial + s, ignore_index, channels);
                if (y < 0)
                    continue;
                const T go = elem ? gp[n * spatial + s] : gp[0];
                const T scale = (reduction == 1) ? (go / valid) : go;
                const T w = wp ? wp[static_cast<int>(y)] : T{1};
                for (int c = 0; c < channels; ++c) {
                    T v = sp[(n * channels + c) * spatial + s];
                    if (c == static_cast<int>(y))
                        v -= T{1};
                    dxp[(n * channels + c) * spatial + s] = w * v * scale;
                }
            }
        }
    };
    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::cross_entropy_backward").not_implemented("dtype not supported");
    return Storage{std::move(dx)};
}

ClassLossForwardResult CpuBackend::nll_loss(const Storage& input,
                                            const Storage& target,
                                            const Storage* weight,
                                            const Shape& input_shape,
                                            const Shape&,
                                            Dtype dt,
                                            int ignore_index,
                                            int reduction) {
    const int n_batch = static_cast<int>(input_shape[0]);
    const int channels = static_cast<int>(input_shape[1]);
    const int spatial = class_loss_spatial(input_shape);
    const std::size_t samples = static_cast<std::size_t>(n_batch) * spatial;
    CpuStorage losses{allocate_aligned_bytes(samples * dtype_size(dt), Device::CPU),
                      samples * dtype_size(dt), dt};
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const CpuStorage* ws = weight ? &std::get<CpuStorage>(*weight) : nullptr;
    std::size_t valid_count = 0;

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        auto* lp = reinterpret_cast<T*>(losses.ptr.get());
        const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
        for (int n = 0; n < n_batch; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const std::int64_t y = class_target("cpu_backend::nll_loss", ts, n * spatial + s,
                                                    ignore_index, channels);
                if (y < 0) {
                    lp[n * spatial + s] = T{0};
                    continue;
                }
                ++valid_count;
                const T w = wp ? wp[static_cast<int>(y)] : T{1};
                lp[n * spatial + s] = -w * xp[(n * channels + static_cast<int>(y)) * spatial + s];
            }
        }
    };
    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::nll_loss").not_implemented("dtype not supported");
    if (valid_count == 0)
        valid_count = 1;
    return {reduce_class_losses(losses, samples, valid_count, dt, reduction), Storage{},
            Storage{make_cpu_scalar(static_cast<double>(valid_count), dt)}};
}

Storage CpuBackend::nll_loss_backward(const Storage& target,
                                      const Storage* weight,
                                      const Storage& valid_count,
                                      const Storage& grad,
                                      const Shape& input_shape,
                                      Dtype dt,
                                      int ignore_index,
                                      int reduction) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        std::optional<Storage> weight32;
        return detail::back_to_f16(
            nll_loss_backward(detail::as_f32(target), detail::as_f32(weight, weight32),
                              detail::as_f32(valid_count), detail::as_f32(grad), input_shape,
                              Dtype::F32, ignore_index, reduction),
            dt);
    }
    const int n_batch = static_cast<int>(input_shape[0]);
    const int channels = static_cast<int>(input_shape[1]);
    const int spatial = class_loss_spatial(input_shape);
    const std::size_t numel = static_cast<std::size_t>(n_batch) * channels * spatial;
    CpuStorage dx{allocate_aligned_bytes(numel * dtype_size(dt), Device::CPU),
                  numel * dtype_size(dt), dt};
    if (dx.nbytes)
        std::memset(dx.ptr.get(), 0, dx.nbytes);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& gs = std::get<CpuStorage>(grad);
    const auto& vc = std::get<CpuStorage>(valid_count);
    const CpuStorage* ws = weight ? &std::get<CpuStorage>(*weight) : nullptr;

    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
        auto* dxp = reinterpret_cast<T*>(dx.ptr.get());
        const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
        const T valid = *reinterpret_cast<const T*>(vc.ptr.get());
        const bool elem = reduction == 0;
        for (int n = 0; n < n_batch; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const std::int64_t y = class_target("cpu_backend::nll_loss_backward", ts,
                                                    n * spatial + s, ignore_index, channels);
                if (y < 0)
                    continue;
                const T go = elem ? gp[n * spatial + s] : gp[0];
                const T scale = (reduction == 1) ? (go / valid) : go;
                const T w = wp ? wp[static_cast<int>(y)] : T{1};
                dxp[(n * channels + static_cast<int>(y)) * spatial + s] = -w * scale;
            }
        }
    };
    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::nll_loss_backward").not_implemented("dtype not supported");
    return Storage{std::move(dx)};
}

void CpuBackend::compute_mse_loss_values(
    const Storage& input, const Storage& target, std::byte* out, std::size_t n, Dtype dt) {
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    if (dt == Dtype::F32) {
        const auto* xp = reinterpret_cast<const float*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const float*>(ts.ptr.get());
        auto* lp = reinterpret_cast<float*>(out);
        for (std::size_t i = 0; i < n; ++i) {
            const float d = xp[i] - tp[i];
            lp[i] = d * d;
        }
    } else if (dt == Dtype::F64) {
        const auto* xp = reinterpret_cast<const double*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const double*>(ts.ptr.get());
        auto* lp = reinterpret_cast<double*>(out);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = xp[i] - tp[i];
            lp[i] = d * d;
        }
    } else {
        ErrorBuilder("cpu_backend::mse_loss").not_implemented("dtype not supported");
    }
}

template <typename T>
T CpuBackend::mse_loss_sum(const Storage& input, const Storage& target, std::size_t n) {
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
    const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
    T sum = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        const T d = xp[i] - tp[i];
        sum += d * d;
    }
    return sum;
}

void CpuBackend::compute_huber_loss_values(const Storage& input,
                                           const Storage& target,
                                           std::byte* out,
                                           std::size_t n,
                                           Dtype dt,
                                           double delta) {
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        auto* lp = reinterpret_cast<T*>(out);
        const T d = static_cast<T>(delta);
        for (std::size_t i = 0; i < n; ++i) {
            const T r = xp[i] - tp[i];
            const T ar = std::abs(r);
            lp[i] = (ar <= d) ? T{0.5} * r * r : d * (ar - T{0.5} * d);
        }
    };
    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::huber_loss").not_implemented("dtype not supported");
}

template <typename T>
T CpuBackend::huber_loss_sum(const Storage& input, const Storage& target, std::size_t n, T delta) {
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
    const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
    T sum = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        const T r = xp[i] - tp[i];
        const T ar = std::abs(r);
        sum += (ar <= delta) ? T{0.5} * r * r : delta * (ar - T{0.5} * delta);
    }
    return sum;
}

void CpuBackend::compute_bce_loss_values(const Storage& input,
                                         const Storage& target,
                                         const Storage& weight,
                                         std::byte* out,
                                         std::size_t n,
                                         Dtype dt,
                                         double eps) {
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& ws = std::get<CpuStorage>(weight);
    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
        auto* lp = reinterpret_cast<T*>(out);
        const T e = static_cast<T>(eps);
        const T one = static_cast<T>(1);
        for (std::size_t i = 0; i < n; ++i) {
            const T p = std::min(std::max(xp[i], e), one - e);
            const T l = -(tp[i] * std::log(p) + (one - tp[i]) * std::log(one - p));
            lp[i] = wp[i] * l;
        }
    };
    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::bce_loss").not_implemented("dtype not supported");
}

template <typename T>
T CpuBackend::bce_loss_sum(
    const Storage& input, const Storage& target, const Storage& weight, std::size_t n, T eps) {
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& ws = std::get<CpuStorage>(weight);
    const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
    const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
    const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
    const T one = static_cast<T>(1);
    T sum = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        const T p = std::min(std::max(xp[i], eps), one - eps);
        const T l = -(tp[i] * std::log(p) + (one - tp[i]) * std::log(one - p));
        sum += wp[i] * l;
    }
    return sum;
}

void CpuBackend::compute_bce_logits_values(const Storage& input,
                                           const Storage& target,
                                           const Storage& weight,
                                           const Storage& pos_weight,
                                           std::byte* out,
                                           std::size_t n,
                                           Dtype dt) {
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& ws = std::get<CpuStorage>(weight);
    const auto& pws = std::get<CpuStorage>(pos_weight);
    auto compute = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
        const auto* pwp = reinterpret_cast<const T*>(pws.ptr.get());
        auto* lp = reinterpret_cast<T*>(out);
        const T one = static_cast<T>(1);
        for (std::size_t i = 0; i < n; ++i) {
            const T x = xp[i];
            const T y = tp[i];
            const T log_weight = (pwp[i] - one) * y + one;
            const T log1pexp = std::log1p(std::exp(-std::abs(x)));
            const T loss = std::max(x, T{0}) - x * y + log_weight * log1pexp;
            lp[i] = wp[i] * loss;
        }
    };
    if (dt == Dtype::F32)
        compute(float{});
    else if (dt == Dtype::F64)
        compute(double{});
    else
        ErrorBuilder("cpu_backend::bce_with_logits_loss").not_implemented("dtype not supported");
}

template <typename T>
T CpuBackend::bce_logits_sum(const Storage& input,
                             const Storage& target,
                             const Storage& weight,
                             const Storage& pos_weight,
                             std::size_t n) {
    const auto& xs = std::get<CpuStorage>(input);
    const auto& ts = std::get<CpuStorage>(target);
    const auto& ws = std::get<CpuStorage>(weight);
    const auto& pws = std::get<CpuStorage>(pos_weight);
    const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
    const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
    const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
    const auto* pwp = reinterpret_cast<const T*>(pws.ptr.get());
    const T one = static_cast<T>(1);
    T sum = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        const T x = xp[i];
        const T y = tp[i];
        const T log_weight = (pwp[i] - one) * y + one;
        const T log1pexp = std::log1p(std::exp(-std::abs(x)));
        const T loss = std::max(x, T{0}) - x * y + log_weight * log1pexp;
        sum += wp[i] * loss;
    }
    return sum;
}

int CpuBackend::class_loss_spatial(const Shape& input_shape) {
    int spatial = 1;
    for (std::size_t i = 2; i < input_shape.size(); ++i)
        spatial *= static_cast<int>(input_shape[i]);
    return spatial;
}

CpuStorage CpuBackend::make_cpu_scalar(double value, Dtype dt) {
    CpuStorage out{allocate_aligned_bytes(dtype_size(dt), Device::CPU), dtype_size(dt), dt};
    if (dt == Dtype::F32)
        *reinterpret_cast<float*>(out.ptr.get()) = static_cast<float>(value);
    else if (dt == Dtype::F64)
        *reinterpret_cast<double*>(out.ptr.get()) = value;
    else
        ErrorBuilder("cpu_backend::class_loss_scalar").not_implemented("dtype not supported");
    return out;
}

std::int64_t CpuBackend::class_target(
    const char* op, const CpuStorage& target, std::size_t i, int ignore_index, int channels) {
    const std::int64_t y = cpu::real_index_reader(op, target)(i);
    if (y == ignore_index)
        return -1;
    return cpu::checked_row(op, y, channels);
}

template <typename T>
T CpuBackend::sum_values(const T* values, std::size_t n) {
    T sum = T{0};
    for (std::size_t i = 0; i < n; ++i)
        sum += values[i];
    return sum;
}

Storage CpuBackend::reduce_class_losses(const CpuStorage& losses,
                                        std::size_t samples,
                                        std::size_t valid_count,
                                        Dtype dt,
                                        int reduction) {
    if (reduction == 0) {
        CpuStorage out{allocate_aligned_bytes(samples * dtype_size(dt), Device::CPU),
                       samples * dtype_size(dt), dt};
        std::memcpy(out.ptr.get(), losses.ptr.get(), out.nbytes);
        return Storage{std::move(out)};
    }
    if (dt == Dtype::F32) {
        const auto* lp = reinterpret_cast<const float*>(losses.ptr.get());
        float sum = sum_values(lp, samples);
        if (reduction == 1)
            sum /= static_cast<float>(valid_count);
        return Storage{make_cpu_scalar(static_cast<double>(sum), dt)};
    }
    if (dt == Dtype::F64) {
        const auto* lp = reinterpret_cast<const double*>(losses.ptr.get());
        double sum = sum_values(lp, samples);
        if (reduction == 1)
            sum /= static_cast<double>(valid_count);
        return Storage{make_cpu_scalar(sum, dt)};
    }
    ErrorBuilder("cpu_backend::class_loss_reduce").not_implemented("dtype not supported");
}

}  // namespace backend
}  // namespace lucid
