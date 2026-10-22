// lucid/_C/backend/cpu/methods/Spatial.cpp
// Out-of-line CpuBackend method definitions for the Spatial family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::drop_block_mask(const Storage& seed,
                                    double drop_prob,
                                    std::int64_t block_size,
                                    const Shape& x_shape,
                                    Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            drop_block_mask(detail::as_f32(seed), drop_prob, block_size, x_shape, Dtype::F32), dt);
    const std::size_t B = static_cast<std::size_t>(x_shape[0]);
    const std::size_t C = static_cast<std::size_t>(x_shape[1]);
    const std::size_t H = static_cast<std::size_t>(x_shape[2]);
    const std::size_t W = static_cast<std::size_t>(x_shape[3]);
    const std::size_t total = B * C * H * W;
    const std::int64_t bsz = block_size;
    const std::int64_t pad_h = bsz / 2;

    const auto& s_cpu = std::get<CpuStorage>(seed);
    CpuStorage block_mask = alloc_cpu(total, dt);

    if (dt == Dtype::F32) {
        const float* sp = reinterpret_cast<const float*>(s_cpu.ptr.get());
        float* mp = reinterpret_cast<float*>(block_mask.ptr.get());
        for (std::size_t b = 0; b < B; ++b)
            for (std::size_t c = 0; c < C; ++c) {
                const float* s = sp + (b * C + c) * H * W;
                float* m = mp + (b * C + c) * H * W;
                for (std::size_t y = 0; y < H; ++y)
                    for (std::size_t x = 0; x < W; ++x) {
                        float mx = 0.f;
                        for (std::int64_t dy = 0; dy < bsz; ++dy) {
                            const std::int64_t yy = static_cast<std::int64_t>(y) + dy - pad_h;
                            if (yy < 0 || yy >= static_cast<std::int64_t>(H))
                                continue;
                            for (std::int64_t dx = 0; dx < bsz; ++dx) {
                                const std::int64_t xx = static_cast<std::int64_t>(x) + dx - pad_h;
                                if (xx < 0 || xx >= static_cast<std::int64_t>(W))
                                    continue;
                                const float v = s[yy * W + xx];
                                if (v > mx)
                                    mx = v;
                            }
                        }
                        m[y * W + x] = mx;
                    }
            }
        const float fs = static_cast<float>(1.0 / (1.0 - drop_prob));
        for (std::size_t i = 0; i < total; ++i)
            mp[i] = (1.f - mp[i]) * fs;
    } else if (dt == Dtype::F64) {
        const double* sp = reinterpret_cast<const double*>(s_cpu.ptr.get());
        double* mp = reinterpret_cast<double*>(block_mask.ptr.get());
        for (std::size_t b = 0; b < B; ++b)
            for (std::size_t c = 0; c < C; ++c) {
                const double* s = sp + (b * C + c) * H * W;
                double* m = mp + (b * C + c) * H * W;
                for (std::size_t y = 0; y < H; ++y)
                    for (std::size_t x = 0; x < W; ++x) {
                        double mx = 0.0;
                        for (std::int64_t dy = 0; dy < bsz; ++dy) {
                            const std::int64_t yy = static_cast<std::int64_t>(y) + dy - pad_h;
                            if (yy < 0 || yy >= static_cast<std::int64_t>(H))
                                continue;
                            for (std::int64_t dx = 0; dx < bsz; ++dx) {
                                const std::int64_t xx = static_cast<std::int64_t>(x) + dx - pad_h;
                                if (xx < 0 || xx >= static_cast<std::int64_t>(W))
                                    continue;
                                const double v = s[yy * W + xx];
                                if (v > mx)
                                    mx = v;
                            }
                        }
                        m[y * W + x] = mx;
                    }
            }
        const double scale = 1.0 / (1.0 - drop_prob);
        for (std::size_t i = 0; i < total; ++i)
            mp[i] = (1.0 - mp[i]) * scale;
    } else {
        ErrorBuilder("cpu_backend::drop_block_mask").not_implemented("dtype not supported");
    }
    return Storage{std::move(block_mask)};
}

std::vector<Storage> CpuBackend::max_pool_nd_forward(const Storage& x,
                                                     const Shape& x_shape,
                                                     const Shape& out_shape,
                                                     const PoolOpts& opts,
                                                     Dtype dt) {
    const int N = opts.N;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    int S[3], O[3];
    int O_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_shape[2 + i]);
        O[i] = static_cast<int>(out_shape[2 + i]);
        O_total *= O[i];
    }
    // Pooling is a *selection*, so every dtype the caller can name has
    // a right answer even though Accelerate only has float kernels for
    // it.  Half widens to float32 (exact — every half is a float32);
    // an integer widens to double, which is exact for bool, int8,
    // int16 and int32 outright and for int64 up to 2^53.  The argmax
    // is an index either way and travels through untouched, which is
    // why the narrowing helpers key off each storage's own dtype.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            max_pool_nd_forward(detail::as_f32(x), x_shape, out_shape, opts, Dtype::F32), dt);
    if (detail::is_int_like(dt))
        return detail::f64_back_to_int(
            max_pool_nd_forward(detail::as_f64(x), x_shape, out_shape, opts, Dtype::F64), dt);
    const auto& xc = std::get<CpuStorage>(x);
    auto y_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * O_total, dt);
    auto am_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * O_total, Dtype::I32);
    auto dispatch = [&](auto tag) {
        using T = decltype(tag);
        const T* xp = reinterpret_cast<const T*>(xc.ptr.get());
        T* yp = reinterpret_cast<T*>(y_cpu.ptr.get());
        auto* ap = reinterpret_cast<std::int32_t*>(am_cpu.ptr.get());
        if constexpr (std::is_same_v<T, float>) {
            if (N == 1)
                cpu::max_pool1d_forward_f32(xp, yp, ap, B, C, S[0], opts.K[0], O[0], opts.stride[0],
                                            opts.pad[0]);
            else if (N == 2)
                cpu::max_pool2d_forward_f32(xp, yp, ap, B, C, S[0], S[1], opts.K[0], opts.K[1],
                                            O[0], O[1], opts.stride[0], opts.stride[1], opts.pad[0],
                                            opts.pad[1]);
            else
                cpu::max_pool3d_forward_f32(xp, yp, ap, B, C, S[0], S[1], S[2], opts.K[0],
                                            opts.K[1], opts.K[2], O[0], O[1], O[2], opts.stride[0],
                                            opts.stride[1], opts.stride[2], opts.pad[0],
                                            opts.pad[1], opts.pad[2]);
        } else {
            if (N == 1)
                cpu::max_pool1d_forward_f64(xp, yp, ap, B, C, S[0], opts.K[0], O[0], opts.stride[0],
                                            opts.pad[0]);
            else if (N == 2)
                cpu::max_pool2d_forward_f64(xp, yp, ap, B, C, S[0], S[1], opts.K[0], opts.K[1],
                                            O[0], O[1], opts.stride[0], opts.stride[1], opts.pad[0],
                                            opts.pad[1]);
            else
                cpu::max_pool3d_forward_f64(xp, yp, ap, B, C, S[0], S[1], S[2], opts.K[0],
                                            opts.K[1], opts.K[2], O[0], O[1], O[2], opts.stride[0],
                                            opts.stride[1], opts.stride[2], opts.pad[0],
                                            opts.pad[1], opts.pad[2]);
        }
    };
    if (dt == Dtype::F32)
        dispatch(float{});
    else if (dt == Dtype::F64)
        dispatch(double{});
    else
        ErrorBuilder("max_pool_nd_forward").not_implemented("dtype");
    return {Storage{std::move(y_cpu)}, Storage{std::move(am_cpu)}};
}

Storage CpuBackend::max_pool_nd_backward(const Storage& grad_out,
                                         const Storage& saved_argmax,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) {
    const int N = opts.N;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    int S[3], O[3];
    int S_total = 1, O_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_shape[2 + i]);
        O[i] = static_cast<int>(out_shape[2 + i]);
        S_total *= S[i];
        O_total *= O[i];
    }
    (void)O_total;
    if (detail::is_half_like(dt))
        return detail::back_to_f16(max_pool_nd_backward(detail::as_f32(grad_out), saved_argmax,
                                                        x_shape, out_shape, opts, Dtype::F32),
                                   dt);
    if (detail::is_int_like(dt))
        return detail::f64_back_to_int(max_pool_nd_backward(detail::as_f64(grad_out), saved_argmax,
                                                            x_shape, out_shape, opts, Dtype::F64),
                                       dt);
    const auto& gc = std::get<CpuStorage>(grad_out);
    const auto& ac = std::get<CpuStorage>(saved_argmax);
    auto dx_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * S_total, dt);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    auto dispatch = [&](auto tag) {
        using T = decltype(tag);
        const T* gp = reinterpret_cast<const T*>(gc.ptr.get());
        const auto* ap = reinterpret_cast<const std::int32_t*>(ac.ptr.get());
        T* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
        if constexpr (std::is_same_v<T, float>) {
            if (N == 1)
                cpu::max_pool1d_backward_f32(gp, ap, dxp, B, C, S[0], O[0]);
            else if (N == 2)
                cpu::max_pool2d_backward_f32(gp, ap, dxp, B, C, S[0], S[1], O[0], O[1]);
            else
                cpu::max_pool3d_backward_f32(gp, ap, dxp, B, C, S[0], S[1], S[2], O[0], O[1], O[2]);
        } else {
            if (N == 1)
                cpu::max_pool1d_backward_f64(gp, ap, dxp, B, C, S[0], O[0]);
            else if (N == 2)
                cpu::max_pool2d_backward_f64(gp, ap, dxp, B, C, S[0], S[1], O[0], O[1]);
            else
                cpu::max_pool3d_backward_f64(gp, ap, dxp, B, C, S[0], S[1], S[2], O[0], O[1], O[2]);
        }
    };
    if (dt == Dtype::F32)
        dispatch(float{});
    else if (dt == Dtype::F64)
        dispatch(double{});
    else
        ErrorBuilder("max_pool_nd_backward").not_implemented("dtype");
    return Storage{std::move(dx_cpu)};
}

Storage CpuBackend::avg_pool_nd_forward(const Storage& x,
                                        const Shape& x_shape,
                                        const Shape& out_shape,
                                        const PoolOpts& opts,
                                        Dtype dt) {
    const int N = opts.N;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    int S[3], O[3];
    int O_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_shape[2 + i]);
        O[i] = static_cast<int>(out_shape[2 + i]);
        O_total *= O[i];
    }
    // Same route as max_pool above.  The average of integers rounds to
    // the caller's integer width on the way out, which is what an
    // integer-typed mean means.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            avg_pool_nd_forward(detail::as_f32(x), x_shape, out_shape, opts, Dtype::F32), dt);
    if (detail::is_int_like(dt))
        return detail::f64_back_to_int(
            avg_pool_nd_forward(detail::as_f64(x), x_shape, out_shape, opts, Dtype::F64), dt,
            detail::IntRounding::TowardZero);
    const auto& xc = std::get<CpuStorage>(x);
    auto y_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * O_total, dt);
    auto dispatch = [&](auto tag) {
        using T = decltype(tag);
        const T* xp = reinterpret_cast<const T*>(xc.ptr.get());
        T* yp = reinterpret_cast<T*>(y_cpu.ptr.get());
        if constexpr (std::is_same_v<T, float>) {
            if (N == 1)
                cpu::avg_pool1d_forward_f32(xp, yp, B, C, S[0], opts.K[0], O[0], opts.stride[0],
                                            opts.pad[0], opts.count_include_pad);
            else if (N == 2)
                cpu::avg_pool2d_forward_f32(xp, yp, B, C, S[0], S[1], opts.K[0], opts.K[1], O[0],
                                            O[1], opts.stride[0], opts.stride[1], opts.pad[0],
                                            opts.pad[1], opts.count_include_pad);
            else
                cpu::avg_pool3d_forward_f32(xp, yp, B, C, S[0], S[1], S[2], opts.K[0], opts.K[1],
                                            opts.K[2], O[0], O[1], O[2], opts.stride[0],
                                            opts.stride[1], opts.stride[2], opts.pad[0],
                                            opts.pad[1], opts.pad[2], opts.count_include_pad);
        } else {
            if (N == 1)
                cpu::avg_pool1d_forward_f64(xp, yp, B, C, S[0], opts.K[0], O[0], opts.stride[0],
                                            opts.pad[0], opts.count_include_pad);
            else if (N == 2)
                cpu::avg_pool2d_forward_f64(xp, yp, B, C, S[0], S[1], opts.K[0], opts.K[1], O[0],
                                            O[1], opts.stride[0], opts.stride[1], opts.pad[0],
                                            opts.pad[1], opts.count_include_pad);
            else
                cpu::avg_pool3d_forward_f64(xp, yp, B, C, S[0], S[1], S[2], opts.K[0], opts.K[1],
                                            opts.K[2], O[0], O[1], O[2], opts.stride[0],
                                            opts.stride[1], opts.stride[2], opts.pad[0],
                                            opts.pad[1], opts.pad[2], opts.count_include_pad);
        }
    };
    if (dt == Dtype::F32)
        dispatch(float{});
    else if (dt == Dtype::F64)
        dispatch(double{});
    else
        ErrorBuilder("avg_pool_nd_forward").not_implemented("dtype");
    return Storage{std::move(y_cpu)};
}

Storage CpuBackend::avg_pool_nd_backward(const Storage& grad_out,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) {
    const int N = opts.N;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    int S[3], O[3];
    int S_total = 1, O_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_shape[2 + i]);
        O[i] = static_cast<int>(out_shape[2 + i]);
        S_total *= S[i];
        O_total *= O[i];
    }
    (void)O_total;
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            avg_pool_nd_backward(detail::as_f32(grad_out), x_shape, out_shape, opts, Dtype::F32),
            dt);
    if (detail::is_int_like(dt))
        return detail::f64_back_to_int(
            avg_pool_nd_backward(detail::as_f64(grad_out), x_shape, out_shape, opts, Dtype::F64),
            dt, detail::IntRounding::TowardZero);
    const auto& gc = std::get<CpuStorage>(grad_out);
    auto dx_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * S_total, dt);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    auto dispatch = [&](auto tag) {
        using T = decltype(tag);
        const T* gp = reinterpret_cast<const T*>(gc.ptr.get());
        T* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
        if constexpr (std::is_same_v<T, float>) {
            if (N == 1)
                cpu::avg_pool1d_backward_f32(gp, dxp, B, C, S[0], opts.K[0], O[0], opts.stride[0],
                                             opts.pad[0], opts.count_include_pad);
            else if (N == 2)
                cpu::avg_pool2d_backward_f32(gp, dxp, B, C, S[0], S[1], opts.K[0], opts.K[1], O[0],
                                             O[1], opts.stride[0], opts.stride[1], opts.pad[0],
                                             opts.pad[1], opts.count_include_pad);
            else
                cpu::avg_pool3d_backward_f32(gp, dxp, B, C, S[0], S[1], S[2], opts.K[0], opts.K[1],
                                             opts.K[2], O[0], O[1], O[2], opts.stride[0],
                                             opts.stride[1], opts.stride[2], opts.pad[0],
                                             opts.pad[1], opts.pad[2], opts.count_include_pad);
        } else {
            if (N == 1)
                cpu::avg_pool1d_backward_f64(gp, dxp, B, C, S[0], opts.K[0], O[0], opts.stride[0],
                                             opts.pad[0], opts.count_include_pad);
            else if (N == 2)
                cpu::avg_pool2d_backward_f64(gp, dxp, B, C, S[0], S[1], opts.K[0], opts.K[1], O[0],
                                             O[1], opts.stride[0], opts.stride[1], opts.pad[0],
                                             opts.pad[1], opts.count_include_pad);
            else
                cpu::avg_pool3d_backward_f64(gp, dxp, B, C, S[0], S[1], S[2], opts.K[0], opts.K[1],
                                             opts.K[2], O[0], O[1], O[2], opts.stride[0],
                                             opts.stride[1], opts.stride[2], opts.pad[0],
                                             opts.pad[1], opts.pad[2], opts.count_include_pad);
        }
    };
    if (dt == Dtype::F32)
        dispatch(float{});
    else if (dt == Dtype::F64)
        dispatch(double{});
    else
        ErrorBuilder("avg_pool_nd_backward").not_implemented("dtype");
    return Storage{std::move(dx_cpu)};
}

Storage CpuBackend::interpolate_nearest_2d_forward(
    const Storage& input, const Shape& in_shape, int H_out, int W_out, Dtype dt) {
    return cpu::interp::resample(
        *this, input,
        cpu::interp::make_request(cpu::interp::Filter::Nearest, in_shape, 1, H_out, W_out, false),
        dt, false);
}

Storage CpuBackend::interpolate_nearest_3d_forward(
    const Storage& input, const Shape& in_shape, int D_out, int H_out, int W_out, Dtype dt) {
    return cpu::interp::resample(*this, input,
                                 cpu::interp::make_request(cpu::interp::Filter::Nearest, in_shape,
                                                           D_out, H_out, W_out, false),
                                 dt, false);
}

Storage CpuBackend::interpolate_nearest_2d_backward(
    const Storage& grad_out, const Shape& in_shape, int H_out, int W_out, Dtype dt) {
    return cpu::interp::resample(
        *this, grad_out,
        cpu::interp::make_request(cpu::interp::Filter::Nearest, in_shape, 1, H_out, W_out, false),
        dt, true);
}

Storage CpuBackend::interpolate_nearest_3d_backward(
    const Storage& grad_out, const Shape& in_shape, int D_out, int H_out, int W_out, Dtype dt) {
    return cpu::interp::resample(*this, grad_out,
                                 cpu::interp::make_request(cpu::interp::Filter::Nearest, in_shape,
                                                           D_out, H_out, W_out, false),
                                 dt, true);
}

Storage CpuBackend::interpolate_bilinear_forward(const Storage& input,
                                                 const Shape& in_shape,
                                                 int H_out,
                                                 int W_out,
                                                 bool align_corners,
                                                 Dtype dt) {
    return cpu::interp::resample(*this, input,
                                 cpu::interp::make_request(cpu::interp::Filter::Linear, in_shape, 1,
                                                           H_out, W_out, align_corners),
                                 dt, false);
}

Storage CpuBackend::interpolate_bilinear_backward(const Storage& grad_out,
                                                  const Shape& in_shape,
                                                  int H_out,
                                                  int W_out,
                                                  bool align_corners,
                                                  Dtype dt) {
    return cpu::interp::resample(*this, grad_out,
                                 cpu::interp::make_request(cpu::interp::Filter::Linear, in_shape, 1,
                                                           H_out, W_out, align_corners),
                                 dt, true);
}

Storage CpuBackend::interpolate_trilinear_forward(const Storage& input,
                                                  const Shape& in_shape,
                                                  int D_out,
                                                  int H_out,
                                                  int W_out,
                                                  bool align_corners,
                                                  Dtype dt) {
    return cpu::interp::resample(*this, input,
                                 cpu::interp::make_request(cpu::interp::Filter::Linear, in_shape,
                                                           D_out, H_out, W_out, align_corners),
                                 dt, false);
}

Storage CpuBackend::interpolate_trilinear_backward(const Storage& grad_out,
                                                   const Shape& in_shape,
                                                   int D_out,
                                                   int H_out,
                                                   int W_out,
                                                   bool align_corners,
                                                   Dtype dt) {
    return cpu::interp::resample(*this, grad_out,
                                 cpu::interp::make_request(cpu::interp::Filter::Linear, in_shape,
                                                           D_out, H_out, W_out, align_corners),
                                 dt, true);
}

Storage CpuBackend::affine_grid_forward(
    const Storage& theta, int N, int H, int W, bool align_corners, Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            affine_grid_forward(detail::as_f32(theta), N, H, W, align_corners, Dtype::F32), dt);
    const auto& th = std::get<CpuStorage>(theta);
    auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * H * W * 2, dt);

    auto x_norm_fn = [](int w, int W2, bool align, auto T_tag) {
        using T = decltype(T_tag);
        if (align)
            return W2 > 1 ? T{-1} + (T{2} * static_cast<T>(w)) / static_cast<T>(W2 - 1) : T{0};
        return T{-1} + (T{2} * static_cast<T>(w) + T{1}) / static_cast<T>(W2);
    };

    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* tp = reinterpret_cast<const T*>(th.ptr.get());
        T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
        for (int n = 0; n < N; ++n) {
            const T t00 = tp[n * 6 + 0], t01 = tp[n * 6 + 1], t02 = tp[n * 6 + 2];
            const T t10 = tp[n * 6 + 3], t11 = tp[n * 6 + 4], t12 = tp[n * 6 + 5];
            for (int h = 0; h < H; ++h) {
                const T y = x_norm_fn(h, H, align_corners, T{});
                for (int w = 0; w < W; ++w) {
                    const T x = x_norm_fn(w, W, align_corners, T{});
                    const std::size_t base = ((static_cast<std::size_t>(n) * H + h) * W + w) * 2;
                    op[base + 0] = t00 * x + t01 * y + t02;
                    op[base + 1] = t10 * x + t11 * y + t12;
                }
            }
        }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu::affine_grid_forward").not_implemented("dtype must be F32/F64");
    return Storage{std::move(out_cpu)};
}

Storage CpuBackend::affine_grid_backward(
    const Storage& grad_grid, int N, int H, int W, bool align_corners, Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            affine_grid_backward(detail::as_f32(grad_grid), N, H, W, align_corners, Dtype::F32),
            dt);
    const auto& gs = std::get<CpuStorage>(grad_grid);
    auto dtheta = alloc_cpu(static_cast<std::size_t>(N) * 2 * 3, dt);
    std::memset(dtheta.ptr.get(), 0, dtheta.nbytes);

    auto x_norm_fn = [](int w, int W2, bool align, auto T_tag) {
        using T = decltype(T_tag);
        if (align)
            return W2 > 1 ? T{-1} + (T{2} * static_cast<T>(w)) / static_cast<T>(W2 - 1) : T{0};
        return T{-1} + (T{2} * static_cast<T>(w) + T{1}) / static_cast<T>(W2);
    };

    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
        T* dp = reinterpret_cast<T*>(dtheta.ptr.get());
        for (int n = 0; n < N; ++n) {
            T s00 = T{0}, s01 = T{0}, s02 = T{0};
            T s10 = T{0}, s11 = T{0}, s12 = T{0};
            for (int h = 0; h < H; ++h) {
                const T y = x_norm_fn(h, H, align_corners, T{});
                for (int w = 0; w < W; ++w) {
                    const T x = x_norm_fn(w, W, align_corners, T{});
                    const std::size_t base = ((static_cast<std::size_t>(n) * H + h) * W + w) * 2;
                    const T gx = gp[base + 0];
                    const T gy = gp[base + 1];
                    s00 += gx * x;
                    s01 += gx * y;
                    s02 += gx;
                    s10 += gy * x;
                    s11 += gy * y;
                    s12 += gy;
                }
            }
            dp[n * 6 + 0] = s00;
            dp[n * 6 + 1] = s01;
            dp[n * 6 + 2] = s02;
            dp[n * 6 + 3] = s10;
            dp[n * 6 + 4] = s11;
            dp[n * 6 + 5] = s12;
        }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu::affine_grid_backward").not_implemented("dtype must be F32/F64");
    return Storage{std::move(dtheta)};
}

Storage CpuBackend::grid_sample_forward(const Storage& input,
                                        const Storage& grid,
                                        const Shape& in_shape,
                                        const Shape& grid_shape,
                                        int mode,
                                        int padding_mode,
                                        bool align_corners,
                                        Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(grid_sample_forward(detail::as_f32(input), detail::as_f32(grid),
                                                       in_shape, grid_shape, mode, padding_mode,
                                                       align_corners, Dtype::F32),
                                   dt);
    const auto g = cpu::grid::geometry(in_shape, grid_shape, mode, padding_mode, align_corners);
    auto out = alloc_cpu(static_cast<std::size_t>(g.N * g.C) * g.out_plane(), dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        cpu::grid::forward(reinterpret_cast<const T*>(std::get<CpuStorage>(input).ptr.get()),
                           reinterpret_cast<const T*>(std::get<CpuStorage>(grid).ptr.get()),
                           reinterpret_cast<T*>(out.ptr.get()), g);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu::grid_sample_forward").not_implemented("dtype must be F32/F64");
    return Storage{std::move(out)};
}

std::vector<Storage> CpuBackend::grid_sample_backward(const Storage& grad_out,
                                                      const Storage& input,
                                                      const Storage& grid,
                                                      const Shape& in_shape,
                                                      const Shape& grid_shape,
                                                      int mode,
                                                      int padding_mode,
                                                      bool align_corners,
                                                      Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(grid_sample_backward(detail::as_f32(grad_out),
                                                        detail::as_f32(input), detail::as_f32(grid),
                                                        in_shape, grid_shape, mode, padding_mode,
                                                        align_corners, Dtype::F32),
                                   dt);
    const auto g = cpu::grid::geometry(in_shape, grid_shape, mode, padding_mode, align_corners);
    auto dx = alloc_cpu(static_cast<std::size_t>(g.N * g.C) * g.in_plane(), dt);
    auto dg = alloc_cpu(static_cast<std::size_t>(g.N) * g.out_plane() * 2, dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        auto in = [](const Storage& s) {
            return reinterpret_cast<const T*>(std::get<CpuStorage>(s).ptr.get());
        };
        cpu::grid::backward(in(grad_out), in(input), in(grid), reinterpret_cast<T*>(dx.ptr.get()),
                            reinterpret_cast<T*>(dg.ptr.get()), g);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu::grid_sample_backward").not_implemented("dtype must be F32/F64");
    return {Storage{std::move(dx)}, Storage{std::move(dg)}};
}

Storage CpuBackend::rotate_forward(const Storage& input,
                                   const Shape& shape,
                                   double angle_rad_neg,
                                   double cx,
                                   double cy,
                                   Dtype dt) {
    const int N = static_cast<int>(shape[0]);
    const int C = static_cast<int>(shape[1]);
    const int H = static_cast<int>(shape[2]);
    const int W = static_cast<int>(shape[3]);
    const auto& xs = std::get<CpuStorage>(input);
    auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * H * W, dt);
    const double cosv = std::cos(angle_rad_neg);
    const double sinv = std::sin(angle_rad_neg);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
        T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
        for (int n = 0; n < N; ++n)
            for (int ch = 0; ch < C; ++ch) {
                const T* base = xp + (n * C + ch) * H * W;
                T* obase = op + (n * C + ch) * H * W;
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        const double xsd = cosv * (x - cx) - sinv * (y - cy) + cx;
                        const double ysd = sinv * (x - cx) + cosv * (y - cy) + cy;
                        const auto xs2 = resample::index_of(std::floor(xsd + 0.5));
                        const auto ys2 = resample::index_of(std::floor(ysd + 0.5));
                        if (xs2 < 0 || xs2 >= W || ys2 < 0 || ys2 >= H)
                            obase[y * W + x] = T{0};
                        else
                            obase[y * W + x] = base[ys2 * W + xs2];
                    }
            }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu::rotate_forward").not_implemented("dtype");
    return Storage{std::move(out_cpu)};
}

}  // namespace backend
}  // namespace lucid
