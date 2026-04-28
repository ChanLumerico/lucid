#include "Interpolate.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Exceptions.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/TensorImpl.h"
#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype  = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr    = allocate_aligned_bytes(s.nbytes);
    return s;
}

template <typename T>
inline T src_coord(int out_idx, int in_dim, int out_dim, bool align_corners) {
    if (align_corners) {
        if (out_dim <= 1) return T{0};
        return static_cast<T>(out_idx) * static_cast<T>(in_dim - 1)
                / static_cast<T>(out_dim - 1);
    }
    const T x = (static_cast<T>(out_idx) + T{0.5})
                * static_cast<T>(in_dim) / static_cast<T>(out_dim) - T{0.5};
    return x;
}

inline ::mlx::core::array mlx_scalar(double v, ::mlx::core::Dtype dt) {
    return ::mlx::core::astype(::mlx::core::array(static_cast<float>(v)), dt);
}

// Build a 1-D MLX array of source-coordinates of length out_dim.
::mlx::core::array build_src_coords(int in_dim, int out_dim, bool align_corners,
                                      ::mlx::core::Dtype dt) {
    auto idx = ::mlx::core::astype(::mlx::core::arange(0, out_dim, 1), dt);
    if (align_corners) {
        if (out_dim <= 1) return ::mlx::core::zeros({out_dim}, dt);
        auto step = mlx_scalar(static_cast<double>(in_dim - 1) /
                                 static_cast<double>(out_dim - 1), dt);
        return ::mlx::core::multiply(idx, step);
    }
    auto half = mlx_scalar(0.5, dt);
    auto scale = mlx_scalar(static_cast<double>(in_dim) /
                              static_cast<double>(out_dim), dt);
    auto a = ::mlx::core::add(idx, half);
    return ::mlx::core::subtract(::mlx::core::multiply(a, scale), half);
}

}  // namespace

// =====================================================================
// interpolate_bilinear (4-D)
// =====================================================================

const OpSchema InterpolateBilinearBackward::schema_v1{
    "interpolate_bilinear", 1, AmpPolicy::Promote, true};

namespace {

template <typename T>
void bilinear_forward_cpu(const T* xp, T* op,
                           int N, int C, int H_in, int W_in,
                           int H_out, int W_out, bool align_corners) {
    const std::size_t in_chan  = static_cast<std::size_t>(H_in)  * W_in;
    const std::size_t out_chan = static_cast<std::size_t>(H_out) * W_out;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                T iy = src_coord<T>(h, H_in, H_out, align_corners);
                if (iy < T{0}) iy = T{0};
                if (iy > static_cast<T>(H_in - 1)) iy = static_cast<T>(H_in - 1);
                int y0 = static_cast<int>(std::floor(iy));
                int y1 = std::min(y0 + 1, H_in - 1);
                const T dy = iy - static_cast<T>(y0);
                for (int w = 0; w < W_out; ++w) {
                    T ix = src_coord<T>(w, W_in, W_out, align_corners);
                    if (ix < T{0}) ix = T{0};
                    if (ix > static_cast<T>(W_in - 1)) ix = static_cast<T>(W_in - 1);
                    int x0 = static_cast<int>(std::floor(ix));
                    int x1 = std::min(x0 + 1, W_in - 1);
                    const T dx = ix - static_cast<T>(x0);
                    const T w00 = (T{1} - dy) * (T{1} - dx);
                    const T w01 = (T{1} - dy) * dx;
                    const T w10 = dy * (T{1} - dx);
                    const T w11 = dy * dx;
                    const T* base = xp + (n * C + c) * in_chan;
                    op[(n * C + c) * out_chan + h * W_out + w] =
                        base[y0 * W_in + x0] * w00 +
                        base[y0 * W_in + x1] * w01 +
                        base[y1 * W_in + x0] * w10 +
                        base[y1 * W_in + x1] * w11;
                }
            }
        }
    }
}

template <typename T>
void bilinear_backward_cpu(const T* go_p, T* dx_p,
                            int N, int C, int H_in, int W_in,
                            int H_out, int W_out, bool align_corners) {
    const std::size_t in_chan  = static_cast<std::size_t>(H_in)  * W_in;
    const std::size_t out_chan = static_cast<std::size_t>(H_out) * W_out;
    std::memset(dx_p, 0, sizeof(T) * static_cast<std::size_t>(N) * C * in_chan);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                T iy = src_coord<T>(h, H_in, H_out, align_corners);
                if (iy < T{0}) iy = T{0};
                if (iy > static_cast<T>(H_in - 1)) iy = static_cast<T>(H_in - 1);
                int y0 = static_cast<int>(std::floor(iy));
                int y1 = std::min(y0 + 1, H_in - 1);
                const T dy = iy - static_cast<T>(y0);
                for (int w = 0; w < W_out; ++w) {
                    T ix = src_coord<T>(w, W_in, W_out, align_corners);
                    if (ix < T{0}) ix = T{0};
                    if (ix > static_cast<T>(W_in - 1)) ix = static_cast<T>(W_in - 1);
                    int x0 = static_cast<int>(std::floor(ix));
                    int x1 = std::min(x0 + 1, W_in - 1);
                    const T dx = ix - static_cast<T>(x0);
                    const T w00 = (T{1} - dy) * (T{1} - dx);
                    const T w01 = (T{1} - dy) * dx;
                    const T w10 = dy * (T{1} - dx);
                    const T w11 = dy * dx;
                    const T g = go_p[(n * C + c) * out_chan + h * W_out + w];
                    T* base = dx_p + (n * C + c) * in_chan;
                    base[y0 * W_in + x0] += g * w00;
                    base[y0 * W_in + x1] += g * w01;
                    base[y1 * W_in + x0] += g * w10;
                    base[y1 * W_in + x1] += g * w11;
                }
            }
        }
    }
}

// GPU helper: gather input at integer (y, x) indices using flat-index take.
// y/x: shape [H_out, W_out] int64. Result: [N, C, H_out, W_out].
::mlx::core::array gather_2d_corner(const ::mlx::core::array& input,
                                      const ::mlx::core::array& y_idx,
                                      const ::mlx::core::array& x_idx,
                                      int N, int C, int H_in, int W_in,
                                      int H_out, int W_out) {
    auto n_idx = ::mlx::core::reshape(
        ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int64),
        {N, 1, 1, 1});
    auto c_idx = ::mlx::core::reshape(
        ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int64),
        {1, C, 1, 1});
    auto y_b = ::mlx::core::reshape(y_idx, {1, 1, H_out, W_out});
    auto x_b = ::mlx::core::reshape(x_idx, {1, 1, H_out, W_out});
    auto sn = ::mlx::core::astype(::mlx::core::array(C * H_in * W_in), ::mlx::core::int64);
    auto sc = ::mlx::core::astype(::mlx::core::array(H_in * W_in), ::mlx::core::int64);
    auto sy = ::mlx::core::astype(::mlx::core::array(W_in), ::mlx::core::int64);
    auto flat = ::mlx::core::add(
        ::mlx::core::add(::mlx::core::multiply(n_idx, sn),
                          ::mlx::core::multiply(c_idx, sc)),
        ::mlx::core::add(::mlx::core::multiply(y_b, sy), x_b));
    auto flat_b = ::mlx::core::broadcast_to(flat, {N, C, H_out, W_out});
    auto in_flat = ::mlx::core::reshape(input, {N * C * H_in * W_in});
    return ::mlx::core::take(in_flat, flat_b);
}

}  // namespace

TensorImplPtr InterpolateBilinearBackward::forward(const TensorImplPtr& input,
                                                     int H_out, int W_out,
                                                     bool align_corners) {
    if (!input) throw LucidError("interpolate_bilinear: null input");
    if (input->shape_.size() != 4)
        throw ShapeMismatch(input->shape_, Shape{},
                             "interpolate_bilinear: input must be 4-D (N, C, H, W)");
    const int N = static_cast<int>(input->shape_[0]);
    const int C = static_cast<int>(input->shape_[1]);
    const int H_in = static_cast<int>(input->shape_[2]);
    const int W_in = static_cast<int>(input->shape_[3]);
    Shape out_shape{N, C, H_out, W_out};
    OpScope scope{schema_v1.name, input->device_, input->dtype_, out_shape};

    Storage out_storage;
    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype_);
        // Build float source coordinates [H_out] / [W_out], clip to bounds.
        auto ys = build_src_coords(H_in, H_out, align_corners, mlx_dt);
        auto xs = build_src_coords(W_in, W_out, align_corners, mlx_dt);
        auto zero = mlx_scalar(0.0, mlx_dt);
        auto Hm1 = mlx_scalar(H_in - 1, mlx_dt);
        auto Wm1 = mlx_scalar(W_in - 1, mlx_dt);
        ys = ::mlx::core::clip(ys, std::optional<::mlx::core::array>(zero),
                                  std::optional<::mlx::core::array>(Hm1));
        xs = ::mlx::core::clip(xs, std::optional<::mlx::core::array>(zero),
                                  std::optional<::mlx::core::array>(Wm1));
        auto y0_f = ::mlx::core::floor(ys);
        auto x0_f = ::mlx::core::floor(xs);
        auto dy = ::mlx::core::subtract(ys, y0_f);
        auto dx = ::mlx::core::subtract(xs, x0_f);
        auto y0 = ::mlx::core::astype(y0_f, ::mlx::core::int64);
        auto x0 = ::mlx::core::astype(x0_f, ::mlx::core::int64);
        auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
        auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
        auto y1 = ::mlx::core::minimum(::mlx::core::add(y0,
                            ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64)),
                          Hm1_i);
        auto x1 = ::mlx::core::minimum(::mlx::core::add(x0,
                            ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64)),
                          Wm1_i);
        // Broadcast to 2-D index arrays [H_out, W_out].
        auto y0_2d = ::mlx::core::broadcast_to(::mlx::core::reshape(y0, {H_out, 1}),
                                                 {H_out, W_out});
        auto y1_2d = ::mlx::core::broadcast_to(::mlx::core::reshape(y1, {H_out, 1}),
                                                 {H_out, W_out});
        auto x0_2d = ::mlx::core::broadcast_to(::mlx::core::reshape(x0, {1, W_out}),
                                                 {H_out, W_out});
        auto x1_2d = ::mlx::core::broadcast_to(::mlx::core::reshape(x1, {1, W_out}),
                                                 {H_out, W_out});
        auto I00 = gather_2d_corner(*gx.arr, y0_2d, x0_2d, N, C, H_in, W_in, H_out, W_out);
        auto I01 = gather_2d_corner(*gx.arr, y0_2d, x1_2d, N, C, H_in, W_in, H_out, W_out);
        auto I10 = gather_2d_corner(*gx.arr, y1_2d, x0_2d, N, C, H_in, W_in, H_out, W_out);
        auto I11 = gather_2d_corner(*gx.arr, y1_2d, x1_2d, N, C, H_in, W_in, H_out, W_out);
        // Build weights [1, 1, H_out, W_out].
        auto one = mlx_scalar(1.0, mlx_dt);
        auto dy_2d = ::mlx::core::broadcast_to(::mlx::core::reshape(dy, {1, 1, H_out, 1}),
                                                 {1, 1, H_out, W_out});
        auto dx_2d = ::mlx::core::broadcast_to(::mlx::core::reshape(dx, {1, 1, 1, W_out}),
                                                 {1, 1, H_out, W_out});
        auto omdy = ::mlx::core::subtract(one, dy_2d);
        auto omdx = ::mlx::core::subtract(one, dx_2d);
        auto w00 = ::mlx::core::multiply(omdy, omdx);
        auto w01 = ::mlx::core::multiply(omdy, dx_2d);
        auto w10 = ::mlx::core::multiply(dy_2d, omdx);
        auto w11 = ::mlx::core::multiply(dy_2d, dx_2d);
        auto y_out = ::mlx::core::add(
            ::mlx::core::add(::mlx::core::multiply(I00, w00),
                              ::mlx::core::multiply(I01, w01)),
            ::mlx::core::add(::mlx::core::multiply(I10, w10),
                              ::mlx::core::multiply(I11, w11)));
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y_out), input->dtype_)};
    } else {
        auto out_cpu = allocate_size(static_cast<std::size_t>(N) * C * H_out * W_out,
                                      input->dtype_);
        const auto& xs = std::get<CpuStorage>(input->storage_);
        if (input->dtype_ == Dtype::F32) {
            bilinear_forward_cpu<float>(
                reinterpret_cast<const float*>(xs.ptr.get()),
                reinterpret_cast<float*>(out_cpu.ptr.get()),
                N, C, H_in, W_in, H_out, W_out, align_corners);
        } else if (input->dtype_ == Dtype::F64) {
            bilinear_forward_cpu<double>(
                reinterpret_cast<const double*>(xs.ptr.get()),
                reinterpret_cast<double*>(out_cpu.ptr.get()),
                N, C, H_in, W_in, H_out, W_out, align_corners);
        } else {
            throw NotImplementedError(
                "interpolate_bilinear: dtype must be F32/F64");
        }
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                             out_shape, input->dtype_,
                                             input->device_, false);
    if (!GradMode::is_enabled() || !input->requires_grad_) return out;
    auto x_edge = detail::ensure_grad_fn(input);
    auto bwd = std::make_shared<InterpolateBilinearBackward>();
    bwd->input_shapes_ = {input->shape_};
    bwd->out_shape_    = out_shape;
    bwd->dtype_        = input->dtype_;
    bwd->device_       = input->device_;
    bwd->input_tensors_ = {input};
    bwd->H_in_ = H_in; bwd->W_in_ = W_in;
    bwd->H_out_ = H_out; bwd->W_out_ = W_out;
    bwd->align_corners_ = align_corners;
    bwd->orig_shape_ = input->shape_;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions({input->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> InterpolateBilinearBackward::apply(Storage grad_out) {
    const int N = static_cast<int>(orig_shape_[0]);
    const int C = static_cast<int>(orig_shape_[1]);
    if (device_ == Device::GPU) {
        // Native MLX scatter_add path: each output cell distributes its
        // gradient to four corner positions of the input with weights
        // (1-dy)(1-dx), (1-dy)dx, dy(1-dx), dy*dx. We implement this as
        // four scatter_add calls on a zero base.
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);

        auto ys = build_src_coords(H_in_, H_out_, align_corners_, mlx_dt);
        auto xs = build_src_coords(W_in_, W_out_, align_corners_, mlx_dt);
        auto zero = mlx_scalar(0.0, mlx_dt);
        auto one = mlx_scalar(1.0, mlx_dt);
        auto Hm1 = mlx_scalar(H_in_ - 1, mlx_dt);
        auto Wm1 = mlx_scalar(W_in_ - 1, mlx_dt);
        ys = ::mlx::core::clip(ys, std::optional<::mlx::core::array>(zero),
                                  std::optional<::mlx::core::array>(Hm1));
        xs = ::mlx::core::clip(xs, std::optional<::mlx::core::array>(zero),
                                  std::optional<::mlx::core::array>(Wm1));
        auto y0_f = ::mlx::core::floor(ys);
        auto x0_f = ::mlx::core::floor(xs);
        auto dy_1d = ::mlx::core::subtract(ys, y0_f);
        auto dx_1d = ::mlx::core::subtract(xs, x0_f);
        auto y0 = ::mlx::core::astype(y0_f, ::mlx::core::int32);
        auto x0 = ::mlx::core::astype(x0_f, ::mlx::core::int32);
        auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in_ - 1),
                                           ::mlx::core::int32);
        auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in_ - 1),
                                           ::mlx::core::int32);
        auto y1 = ::mlx::core::minimum(::mlx::core::add(y0,
                            ::mlx::core::astype(::mlx::core::array(1),
                                                 ::mlx::core::int32)),
                          Hm1_i);
        auto x1 = ::mlx::core::minimum(::mlx::core::add(x0,
                            ::mlx::core::astype(::mlx::core::array(1),
                                                 ::mlx::core::int32)),
                          Wm1_i);

        // Broadcast dy/dx and y0/y1/x0/x1 to (N, C, H_out, W_out).
        ::mlx::core::Shape full_idx_shape{N, C, H_out_, W_out_};
        auto reshape_y = [&](const ::mlx::core::array& a) {
            return ::mlx::core::broadcast_to(
                ::mlx::core::reshape(a, {1, 1, H_out_, 1}), full_idx_shape);
        };
        auto reshape_x = [&](const ::mlx::core::array& a) {
            return ::mlx::core::broadcast_to(
                ::mlx::core::reshape(a, {1, 1, 1, W_out_}), full_idx_shape);
        };
        auto y0_b = reshape_y(y0);
        auto y1_b = reshape_y(y1);
        auto x0_b = reshape_x(x0);
        auto x1_b = reshape_x(x1);
        auto dy_b = reshape_y(dy_1d);
        auto dx_b = reshape_x(dx_1d);
        auto wy0 = ::mlx::core::subtract(one, dy_b);
        auto wy1 = dy_b;
        auto wx0 = ::mlx::core::subtract(one, dx_b);
        auto wx1 = dx_b;

        auto n_idx = ::mlx::core::broadcast_to(::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int32),
            {N, 1, 1, 1}), full_idx_shape);
        auto c_idx = ::mlx::core::broadcast_to(::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int32),
            {1, C, 1, 1}), full_idx_shape);

        ::mlx::core::Shape upd_shape{N, C, H_out_, W_out_, 1, 1, 1, 1};
        ::mlx::core::Shape base_shape{N, C, H_in_, W_in_};
        auto base = ::mlx::core::zeros(base_shape, mlx_dt);
        std::vector<int> axes_v{0, 1, 2, 3};

        auto scatter_one = [&](const ::mlx::core::array& y_idx,
                                const ::mlx::core::array& x_idx,
                                const ::mlx::core::array& weight) {
            std::vector<::mlx::core::array> idxs{n_idx, c_idx, y_idx, x_idx};
            auto upd = ::mlx::core::reshape(
                ::mlx::core::multiply(*gg.arr, weight), upd_shape);
            base = ::mlx::core::scatter_add(base, idxs, upd, axes_v);
        };
        scatter_one(y0_b, x0_b, ::mlx::core::multiply(wy0, wx0));
        scatter_one(y0_b, x1_b, ::mlx::core::multiply(wy0, wx1));
        scatter_one(y1_b, x0_b, ::mlx::core::multiply(wy1, wx0));
        scatter_one(y1_b, x1_b, ::mlx::core::multiply(wy1, wx1));

        return {Storage{gpu::wrap_mlx_array(std::move(base), dtype_)}};
    }
    auto dx_cpu = allocate_size(
        static_cast<std::size_t>(N) * C * H_in_ * W_in_, dtype_);
    const auto& go = std::get<CpuStorage>(grad_out);
    if (dtype_ == Dtype::F32) {
        bilinear_backward_cpu<float>(
            reinterpret_cast<const float*>(go.ptr.get()),
            reinterpret_cast<float*>(dx_cpu.ptr.get()),
            N, C, H_in_, W_in_, H_out_, W_out_, align_corners_);
    } else if (dtype_ == Dtype::F64) {
        bilinear_backward_cpu<double>(
            reinterpret_cast<const double*>(go.ptr.get()),
            reinterpret_cast<double*>(dx_cpu.ptr.get()),
            N, C, H_in_, W_in_, H_out_, W_out_, align_corners_);
    } else {
        throw NotImplementedError(
            "interpolate_bilinear backward: dtype not supported");
    }
    return {Storage{std::move(dx_cpu)}};
}

TensorImplPtr interpolate_bilinear_op(const TensorImplPtr& input,
                                        int H_out, int W_out, bool align_corners) {
    return InterpolateBilinearBackward::forward(input, H_out, W_out, align_corners);
}
LUCID_REGISTER_OP(InterpolateBilinearBackward)

// =====================================================================
// interpolate_trilinear (5-D)
// =====================================================================

const OpSchema InterpolateTrilinearBackward::schema_v1{
    "interpolate_trilinear", 1, AmpPolicy::Promote, true};

namespace {

template <typename T>
void trilinear_forward_cpu(const T* xp, T* op,
                            int N, int C, int D_in, int H_in, int W_in,
                            int D_out, int H_out, int W_out, bool align) {
    const std::size_t in_chan  = static_cast<std::size_t>(D_in)  * H_in  * W_in;
    const std::size_t out_chan = static_cast<std::size_t>(D_out) * H_out * W_out;
    auto coord = [&](int o, int in_n, int out_n) {
        T v = src_coord<T>(o, in_n, out_n, align);
        if (v < T{0}) v = T{0};
        if (v > static_cast<T>(in_n - 1)) v = static_cast<T>(in_n - 1);
        return v;
    };
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
        const T* base = xp + (n * C + c) * in_chan;
        T* out_base = op + (n * C + c) * out_chan;
        for (int d = 0; d < D_out; ++d) {
            T iz = coord(d, D_in, D_out);
            int z0 = static_cast<int>(std::floor(iz));
            int z1 = std::min(z0 + 1, D_in - 1);
            const T dz = iz - static_cast<T>(z0);
            for (int h = 0; h < H_out; ++h) {
                T iy = coord(h, H_in, H_out);
                int y0 = static_cast<int>(std::floor(iy));
                int y1 = std::min(y0 + 1, H_in - 1);
                const T dy = iy - static_cast<T>(y0);
                for (int w = 0; w < W_out; ++w) {
                    T ix = coord(w, W_in, W_out);
                    int x0 = static_cast<int>(std::floor(ix));
                    int x1 = std::min(x0 + 1, W_in - 1);
                    const T dx = ix - static_cast<T>(x0);
                    auto idx = [&](int z, int y, int x) -> std::size_t {
                        return static_cast<std::size_t>(z) * H_in * W_in
                              + static_cast<std::size_t>(y) * W_in + x;
                    };
                    const T c000 = base[idx(z0, y0, x0)];
                    const T c001 = base[idx(z0, y0, x1)];
                    const T c010 = base[idx(z0, y1, x0)];
                    const T c011 = base[idx(z0, y1, x1)];
                    const T c100 = base[idx(z1, y0, x0)];
                    const T c101 = base[idx(z1, y0, x1)];
                    const T c110 = base[idx(z1, y1, x0)];
                    const T c111 = base[idx(z1, y1, x1)];
                    const T c00 = c000 * (T{1} - dx) + c001 * dx;
                    const T c01 = c010 * (T{1} - dx) + c011 * dx;
                    const T c10 = c100 * (T{1} - dx) + c101 * dx;
                    const T c11 = c110 * (T{1} - dx) + c111 * dx;
                    const T c0 = c00 * (T{1} - dy) + c01 * dy;
                    const T c1 = c10 * (T{1} - dy) + c11 * dy;
                    out_base[d * H_out * W_out + h * W_out + w] =
                        c0 * (T{1} - dz) + c1 * dz;
                }
            }
        }
    }
}

template <typename T>
void trilinear_backward_cpu(const T* go_p, T* dx_p,
                             int N, int C, int D_in, int H_in, int W_in,
                             int D_out, int H_out, int W_out, bool align) {
    const std::size_t in_chan  = static_cast<std::size_t>(D_in)  * H_in  * W_in;
    const std::size_t out_chan = static_cast<std::size_t>(D_out) * H_out * W_out;
    std::memset(dx_p, 0, sizeof(T) * static_cast<std::size_t>(N) * C * in_chan);
    auto coord = [&](int o, int in_n, int out_n) {
        T v = src_coord<T>(o, in_n, out_n, align);
        if (v < T{0}) v = T{0};
        if (v > static_cast<T>(in_n - 1)) v = static_cast<T>(in_n - 1);
        return v;
    };
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
        T* base = dx_p + (n * C + c) * in_chan;
        const T* go_base = go_p + (n * C + c) * out_chan;
        for (int d = 0; d < D_out; ++d) {
            T iz = coord(d, D_in, D_out);
            int z0 = static_cast<int>(std::floor(iz));
            int z1 = std::min(z0 + 1, D_in - 1);
            const T dz = iz - static_cast<T>(z0);
            for (int h = 0; h < H_out; ++h) {
                T iy = coord(h, H_in, H_out);
                int y0 = static_cast<int>(std::floor(iy));
                int y1 = std::min(y0 + 1, H_in - 1);
                const T dy = iy - static_cast<T>(y0);
                for (int w = 0; w < W_out; ++w) {
                    T ix = coord(w, W_in, W_out);
                    int x0 = static_cast<int>(std::floor(ix));
                    int x1 = std::min(x0 + 1, W_in - 1);
                    const T dx = ix - static_cast<T>(x0);
                    const T g = go_base[d * H_out * W_out + h * W_out + w];
                    auto add = [&](int z, int y, int x, T weight) {
                        base[static_cast<std::size_t>(z) * H_in * W_in
                              + static_cast<std::size_t>(y) * W_in + x]
                          += g * weight;
                    };
                    const T omdx = T{1} - dx, omdy = T{1} - dy, omdz = T{1} - dz;
                    add(z0, y0, x0, omdz * omdy * omdx);
                    add(z0, y0, x1, omdz * omdy * dx);
                    add(z0, y1, x0, omdz * dy * omdx);
                    add(z0, y1, x1, omdz * dy * dx);
                    add(z1, y0, x0, dz * omdy * omdx);
                    add(z1, y0, x1, dz * omdy * dx);
                    add(z1, y1, x0, dz * dy * omdx);
                    add(z1, y1, x1, dz * dy * dx);
                }
            }
        }
    }
}

}  // namespace

TensorImplPtr InterpolateTrilinearBackward::forward(const TensorImplPtr& input,
                                                      int D_out, int H_out, int W_out,
                                                      bool align_corners) {
    if (!input) throw LucidError("interpolate_trilinear: null input");
    if (input->shape_.size() != 5)
        throw ShapeMismatch(input->shape_, Shape{},
                             "interpolate_trilinear: input must be 5-D");
    const int N = static_cast<int>(input->shape_[0]);
    const int C = static_cast<int>(input->shape_[1]);
    const int D_in = static_cast<int>(input->shape_[2]);
    const int H_in = static_cast<int>(input->shape_[3]);
    const int W_in = static_cast<int>(input->shape_[4]);
    Shape out_shape{N, C, D_out, H_out, W_out};
    OpScope scope{schema_v1.name, input->device_, input->dtype_, out_shape};

    Storage out_storage;
    if (input->device_ == Device::GPU) {
        // Native MLX forward via 8-corner gather + weighted sum.
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype_);

        auto zs = build_src_coords(D_in, D_out, align_corners, mlx_dt);
        auto ys = build_src_coords(H_in, H_out, align_corners, mlx_dt);
        auto xs = build_src_coords(W_in, W_out, align_corners, mlx_dt);
        auto zero_f = mlx_scalar(0.0, mlx_dt);
        auto Dm1 = mlx_scalar(D_in - 1, mlx_dt);
        auto Hm1 = mlx_scalar(H_in - 1, mlx_dt);
        auto Wm1 = mlx_scalar(W_in - 1, mlx_dt);
        zs = ::mlx::core::clip(zs, std::optional<::mlx::core::array>(zero_f),
                                  std::optional<::mlx::core::array>(Dm1));
        ys = ::mlx::core::clip(ys, std::optional<::mlx::core::array>(zero_f),
                                  std::optional<::mlx::core::array>(Hm1));
        xs = ::mlx::core::clip(xs, std::optional<::mlx::core::array>(zero_f),
                                  std::optional<::mlx::core::array>(Wm1));
        auto z0_f = ::mlx::core::floor(zs);
        auto y0_f = ::mlx::core::floor(ys);
        auto x0_f = ::mlx::core::floor(xs);
        auto dz = ::mlx::core::subtract(zs, z0_f);
        auto dy = ::mlx::core::subtract(ys, y0_f);
        auto dx = ::mlx::core::subtract(xs, x0_f);
        auto z0 = ::mlx::core::astype(z0_f, ::mlx::core::int64);
        auto y0 = ::mlx::core::astype(y0_f, ::mlx::core::int64);
        auto x0 = ::mlx::core::astype(x0_f, ::mlx::core::int64);
        auto z1 = ::mlx::core::minimum(::mlx::core::add(z0,
                ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64)),
                ::mlx::core::astype(::mlx::core::array(D_in - 1), ::mlx::core::int64));
        auto y1 = ::mlx::core::minimum(::mlx::core::add(y0,
                ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64)),
                ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64));
        auto x1 = ::mlx::core::minimum(::mlx::core::add(x0,
                ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64)),
                ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64));

        auto gather_corner = [&](const ::mlx::core::array& zi,
                                   const ::mlx::core::array& yi,
                                   const ::mlx::core::array& xi) {
            auto n_idx = ::mlx::core::reshape(
                ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int64),
                {N, 1, 1, 1, 1});
            auto c_idx = ::mlx::core::reshape(
                ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int64),
                {1, C, 1, 1, 1});
            auto z_b = ::mlx::core::reshape(zi, {1, 1, D_out, 1, 1});
            auto y_b = ::mlx::core::reshape(yi, {1, 1, 1, H_out, 1});
            auto x_b = ::mlx::core::reshape(xi, {1, 1, 1, 1, W_out});
            auto sN = ::mlx::core::astype(::mlx::core::array(C * D_in * H_in * W_in),
                                            ::mlx::core::int64);
            auto sC = ::mlx::core::astype(::mlx::core::array(D_in * H_in * W_in),
                                            ::mlx::core::int64);
            auto sD = ::mlx::core::astype(::mlx::core::array(H_in * W_in),
                                            ::mlx::core::int64);
            auto sH = ::mlx::core::astype(::mlx::core::array(W_in),
                                            ::mlx::core::int64);
            auto flat = ::mlx::core::add(
                ::mlx::core::add(
                    ::mlx::core::add(::mlx::core::multiply(n_idx, sN),
                                      ::mlx::core::multiply(c_idx, sC)),
                    ::mlx::core::add(::mlx::core::multiply(z_b, sD),
                                      ::mlx::core::multiply(y_b, sH))),
                x_b);
            auto flat_b = ::mlx::core::broadcast_to(flat,
                ::mlx::core::Shape{N, C, D_out, H_out, W_out});
            auto in_flat = ::mlx::core::reshape(*gx.arr,
                ::mlx::core::Shape{N * C * D_in * H_in * W_in});
            return ::mlx::core::take(in_flat, flat_b);
        };

        auto reshape_z = [&](const ::mlx::core::array& a) {
            return ::mlx::core::broadcast_to(
                ::mlx::core::reshape(a, {1, 1, D_out, 1, 1}),
                ::mlx::core::Shape{N, C, D_out, H_out, W_out});
        };
        auto reshape_y = [&](const ::mlx::core::array& a) {
            return ::mlx::core::broadcast_to(
                ::mlx::core::reshape(a, {1, 1, 1, H_out, 1}),
                ::mlx::core::Shape{N, C, D_out, H_out, W_out});
        };
        auto reshape_x = [&](const ::mlx::core::array& a) {
            return ::mlx::core::broadcast_to(
                ::mlx::core::reshape(a, {1, 1, 1, 1, W_out}),
                ::mlx::core::Shape{N, C, D_out, H_out, W_out});
        };
        auto one = mlx_scalar(1.0, mlx_dt);
        auto wz0 = ::mlx::core::subtract(one, reshape_z(dz));
        auto wz1 = reshape_z(dz);
        auto wy0 = ::mlx::core::subtract(one, reshape_y(dy));
        auto wy1 = reshape_y(dy);
        auto wx0 = ::mlx::core::subtract(one, reshape_x(dx));
        auto wx1 = reshape_x(dx);

        auto c000 = ::mlx::core::multiply(gather_corner(z0, y0, x0),
                        ::mlx::core::multiply(wz0,
                            ::mlx::core::multiply(wy0, wx0)));
        auto c001 = ::mlx::core::multiply(gather_corner(z0, y0, x1),
                        ::mlx::core::multiply(wz0,
                            ::mlx::core::multiply(wy0, wx1)));
        auto c010 = ::mlx::core::multiply(gather_corner(z0, y1, x0),
                        ::mlx::core::multiply(wz0,
                            ::mlx::core::multiply(wy1, wx0)));
        auto c011 = ::mlx::core::multiply(gather_corner(z0, y1, x1),
                        ::mlx::core::multiply(wz0,
                            ::mlx::core::multiply(wy1, wx1)));
        auto c100 = ::mlx::core::multiply(gather_corner(z1, y0, x0),
                        ::mlx::core::multiply(wz1,
                            ::mlx::core::multiply(wy0, wx0)));
        auto c101 = ::mlx::core::multiply(gather_corner(z1, y0, x1),
                        ::mlx::core::multiply(wz1,
                            ::mlx::core::multiply(wy0, wx1)));
        auto c110 = ::mlx::core::multiply(gather_corner(z1, y1, x0),
                        ::mlx::core::multiply(wz1,
                            ::mlx::core::multiply(wy1, wx0)));
        auto c111 = ::mlx::core::multiply(gather_corner(z1, y1, x1),
                        ::mlx::core::multiply(wz1,
                            ::mlx::core::multiply(wy1, wx1)));
        auto out = ::mlx::core::add(
            ::mlx::core::add(::mlx::core::add(c000, c001),
                              ::mlx::core::add(c010, c011)),
            ::mlx::core::add(::mlx::core::add(c100, c101),
                              ::mlx::core::add(c110, c111)));
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out),
                                                    input->dtype_)};
    } else {
        auto out_cpu = allocate_size(
            static_cast<std::size_t>(N) * C * D_out * H_out * W_out,
            input->dtype_);
        const auto& xs = std::get<CpuStorage>(input->storage_);
        if (input->dtype_ == Dtype::F32)
            trilinear_forward_cpu<float>(
                reinterpret_cast<const float*>(xs.ptr.get()),
                reinterpret_cast<float*>(out_cpu.ptr.get()),
                N, C, D_in, H_in, W_in, D_out, H_out, W_out, align_corners);
        else if (input->dtype_ == Dtype::F64)
            trilinear_forward_cpu<double>(
                reinterpret_cast<const double*>(xs.ptr.get()),
                reinterpret_cast<double*>(out_cpu.ptr.get()),
                N, C, D_in, H_in, W_in, D_out, H_out, W_out, align_corners);
        else
            throw NotImplementedError("interpolate_trilinear: dtype must be F32/F64");
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                             out_shape, input->dtype_,
                                             input->device_, false);
    if (!GradMode::is_enabled() || !input->requires_grad_) return out;
    auto x_edge = detail::ensure_grad_fn(input);
    auto bwd = std::make_shared<InterpolateTrilinearBackward>();
    bwd->input_shapes_ = {input->shape_};
    bwd->out_shape_    = out_shape;
    bwd->dtype_        = input->dtype_;
    bwd->device_       = input->device_;
    bwd->input_tensors_ = {input};
    bwd->D_in_ = D_in; bwd->H_in_ = H_in; bwd->W_in_ = W_in;
    bwd->D_out_ = D_out; bwd->H_out_ = H_out; bwd->W_out_ = W_out;
    bwd->align_corners_ = align_corners;
    bwd->orig_shape_ = input->shape_;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions({input->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> InterpolateTrilinearBackward::apply(Storage grad_out) {
    const int N = static_cast<int>(orig_shape_[0]);
    const int C = static_cast<int>(orig_shape_[1]);

    if (device_ == Device::GPU) {
        // Native MLX scatter_add — 8 corners per output cell.
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);

        auto zs = build_src_coords(D_in_, D_out_, align_corners_, mlx_dt);
        auto ys = build_src_coords(H_in_, H_out_, align_corners_, mlx_dt);
        auto xs = build_src_coords(W_in_, W_out_, align_corners_, mlx_dt);
        auto zero = mlx_scalar(0.0, mlx_dt);
        auto one = mlx_scalar(1.0, mlx_dt);
        auto Dm1 = mlx_scalar(D_in_ - 1, mlx_dt);
        auto Hm1 = mlx_scalar(H_in_ - 1, mlx_dt);
        auto Wm1 = mlx_scalar(W_in_ - 1, mlx_dt);
        zs = ::mlx::core::clip(zs, std::optional<::mlx::core::array>(zero),
                                  std::optional<::mlx::core::array>(Dm1));
        ys = ::mlx::core::clip(ys, std::optional<::mlx::core::array>(zero),
                                  std::optional<::mlx::core::array>(Hm1));
        xs = ::mlx::core::clip(xs, std::optional<::mlx::core::array>(zero),
                                  std::optional<::mlx::core::array>(Wm1));
        auto z0_f = ::mlx::core::floor(zs);
        auto y0_f = ::mlx::core::floor(ys);
        auto x0_f = ::mlx::core::floor(xs);
        auto dz1 = ::mlx::core::subtract(zs, z0_f);
        auto dy1 = ::mlx::core::subtract(ys, y0_f);
        auto dx1 = ::mlx::core::subtract(xs, x0_f);
        auto z0 = ::mlx::core::astype(z0_f, ::mlx::core::int32);
        auto y0 = ::mlx::core::astype(y0_f, ::mlx::core::int32);
        auto x0 = ::mlx::core::astype(x0_f, ::mlx::core::int32);
        auto Dm1_i = ::mlx::core::astype(::mlx::core::array(D_in_ - 1),
                                           ::mlx::core::int32);
        auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in_ - 1),
                                           ::mlx::core::int32);
        auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in_ - 1),
                                           ::mlx::core::int32);
        auto z1 = ::mlx::core::minimum(::mlx::core::add(z0,
                ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int32)),
                Dm1_i);
        auto y1 = ::mlx::core::minimum(::mlx::core::add(y0,
                ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int32)),
                Hm1_i);
        auto x1 = ::mlx::core::minimum(::mlx::core::add(x0,
                ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int32)),
                Wm1_i);

        ::mlx::core::Shape full{N, C, D_out_, H_out_, W_out_};
        auto reshape_z = [&](const ::mlx::core::array& a) {
            return ::mlx::core::broadcast_to(
                ::mlx::core::reshape(a, {1, 1, D_out_, 1, 1}), full);
        };
        auto reshape_y = [&](const ::mlx::core::array& a) {
            return ::mlx::core::broadcast_to(
                ::mlx::core::reshape(a, {1, 1, 1, H_out_, 1}), full);
        };
        auto reshape_x = [&](const ::mlx::core::array& a) {
            return ::mlx::core::broadcast_to(
                ::mlx::core::reshape(a, {1, 1, 1, 1, W_out_}), full);
        };
        auto z0_b = reshape_z(z0);
        auto z1_b = reshape_z(z1);
        auto y0_b = reshape_y(y0);
        auto y1_b = reshape_y(y1);
        auto x0_b = reshape_x(x0);
        auto x1_b = reshape_x(x1);
        auto dz_b = reshape_z(dz1);
        auto dy_b = reshape_y(dy1);
        auto dx_b = reshape_x(dx1);
        auto wz0 = ::mlx::core::subtract(one, dz_b);
        auto wz1 = dz_b;
        auto wy0 = ::mlx::core::subtract(one, dy_b);
        auto wy1 = dy_b;
        auto wx0 = ::mlx::core::subtract(one, dx_b);
        auto wx1 = dx_b;

        auto n_idx = ::mlx::core::broadcast_to(::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int32),
            {N, 1, 1, 1, 1}), full);
        auto c_idx = ::mlx::core::broadcast_to(::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int32),
            {1, C, 1, 1, 1}), full);

        ::mlx::core::Shape upd_shape{N, C, D_out_, H_out_, W_out_, 1, 1, 1, 1, 1};
        ::mlx::core::Shape base_shape{N, C, D_in_, H_in_, W_in_};
        auto base = ::mlx::core::zeros(base_shape, mlx_dt);
        std::vector<int> axes_v{0, 1, 2, 3, 4};

        auto scatter_one = [&](const ::mlx::core::array& zi,
                                const ::mlx::core::array& yi,
                                const ::mlx::core::array& xi,
                                const ::mlx::core::array& weight) {
            std::vector<::mlx::core::array> idxs{n_idx, c_idx, zi, yi, xi};
            auto upd = ::mlx::core::reshape(
                ::mlx::core::multiply(*gg.arr, weight), upd_shape);
            base = ::mlx::core::scatter_add(base, idxs, upd, axes_v);
        };
        scatter_one(z0_b, y0_b, x0_b,
                     ::mlx::core::multiply(wz0, ::mlx::core::multiply(wy0, wx0)));
        scatter_one(z0_b, y0_b, x1_b,
                     ::mlx::core::multiply(wz0, ::mlx::core::multiply(wy0, wx1)));
        scatter_one(z0_b, y1_b, x0_b,
                     ::mlx::core::multiply(wz0, ::mlx::core::multiply(wy1, wx0)));
        scatter_one(z0_b, y1_b, x1_b,
                     ::mlx::core::multiply(wz0, ::mlx::core::multiply(wy1, wx1)));
        scatter_one(z1_b, y0_b, x0_b,
                     ::mlx::core::multiply(wz1, ::mlx::core::multiply(wy0, wx0)));
        scatter_one(z1_b, y0_b, x1_b,
                     ::mlx::core::multiply(wz1, ::mlx::core::multiply(wy0, wx1)));
        scatter_one(z1_b, y1_b, x0_b,
                     ::mlx::core::multiply(wz1, ::mlx::core::multiply(wy1, wx0)));
        scatter_one(z1_b, y1_b, x1_b,
                     ::mlx::core::multiply(wz1, ::mlx::core::multiply(wy1, wx1)));

        return {Storage{gpu::wrap_mlx_array(std::move(base), dtype_)}};
    }

    auto dx_cpu = allocate_size(
        static_cast<std::size_t>(N) * C * D_in_ * H_in_ * W_in_, dtype_);
    const auto& go = std::get<CpuStorage>(grad_out);
    if (dtype_ == Dtype::F32)
        trilinear_backward_cpu<float>(
            reinterpret_cast<const float*>(go.ptr.get()),
            reinterpret_cast<float*>(dx_cpu.ptr.get()),
            N, C, D_in_, H_in_, W_in_, D_out_, H_out_, W_out_, align_corners_);
    else if (dtype_ == Dtype::F64)
        trilinear_backward_cpu<double>(
            reinterpret_cast<const double*>(go.ptr.get()),
            reinterpret_cast<double*>(dx_cpu.ptr.get()),
            N, C, D_in_, H_in_, W_in_, D_out_, H_out_, W_out_, align_corners_);
    else
        throw NotImplementedError("interpolate_trilinear backward: dtype not supported");
    return {Storage{std::move(dx_cpu)}};
}

TensorImplPtr interpolate_trilinear_op(const TensorImplPtr& input,
                                          int D_out, int H_out, int W_out,
                                          bool align_corners) {
    return InterpolateTrilinearBackward::forward(input, D_out, H_out, W_out,
                                                   align_corners);
}
LUCID_REGISTER_OP(InterpolateTrilinearBackward)

// =====================================================================
// interpolate_nearest (no autograd: indices are non-differentiable).
// =====================================================================

namespace {

template <typename T>
void nearest2d_cpu(const T* xp, T* op, int N, int C, int H_in, int W_in,
                    int H_out, int W_out) {
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
        const T* base = xp + ((n * C + c) * H_in) * W_in;
        T* out_base = op + ((n * C + c) * H_out) * W_out;
        for (int h = 0; h < H_out; ++h) {
            int yh = static_cast<int>(std::floor(
                static_cast<double>(h) * H_in / H_out));
            yh = std::clamp(yh, 0, H_in - 1);
            for (int w = 0; w < W_out; ++w) {
                int xw = static_cast<int>(std::floor(
                    static_cast<double>(w) * W_in / W_out));
                xw = std::clamp(xw, 0, W_in - 1);
                out_base[h * W_out + w] = base[yh * W_in + xw];
            }
        }
    }
}

template <typename T>
void nearest3d_cpu(const T* xp, T* op,
                    int N, int C, int D_in, int H_in, int W_in,
                    int D_out, int H_out, int W_out) {
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
        const T* base = xp + (n * C + c) * D_in * H_in * W_in;
        T* out_base = op + (n * C + c) * D_out * H_out * W_out;
        for (int d = 0; d < D_out; ++d) {
            int dz = std::clamp(static_cast<int>(std::floor(
                static_cast<double>(d) * D_in / D_out)), 0, D_in - 1);
            for (int h = 0; h < H_out; ++h) {
                int yh = std::clamp(static_cast<int>(std::floor(
                    static_cast<double>(h) * H_in / H_out)), 0, H_in - 1);
                for (int w = 0; w < W_out; ++w) {
                    int xw = std::clamp(static_cast<int>(std::floor(
                        static_cast<double>(w) * W_in / W_out)), 0, W_in - 1);
                    out_base[(d * H_out + h) * W_out + w] =
                        base[(dz * H_in + yh) * W_in + xw];
                }
            }
        }
    }
}

}  // namespace

TensorImplPtr interpolate_nearest_2d_op(const TensorImplPtr& input,
                                          int H_out, int W_out) {
    if (!input) throw LucidError("interpolate_nearest: null input");
    if (input->shape_.size() != 4)
        throw ShapeMismatch(input->shape_, Shape{},
                             "interpolate_nearest: 4-D input required");
    const int N = static_cast<int>(input->shape_[0]);
    const int C = static_cast<int>(input->shape_[1]);
    const int H_in = static_cast<int>(input->shape_[2]);
    const int W_in = static_cast<int>(input->shape_[3]);
    Shape out_shape{N, C, H_out, W_out};
    OpScope scope{"interpolate_nearest_2d", input->device_,
                   input->dtype_, out_shape};
    Storage out_storage;
    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        // Build int index arrays: yh[H_out], xw[W_out].
        ::mlx::core::array yh = ::mlx::core::astype(
            ::mlx::core::arange(0, H_out, 1), ::mlx::core::float32);
        ::mlx::core::array xw = ::mlx::core::astype(
            ::mlx::core::arange(0, W_out, 1), ::mlx::core::float32);
        auto sH = ::mlx::core::astype(::mlx::core::array(
            static_cast<float>(H_in) / static_cast<float>(H_out)), ::mlx::core::float32);
        auto sW = ::mlx::core::astype(::mlx::core::array(
            static_cast<float>(W_in) / static_cast<float>(W_out)), ::mlx::core::float32);
        yh = ::mlx::core::floor(::mlx::core::multiply(yh, sH));
        xw = ::mlx::core::floor(::mlx::core::multiply(xw, sW));
        auto Hm1 = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::float32);
        auto Wm1 = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::float32);
        auto zero_f = ::mlx::core::astype(::mlx::core::array(0.0f), ::mlx::core::float32);
        yh = ::mlx::core::clip(yh,
            std::optional<::mlx::core::array>(zero_f),
            std::optional<::mlx::core::array>(Hm1));
        xw = ::mlx::core::clip(xw,
            std::optional<::mlx::core::array>(zero_f),
            std::optional<::mlx::core::array>(Wm1));
        auto y_int = ::mlx::core::astype(yh, ::mlx::core::int64);
        auto x_int = ::mlx::core::astype(xw, ::mlx::core::int64);
        auto y_2d = ::mlx::core::broadcast_to(::mlx::core::reshape(y_int, {H_out, 1}),
                                                {H_out, W_out});
        auto x_2d = ::mlx::core::broadcast_to(::mlx::core::reshape(x_int, {1, W_out}),
                                                {H_out, W_out});
        auto out = gather_2d_corner(*gx.arr, y_2d, x_2d,
                                       N, C, H_in, W_in, H_out, W_out);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), input->dtype_)};
    } else {
        auto out_cpu = allocate_size(
            static_cast<std::size_t>(N) * C * H_out * W_out, input->dtype_);
        const auto& xs = std::get<CpuStorage>(input->storage_);
        if (input->dtype_ == Dtype::F32)
            nearest2d_cpu<float>(
                reinterpret_cast<const float*>(xs.ptr.get()),
                reinterpret_cast<float*>(out_cpu.ptr.get()),
                N, C, H_in, W_in, H_out, W_out);
        else if (input->dtype_ == Dtype::F64)
            nearest2d_cpu<double>(
                reinterpret_cast<const double*>(xs.ptr.get()),
                reinterpret_cast<double*>(out_cpu.ptr.get()),
                N, C, H_in, W_in, H_out, W_out);
        else
            throw NotImplementedError("interpolate_nearest: dtype must be F32/F64");
        out_storage = Storage{std::move(out_cpu)};
    }
    return std::make_shared<TensorImpl>(std::move(out_storage),
                                         out_shape, input->dtype_,
                                         input->device_, false);
}

TensorImplPtr interpolate_nearest_3d_op(const TensorImplPtr& input,
                                          int D_out, int H_out, int W_out) {
    if (!input) throw LucidError("interpolate_nearest_3d: null input");
    if (input->shape_.size() != 5)
        throw ShapeMismatch(input->shape_, Shape{},
                             "interpolate_nearest_3d: 5-D input required");
    const int N = static_cast<int>(input->shape_[0]);
    const int C = static_cast<int>(input->shape_[1]);
    const int D_in = static_cast<int>(input->shape_[2]);
    const int H_in = static_cast<int>(input->shape_[3]);
    const int W_in = static_cast<int>(input->shape_[4]);
    Shape out_shape{N, C, D_out, H_out, W_out};
    OpScope scope{"interpolate_nearest_3d", input->device_,
                   input->dtype_, out_shape};
    Storage out_storage;
    if (input->device_ == Device::GPU) {
        // Native MLX 5-D gather: build per-axis index arrays, compute the
        // flat input offset, take from a flattened input buffer.
        const auto& gx = std::get<GpuStorage>(input->storage_);
        auto build_idx = [&](int in_dim, int out_dim) {
            auto idx = ::mlx::core::astype(::mlx::core::arange(0, out_dim, 1),
                                            ::mlx::core::float32);
            auto scale = mlx_scalar(static_cast<double>(in_dim) /
                                      static_cast<double>(out_dim),
                                    ::mlx::core::float32);
            auto v = ::mlx::core::floor(::mlx::core::multiply(idx, scale));
            auto zero_f = mlx_scalar(0.0, ::mlx::core::float32);
            auto cap = mlx_scalar(in_dim - 1, ::mlx::core::float32);
            v = ::mlx::core::clip(v,
                std::optional<::mlx::core::array>(zero_f),
                std::optional<::mlx::core::array>(cap));
            return ::mlx::core::astype(v, ::mlx::core::int64);
        };
        auto d_idx = build_idx(D_in, D_out);
        auto h_idx = build_idx(H_in, H_out);
        auto w_idx = build_idx(W_in, W_out);

        auto n_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int64),
            {N, 1, 1, 1, 1});
        auto c_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int64),
            {1, C, 1, 1, 1});
        auto d_b = ::mlx::core::reshape(d_idx, {1, 1, D_out, 1, 1});
        auto h_b = ::mlx::core::reshape(h_idx, {1, 1, 1, H_out, 1});
        auto w_b = ::mlx::core::reshape(w_idx, {1, 1, 1, 1, W_out});
        auto sN = ::mlx::core::astype(::mlx::core::array(C * D_in * H_in * W_in),
                                        ::mlx::core::int64);
        auto sC = ::mlx::core::astype(::mlx::core::array(D_in * H_in * W_in),
                                        ::mlx::core::int64);
        auto sD = ::mlx::core::astype(::mlx::core::array(H_in * W_in),
                                        ::mlx::core::int64);
        auto sH = ::mlx::core::astype(::mlx::core::array(W_in),
                                        ::mlx::core::int64);
        auto flat = ::mlx::core::add(
            ::mlx::core::add(
                ::mlx::core::add(::mlx::core::multiply(n_idx, sN),
                                  ::mlx::core::multiply(c_idx, sC)),
                ::mlx::core::add(::mlx::core::multiply(d_b, sD),
                                  ::mlx::core::multiply(h_b, sH))),
            w_b);
        auto flat_b = ::mlx::core::broadcast_to(flat,
            ::mlx::core::Shape{N, C, D_out, H_out, W_out});
        auto in_flat = ::mlx::core::reshape(*gx.arr,
            ::mlx::core::Shape{N * C * D_in * H_in * W_in});
        auto out = ::mlx::core::take(in_flat, flat_b);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out),
                                                    input->dtype_)};
    } else {
        auto out_cpu = allocate_size(
            static_cast<std::size_t>(N) * C * D_out * H_out * W_out,
            input->dtype_);
        const auto& xs = std::get<CpuStorage>(input->storage_);
        if (input->dtype_ == Dtype::F32)
            nearest3d_cpu<float>(
                reinterpret_cast<const float*>(xs.ptr.get()),
                reinterpret_cast<float*>(out_cpu.ptr.get()),
                N, C, D_in, H_in, W_in, D_out, H_out, W_out);
        else if (input->dtype_ == Dtype::F64)
            nearest3d_cpu<double>(
                reinterpret_cast<const double*>(xs.ptr.get()),
                reinterpret_cast<double*>(out_cpu.ptr.get()),
                N, C, D_in, H_in, W_in, D_out, H_out, W_out);
        else
            throw NotImplementedError("interpolate_nearest_3d: dtype must be F32/F64");
        out_storage = Storage{std::move(out_cpu)};
    }
    return std::make_shared<TensorImpl>(std::move(out_storage),
                                         out_shape, input->dtype_,
                                         input->device_, false);
}

}  // namespace lucid
