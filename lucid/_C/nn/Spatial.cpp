#include "Spatial.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Exceptions.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/TensorImpl.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

template <typename T>
T x_norm(int w, int W, bool align_corners) {
    if (align_corners) {
        return W > 1 ? T{-1} + (T{2} * static_cast<T>(w)) / static_cast<T>(W - 1) : T{0};
    }
    return T{-1} + (T{2} * static_cast<T>(w) + T{1}) / static_cast<T>(W);
}

inline ::mlx::core::array mlx_scalar_dt(double v, ::mlx::core::Dtype dt) {
    return ::mlx::core::astype(::mlx::core::array(static_cast<float>(v)), dt);
}

}  // namespace

// =====================================================================
// affine_grid
// =====================================================================

const OpSchema AffineGridBackward::schema_v1{"affine_grid", 1, AmpPolicy::Promote, true};

TensorImplPtr AffineGridBackward::forward(
    const TensorImplPtr& theta, int N, int H, int W, bool align_corners) {
    if (!theta)
        throw LucidError("affine_grid: null theta");
    if (theta->shape_.size() != 3 || theta->shape_[0] != N || theta->shape_[1] != 2 ||
        theta->shape_[2] != 3)
        throw ShapeMismatch(theta->shape_, Shape{static_cast<std::int64_t>(N), 2, 3},
                            "affine_grid: theta must be (N, 2, 3)");

    Shape out_shape{static_cast<std::int64_t>(N), static_cast<std::int64_t>(H),
                    static_cast<std::int64_t>(W), 2};
    OpScope scope{schema_v1.name, theta->device_, theta->dtype_, out_shape};

    Storage out_storage;
    if (theta->device_ == Device::GPU) {
        const auto& gt = std::get<GpuStorage>(theta->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(theta->dtype_);

        // Build x_norm[W], y_norm[H] in input dtype.
        auto build_norm = [&](int dim) {
            if (align_corners) {
                if (dim <= 1)
                    return ::mlx::core::zeros({1}, mlx_dt);
                auto a = ::mlx::core::astype(::mlx::core::arange(0, dim, 1), mlx_dt);
                auto step = ::mlx::core::astype(
                    ::mlx::core::array(2.0f / static_cast<float>(dim - 1)), mlx_dt);
                auto offset = ::mlx::core::astype(::mlx::core::array(-1.0f), mlx_dt);
                return ::mlx::core::add(::mlx::core::multiply(a, step), offset);
            }
            auto a = ::mlx::core::astype(::mlx::core::arange(0, dim, 1), mlx_dt);
            auto two = ::mlx::core::astype(::mlx::core::array(2.0f), mlx_dt);
            auto inv_dim =
                ::mlx::core::astype(::mlx::core::array(1.0f / static_cast<float>(dim)), mlx_dt);
            auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
            // x_norm[w] = -1 + (2w + 1) / W
            auto two_w_p1 = ::mlx::core::add(::mlx::core::multiply(two, a), one);
            return ::mlx::core::subtract(::mlx::core::multiply(two_w_p1, inv_dim), one);
        };
        auto xs = build_norm(W);  // [W]
        auto ys = build_norm(H);  // [H]
        // Build grid_xy of shape [H, W, 3]: (xs broadcast, ys broadcast, ones)
        auto xs_b = ::mlx::core::reshape(xs, {1, W, 1});  // [1, W, 1]
        auto ys_b = ::mlx::core::reshape(ys, {H, 1, 1});  // [H, 1, 1]
        auto ones_b = ::mlx::core::ones({H, W, 1}, mlx_dt);
        auto xs_broad = ::mlx::core::broadcast_to(xs_b, {H, W, 1});
        auto ys_broad = ::mlx::core::broadcast_to(ys_b, {H, W, 1});
        auto grid_xy = ::mlx::core::concatenate(
            std::vector<::mlx::core::array>{xs_broad, ys_broad, ones_b}, /*axis=*/-1);
        // Reshape to [1, HW, 3] for batched matmul. Then theta_T [N, 3, 2].
        auto grid_flat = ::mlx::core::reshape(grid_xy, {1, H * W, 3});
        auto grid_b = ::mlx::core::broadcast_to(grid_flat, {N, H * W, 3});
        auto theta_T = ::mlx::core::transpose(*gt.arr, {0, 2, 1});  // [N, 3, 2]
        auto out_flat = ::mlx::core::matmul(grid_b, theta_T);       // [N, HW, 2]
        auto out_arr = ::mlx::core::reshape(out_flat, {N, H, W, 2});
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out_arr), theta->dtype_)};
    } else {
        auto out_cpu = allocate_size(static_cast<std::size_t>(N) * H * W * 2, theta->dtype_);
        const auto& th = std::get<CpuStorage>(theta->storage_);

        auto run = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T* tp = reinterpret_cast<const T*>(th.ptr.get());
            T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
            for (int n = 0; n < N; ++n) {
                const T t00 = tp[n * 6 + 0], t01 = tp[n * 6 + 1], t02 = tp[n * 6 + 2];
                const T t10 = tp[n * 6 + 3], t11 = tp[n * 6 + 4], t12 = tp[n * 6 + 5];
                for (int h = 0; h < H; ++h) {
                    const T y = x_norm<T>(h, H, align_corners);
                    for (int w = 0; w < W; ++w) {
                        const T x = x_norm<T>(w, W, align_corners);
                        const std::size_t base =
                            ((static_cast<std::size_t>(n) * H + h) * W + w) * 2;
                        op[base + 0] = t00 * x + t01 * y + t02;
                        op[base + 1] = t10 * x + t11 * y + t12;
                    }
                }
            }
        };
        if (theta->dtype_ == Dtype::F32)
            run(float{});
        else if (theta->dtype_ == Dtype::F64)
            run(double{});
        else
            throw NotImplementedError("affine_grid: dtype must be F32/F64");
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, theta->dtype_,
                                            theta->device_, false);

    if (!GradMode::is_enabled() || !theta->requires_grad_)
        return out;

    auto t_edge = detail::ensure_grad_fn(theta);
    auto bwd = std::make_shared<AffineGridBackward>();
    bwd->input_shapes_ = {theta->shape_};
    bwd->out_shape_ = out_shape;
    bwd->dtype_ = theta->dtype_;
    bwd->device_ = theta->device_;
    bwd->input_tensors_ = {theta};
    bwd->saved_inputs_ = {theta->storage_};
    bwd->align_corners_ = align_corners;
    bwd->N_ = N;
    bwd->H_ = H;
    bwd->W_ = W;
    bwd->orig_theta_shape_ = theta->shape_;
    bwd->set_next_edges(std::vector<Edge>{Edge(t_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{static_cast<std::int64_t>(theta->version_)});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> AffineGridBackward::apply(Storage grad_out) {
    if (device_ == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);
        // Reconstruct grid_xy [H, W, 3] (depends only on H, W, align).
        auto build_norm = [&](int dim) {
            if (align_corners_) {
                if (dim <= 1)
                    return ::mlx::core::zeros({1}, mlx_dt);
                auto a = ::mlx::core::astype(::mlx::core::arange(0, dim, 1), mlx_dt);
                auto step = ::mlx::core::astype(
                    ::mlx::core::array(2.0f / static_cast<float>(dim - 1)), mlx_dt);
                auto offset = ::mlx::core::astype(::mlx::core::array(-1.0f), mlx_dt);
                return ::mlx::core::add(::mlx::core::multiply(a, step), offset);
            }
            auto a = ::mlx::core::astype(::mlx::core::arange(0, dim, 1), mlx_dt);
            auto two = ::mlx::core::astype(::mlx::core::array(2.0f), mlx_dt);
            auto inv_dim =
                ::mlx::core::astype(::mlx::core::array(1.0f / static_cast<float>(dim)), mlx_dt);
            auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
            auto two_w_p1 = ::mlx::core::add(::mlx::core::multiply(two, a), one);
            return ::mlx::core::subtract(::mlx::core::multiply(two_w_p1, inv_dim), one);
        };
        auto xs = build_norm(W_);
        auto ys = build_norm(H_);
        auto xs_b = ::mlx::core::reshape(xs, {1, W_, 1});
        auto ys_b = ::mlx::core::reshape(ys, {H_, 1, 1});
        auto ones_b = ::mlx::core::ones({H_, W_, 1}, mlx_dt);
        auto xs_broad = ::mlx::core::broadcast_to(xs_b, {H_, W_, 1});
        auto ys_broad = ::mlx::core::broadcast_to(ys_b, {H_, W_, 1});
        auto grid_xy = ::mlx::core::concatenate(
            std::vector<::mlx::core::array>{xs_broad, ys_broad, ones_b}, -1);
        auto grid_flat = ::mlx::core::reshape(grid_xy, {H_ * W_, 3});  // [HW, 3]
        // dgrid: [N, H, W, 2] → reshape to [N, HW, 2].
        auto dg = ::mlx::core::reshape(*gg.arr, {N_, H_ * W_, 2});
        // dtheta_T [N, 3, 2] = grid_xy^T [3, HW] @ dg [N, HW, 2]; but grid_xy is shared across
        // batch.
        auto grid_T = ::mlx::core::transpose(grid_flat);  // [3, HW]
        auto grid_T_b = ::mlx::core::broadcast_to(::mlx::core::reshape(grid_T, {1, 3, H_ * W_}),
                                                  {N_, 3, H_ * W_});
        auto dtheta_T = ::mlx::core::matmul(grid_T_b, dg);  // [N, 3, 2]
        // contiguous() forces a real layout copy so the wrapped GpuStorage's
        // raw buffer matches the logical [N, 2, 3] shape (otherwise transpose
        // returns a view with original [N, 3, 2] data layout).
        auto dtheta =
            ::mlx::core::contiguous(::mlx::core::transpose(dtheta_T, {0, 2, 1}));  // [N, 2, 3]
        return {Storage{gpu::wrap_mlx_array(std::move(dtheta), dtype_)}};
    }

    auto dtheta = allocate_size(static_cast<std::size_t>(N_) * 2 * 3, dtype_);
    std::memset(dtheta.ptr.get(), 0, dtheta.nbytes);
    const auto& gs = std::get<CpuStorage>(grad_out);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
        T* dp = reinterpret_cast<T*>(dtheta.ptr.get());
        for (int n = 0; n < N_; ++n) {
            T s00 = T{0}, s01 = T{0}, s02 = T{0};
            T s10 = T{0}, s11 = T{0}, s12 = T{0};
            for (int h = 0; h < H_; ++h) {
                const T y = x_norm<T>(h, H_, align_corners_);
                for (int w = 0; w < W_; ++w) {
                    const T x = x_norm<T>(w, W_, align_corners_);
                    const std::size_t base = ((static_cast<std::size_t>(n) * H_ + h) * W_ + w) * 2;
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
    if (dtype_ == Dtype::F32)
        run(float{});
    else if (dtype_ == Dtype::F64)
        run(double{});
    else
        throw NotImplementedError("affine_grid backward: dtype not supported");
    return {Storage{std::move(dtheta)}};
}

TensorImplPtr affine_grid_op(const TensorImplPtr& theta, int N, int H, int W, bool align_corners) {
    return AffineGridBackward::forward(theta, N, H, W, align_corners);
}
LUCID_REGISTER_OP(AffineGridBackward)

// =====================================================================
// grid_sample
// =====================================================================

const OpSchema GridSampleBackward::schema_v1{"grid_sample", 1, AmpPolicy::Promote, true};

namespace {

template <typename T>
T denorm(T g, int dim, bool align_corners) {
    if (align_corners) {
        return (g + T{1}) * static_cast<T>(dim - 1) / T{2};
    }
    return (g + T{1}) * static_cast<T>(dim) / T{2} - static_cast<T>(0.5);
}

template <typename T>
T denorm_grad_factor(int dim, bool align_corners) {
    return align_corners ? static_cast<T>(dim - 1) / T{2} : static_cast<T>(dim) / T{2};
}

}  // namespace

TensorImplPtr GridSampleBackward::forward(const TensorImplPtr& input,
                                          const TensorImplPtr& grid,
                                          int mode,
                                          int padding_mode,
                                          bool align_corners) {
    if (!input || !grid)
        throw LucidError("grid_sample: null input");
    if (input->device_ != grid->device_)
        throw DeviceMismatch(std::string(device_name(input->device_)),
                             std::string(device_name(grid->device_)), "grid_sample: input/grid");
    if (input->shape_.size() != 4)
        throw ShapeMismatch(input->shape_, Shape{},
                            "grid_sample: input must be (N, C, H_in, W_in)");
    if (grid->shape_.size() != 4 || grid->shape_[3] != 2)
        throw ShapeMismatch(grid->shape_, Shape{},
                            "grid_sample: grid must be (N, H_out, W_out, 2)");
    if (input->dtype_ != grid->dtype_)
        throw DtypeMismatch(std::string(dtype_name(input->dtype_)),
                            std::string(dtype_name(grid->dtype_)), "grid_sample");
    if (input->shape_[0] != grid->shape_[0])
        throw ShapeMismatch(input->shape_, grid->shape_, "grid_sample: batch size mismatch");

    const int N = static_cast<int>(input->shape_[0]);
    const int C = static_cast<int>(input->shape_[1]);
    const int H_in = static_cast<int>(input->shape_[2]);
    const int W_in = static_cast<int>(input->shape_[3]);
    const int H_out = static_cast<int>(grid->shape_[1]);
    const int W_out = static_cast<int>(grid->shape_[2]);

    Shape out_shape{static_cast<std::int64_t>(N), static_cast<std::int64_t>(C),
                    static_cast<std::int64_t>(H_out), static_cast<std::int64_t>(W_out)};
    OpScope scope{schema_v1.name, input->device_, input->dtype_, out_shape};

    Storage out_storage;
    if (input->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage_);
        const auto& gg = std::get<GpuStorage>(grid->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype_);

        // Slice grid into ix [N, H_out, W_out] and iy.
        auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
        auto gx_n = ::mlx::core::take(*gg.arr, idx0, /*axis=*/-1);  // [N, H_out, W_out]
        auto gy_n = ::mlx::core::take(*gg.arr, idx1, /*axis=*/-1);

        auto denorm_arr = [&](const ::mlx::core::array& g, int dim) {
            if (align_corners) {
                auto half = ::mlx::core::astype(::mlx::core::array(0.5f), mlx_dt);
                auto dim_m1 =
                    ::mlx::core::astype(::mlx::core::array(static_cast<float>(dim - 1)), mlx_dt);
                auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
                return ::mlx::core::multiply(
                    half, ::mlx::core::multiply(::mlx::core::add(g, one), dim_m1));
            }
            auto half = ::mlx::core::astype(::mlx::core::array(0.5f), mlx_dt);
            auto dim_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(dim)), mlx_dt);
            auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
            // (g + 1) * dim / 2 - 0.5
            auto num = ::mlx::core::multiply(::mlx::core::add(g, one), dim_arr);
            return ::mlx::core::subtract(::mlx::core::multiply(half, num), half);
        };
        auto ix = denorm_arr(gx_n, W_in);
        auto iy = denorm_arr(gy_n, H_in);

        // Helper: gather input[n, c, y, x] for given int y, x arrays of shape [N, H_out, W_out].
        // Returns [N, C, H_out, W_out] gathered values, with zero where (y, x) was OOB
        // and padding_mode == zeros.
        auto gather_corner = [&](const ::mlx::core::array& y_arr, const ::mlx::core::array& x_arr,
                                 bool zero_oob) {
            // OOB mask before clipping.
            auto zero_i = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
            auto Hm1 = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
            auto Wm1 = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
            auto in_y = ::mlx::core::logical_and(::mlx::core::greater_equal(y_arr, zero_i),
                                                 ::mlx::core::less_equal(y_arr, Hm1));
            auto in_x = ::mlx::core::logical_and(::mlx::core::greater_equal(x_arr, zero_i),
                                                 ::mlx::core::less_equal(x_arr, Wm1));
            auto in_bounds = ::mlx::core::logical_and(in_y, in_x);
            // Clip
            auto y_cl = ::mlx::core::clip(y_arr, std::optional<::mlx::core::array>(zero_i),
                                          std::optional<::mlx::core::array>(Hm1));
            auto x_cl = ::mlx::core::clip(x_arr, std::optional<::mlx::core::array>(zero_i),
                                          std::optional<::mlx::core::array>(Wm1));
            // Build flat_idx[n, c, h, w] using broadcasting.
            auto n_idx = ::mlx::core::reshape(
                ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int64),
                {N, 1, 1, 1});
            auto c_idx = ::mlx::core::reshape(
                ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int64),
                {1, C, 1, 1});
            auto y_b = ::mlx::core::reshape(y_cl, {N, 1, H_out, W_out});
            auto x_b = ::mlx::core::reshape(x_cl, {N, 1, H_out, W_out});
            auto stride_n =
                ::mlx::core::astype(::mlx::core::array(C * H_in * W_in), ::mlx::core::int64);
            auto stride_c =
                ::mlx::core::astype(::mlx::core::array(H_in * W_in), ::mlx::core::int64);
            auto stride_y = ::mlx::core::astype(::mlx::core::array(W_in), ::mlx::core::int64);
            auto flat_idx =
                ::mlx::core::add(::mlx::core::add(::mlx::core::multiply(n_idx, stride_n),
                                                  ::mlx::core::multiply(c_idx, stride_c)),
                                 ::mlx::core::add(::mlx::core::multiply(y_b, stride_y), x_b));
            // Broadcast to [N, C, H_out, W_out]
            auto flat_idx_b = ::mlx::core::broadcast_to(flat_idx, {N, C, H_out, W_out});
            auto in_flat = ::mlx::core::reshape(*gx.arr, {N * C * H_in * W_in});
            auto vals = ::mlx::core::take(in_flat, flat_idx_b);
            if (zero_oob) {
                auto in_b_dt = ::mlx::core::astype(in_bounds, mlx_dt);
                auto in_b_b = ::mlx::core::broadcast_to(
                    ::mlx::core::reshape(in_b_dt, {N, 1, H_out, W_out}), {N, C, H_out, W_out});
                vals = ::mlx::core::multiply(vals, in_b_b);
            }
            return vals;
        };

        auto round_int = [&](const ::mlx::core::array& a) {
            return ::mlx::core::astype(::mlx::core::round(a, 0), ::mlx::core::int64);
        };
        auto floor_int = [&](const ::mlx::core::array& a) {
            return ::mlx::core::astype(::mlx::core::floor(a), ::mlx::core::int64);
        };

        ::mlx::core::array y_out{0};
        if (mode == 1) {
            // nearest
            ::mlx::core::array ix_use = ix;
            ::mlx::core::array iy_use = iy;
            if (padding_mode == 1) {
                auto zf = ::mlx::core::astype(::mlx::core::array(0.0f), mlx_dt);
                auto Hm1f =
                    ::mlx::core::astype(::mlx::core::array(static_cast<float>(H_in - 1)), mlx_dt);
                auto Wm1f =
                    ::mlx::core::astype(::mlx::core::array(static_cast<float>(W_in - 1)), mlx_dt);
                ix_use = ::mlx::core::clip(ix, std::optional<::mlx::core::array>(zf),
                                           std::optional<::mlx::core::array>(Wm1f));
                iy_use = ::mlx::core::clip(iy, std::optional<::mlx::core::array>(zf),
                                           std::optional<::mlx::core::array>(Hm1f));
            }
            auto ix_r = round_int(ix_use);
            auto iy_r = round_int(iy_use);
            y_out = gather_corner(iy_r, ix_r, /*zero_oob=*/padding_mode == 0);
        } else {
            // bilinear: optional border clamp first.
            ::mlx::core::array ix_use = ix;
            ::mlx::core::array iy_use = iy;
            if (padding_mode == 1) {
                auto zf = ::mlx::core::astype(::mlx::core::array(0.0f), mlx_dt);
                auto Hm1f =
                    ::mlx::core::astype(::mlx::core::array(static_cast<float>(H_in - 1)), mlx_dt);
                auto Wm1f =
                    ::mlx::core::astype(::mlx::core::array(static_cast<float>(W_in - 1)), mlx_dt);
                ix_use = ::mlx::core::clip(ix, std::optional<::mlx::core::array>(zf),
                                           std::optional<::mlx::core::array>(Wm1f));
                iy_use = ::mlx::core::clip(iy, std::optional<::mlx::core::array>(zf),
                                           std::optional<::mlx::core::array>(Hm1f));
            }
            auto x0 = floor_int(ix_use);
            auto y0 = floor_int(iy_use);
            auto one_i = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
            auto x1 = ::mlx::core::add(x0, one_i);
            auto y1 = ::mlx::core::add(y0, one_i);
            // Weights
            auto x0_f = ::mlx::core::astype(x0, mlx_dt);
            auto y0_f = ::mlx::core::astype(y0, mlx_dt);
            auto x1_f = ::mlx::core::astype(x1, mlx_dt);
            auto y1_f = ::mlx::core::astype(y1, mlx_dt);
            auto wa = ::mlx::core::multiply(::mlx::core::subtract(x1_f, ix_use),
                                            ::mlx::core::subtract(y1_f, iy_use));
            auto wb = ::mlx::core::multiply(::mlx::core::subtract(x1_f, ix_use),
                                            ::mlx::core::subtract(iy_use, y0_f));
            auto wc = ::mlx::core::multiply(::mlx::core::subtract(ix_use, x0_f),
                                            ::mlx::core::subtract(y1_f, iy_use));
            auto wd = ::mlx::core::multiply(::mlx::core::subtract(ix_use, x0_f),
                                            ::mlx::core::subtract(iy_use, y0_f));
            // Reshape weights to [N, 1, H_out, W_out].
            auto wa_b = ::mlx::core::reshape(wa, {N, 1, H_out, W_out});
            auto wb_b = ::mlx::core::reshape(wb, {N, 1, H_out, W_out});
            auto wc_b = ::mlx::core::reshape(wc, {N, 1, H_out, W_out});
            auto wd_b = ::mlx::core::reshape(wd, {N, 1, H_out, W_out});
            auto Ia = gather_corner(y0, x0, /*zero_oob=*/padding_mode == 0);
            auto Ib = gather_corner(y1, x0, padding_mode == 0);
            auto Ic = gather_corner(y0, x1, padding_mode == 0);
            auto Id = gather_corner(y1, x1, padding_mode == 0);
            y_out = ::mlx::core::add(
                ::mlx::core::add(::mlx::core::multiply(Ia, wa_b), ::mlx::core::multiply(Ib, wb_b)),
                ::mlx::core::add(::mlx::core::multiply(Ic, wc_b), ::mlx::core::multiply(Id, wd_b)));
        }
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y_out), input->dtype_)};
        // GPU backward reuses the CPU analytic implementation after downloading
        // saved tensors and uploads the resulting gradients back to GPU.
        auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype_,
                                                input->device_, false);
        if (!GradMode::is_enabled() || !(input->requires_grad_ || grid->requires_grad_))
            return out;

        auto x_edge = detail::ensure_grad_fn(input);
        auto g_edge = detail::ensure_grad_fn(grid);
        auto bwd = std::make_shared<GridSampleBackward>();
        bwd->input_shapes_ = {input->shape_, grid->shape_};
        bwd->out_shape_ = out_shape;
        bwd->dtype_ = input->dtype_;
        bwd->device_ = input->device_;
        bwd->input_tensors_ = {input, grid};
        bwd->saved_inputs_ = {input->storage_, grid->storage_};
        bwd->mode_ = mode;
        bwd->padding_mode_ = padding_mode;
        bwd->align_corners_ = align_corners;
        bwd->input_shape_ = input->shape_;
        bwd->grid_shape_ = grid->shape_;
        bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(g_edge, 0)});
        bwd->set_saved_versions(std::vector<std::int64_t>{
            static_cast<std::int64_t>(input->version_), static_cast<std::int64_t>(grid->version_)});
        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
        return out;
    }

    auto out_cpu = allocate_size(static_cast<std::size_t>(N) * C * H_out * W_out, input->dtype_);
    const auto& xs = std::get<CpuStorage>(input->storage_);
    const auto& gs = std::get<CpuStorage>(grid->storage_);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
        T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
        const std::size_t in_chan = static_cast<std::size_t>(H_in) * W_in;
        const std::size_t out_chan = static_cast<std::size_t>(H_out) * W_out;

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    const std::size_t gidx =
                        ((static_cast<std::size_t>(n) * H_out + h) * W_out + w) * 2;
                    T ix = denorm<T>(gp[gidx + 0], W_in, align_corners);
                    T iy = denorm<T>(gp[gidx + 1], H_in, align_corners);

                    if (mode == 1) {  // nearest
                        // Round half-away-from-zero (matches PyTorch).
                        int ixr = static_cast<int>(std::nearbyint(static_cast<double>(ix)));
                        int iyr = static_cast<int>(std::nearbyint(static_cast<double>(iy)));
                        bool out_of_bounds = false;
                        if (padding_mode == 1) {
                            ixr = std::clamp(ixr, 0, W_in - 1);
                            iyr = std::clamp(iyr, 0, H_in - 1);
                        } else {
                            if (ixr < 0 || ixr > W_in - 1 || iyr < 0 || iyr > H_in - 1) {
                                out_of_bounds = true;
                            }
                            ixr = std::clamp(ixr, 0, W_in - 1);
                            iyr = std::clamp(iyr, 0, H_in - 1);
                        }
                        for (int c = 0; c < C; ++c) {
                            T v = out_of_bounds
                                      ? T{0}
                                      : xp[((static_cast<std::size_t>(n) * C + c) * in_chan) +
                                           iyr * W_in + ixr];
                            op[((static_cast<std::size_t>(n) * C + c) * out_chan) + h * W_out + w] =
                                v;
                        }
                    } else {  // bilinear
                        if (padding_mode == 1) {
                            ix = std::clamp<T>(ix, T{0}, static_cast<T>(W_in - 1));
                            iy = std::clamp<T>(iy, T{0}, static_cast<T>(H_in - 1));
                        }
                        const T x0f = std::floor(ix);
                        const T y0f = std::floor(iy);
                        const int x0 = static_cast<int>(x0f);
                        const int y0 = static_cast<int>(y0f);
                        const int x1 = x0 + 1;
                        const int y1 = y0 + 1;
                        const T wa = (static_cast<T>(x1) - ix) * (static_cast<T>(y1) - iy);
                        const T wb = (static_cast<T>(x1) - ix) * (iy - static_cast<T>(y0));
                        const T wc = (ix - static_cast<T>(x0)) * (static_cast<T>(y1) - iy);
                        const T wd = (ix - static_cast<T>(x0)) * (iy - static_cast<T>(y0));

                        auto fetch = [&](int yi, int xi, int c) -> T {
                            const bool oob = (xi < 0 || xi > W_in - 1 || yi < 0 || yi > H_in - 1);
                            if (oob && padding_mode == 0)
                                return T{0};
                            const int ycl = std::clamp(yi, 0, H_in - 1);
                            const int xcl = std::clamp(xi, 0, W_in - 1);
                            return xp[((static_cast<std::size_t>(n) * C + c) * in_chan) +
                                      ycl * W_in + xcl];
                        };
                        for (int c = 0; c < C; ++c) {
                            const T Ia = fetch(y0, x0, c);
                            const T Ib = fetch(y1, x0, c);
                            const T Ic = fetch(y0, x1, c);
                            const T Id = fetch(y1, x1, c);
                            op[((static_cast<std::size_t>(n) * C + c) * out_chan) + h * W_out + w] =
                                Ia * wa + Ib * wb + Ic * wc + Id * wd;
                        }
                    }
                }
            }
        }
    };
    if (input->dtype_ == Dtype::F32)
        run(float{});
    else if (input->dtype_ == Dtype::F64)
        run(double{});
    else
        throw NotImplementedError("grid_sample: dtype must be F32/F64");

    auto out = std::make_shared<TensorImpl>(Storage{std::move(out_cpu)}, out_shape, input->dtype_,
                                            input->device_, false);

    if (!GradMode::is_enabled() || !(input->requires_grad_ || grid->requires_grad_))
        return out;

    auto x_edge = detail::ensure_grad_fn(input);
    auto g_edge = detail::ensure_grad_fn(grid);
    auto bwd = std::make_shared<GridSampleBackward>();
    bwd->input_shapes_ = {input->shape_, grid->shape_};
    bwd->out_shape_ = out_shape;
    bwd->dtype_ = input->dtype_;
    bwd->device_ = input->device_;
    bwd->input_tensors_ = {input, grid};
    bwd->saved_inputs_ = {input->storage_, grid->storage_};
    bwd->mode_ = mode;
    bwd->padding_mode_ = padding_mode;
    bwd->align_corners_ = align_corners;
    bwd->input_shape_ = input->shape_;
    bwd->grid_shape_ = grid->shape_;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(g_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{static_cast<std::int64_t>(input->version_),
                                                      static_cast<std::int64_t>(grid->version_)});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> GridSampleBackward::apply(Storage grad_out) {
    const int N = static_cast<int>(input_shape_[0]);
    const int C = static_cast<int>(input_shape_[1]);
    const int H_in = static_cast<int>(input_shape_[2]);
    const int W_in = static_cast<int>(input_shape_[3]);
    const int H_out = static_cast<int>(grid_shape_[1]);
    const int W_out = static_cast<int>(grid_shape_[2]);

    // -----------------------------------------------------------------
    // GPU branch — native MLX scatter_add for dinput, broadcast-reduce
    // for dgrid. Avoids the CPU download/upload round-trip.
    // -----------------------------------------------------------------
    if (device_ == Device::GPU) {
        const auto& gx_arr = std::get<GpuStorage>(saved_inputs_[0]);  // input
        const auto& gg_arr = std::get<GpuStorage>(saved_inputs_[1]);  // grid
        const auto& go_arr = std::get<GpuStorage>(grad_out);          // dout
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);

        auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
        auto gx_n = ::mlx::core::take(*gg_arr.arr, idx0, -1);  // [N, H_out, W_out]
        auto gy_n = ::mlx::core::take(*gg_arr.arr, idx1, -1);

        auto half = ::mlx::core::astype(::mlx::core::array(0.5f), mlx_dt);
        auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
        auto zero = ::mlx::core::astype(::mlx::core::array(0.0f), mlx_dt);
        auto Hm1f = ::mlx::core::astype(::mlx::core::array(static_cast<float>(H_in - 1)), mlx_dt);
        auto Wm1f = ::mlx::core::astype(::mlx::core::array(static_cast<float>(W_in - 1)), mlx_dt);

        auto denorm_arr = [&](const ::mlx::core::array& g, int dim) {
            auto dim_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(dim)), mlx_dt);
            auto dim_m1 =
                ::mlx::core::astype(::mlx::core::array(static_cast<float>(dim - 1)), mlx_dt);
            if (align_corners_) {
                return ::mlx::core::multiply(
                    half, ::mlx::core::multiply(::mlx::core::add(g, one), dim_m1));
            }
            auto num = ::mlx::core::multiply(::mlx::core::add(g, one), dim_arr);
            return ::mlx::core::subtract(::mlx::core::multiply(half, num), half);
        };
        auto ix = denorm_arr(gx_n, W_in);
        auto iy = denorm_arr(gy_n, H_in);

        // Track who got clipped (bilinear + border) — dgrid is zeroed where so.
        ::mlx::core::array clipped_x =
            ::mlx::core::astype(::mlx::core::array(false), ::mlx::core::bool_);
        ::mlx::core::array clipped_y = clipped_x;
        clipped_x = ::mlx::core::broadcast_to(clipped_x, gx_n.shape());
        clipped_y = ::mlx::core::broadcast_to(clipped_y, gy_n.shape());
        if (mode_ == 0 && padding_mode_ == 1) {
            auto cx_lo = ::mlx::core::less(ix, zero);
            auto cx_hi = ::mlx::core::greater(ix, Wm1f);
            clipped_x = ::mlx::core::logical_or(cx_lo, cx_hi);
            auto cy_lo = ::mlx::core::less(iy, zero);
            auto cy_hi = ::mlx::core::greater(iy, Hm1f);
            clipped_y = ::mlx::core::logical_or(cy_lo, cy_hi);
            ix = ::mlx::core::clip(ix, std::optional<::mlx::core::array>(zero),
                                   std::optional<::mlx::core::array>(Wm1f));
            iy = ::mlx::core::clip(iy, std::optional<::mlx::core::array>(zero),
                                   std::optional<::mlx::core::array>(Hm1f));
        }

        const std::int64_t total_in = static_cast<std::int64_t>(N) * C * H_in * W_in;
        auto in_flat = ::mlx::core::reshape(*gx_arr.arr, {static_cast<int>(total_in)});

        // Stride constants in int64 for flat-index arithmetic.
        auto sn = ::mlx::core::astype(::mlx::core::array(C * H_in * W_in), ::mlx::core::int64);
        auto sc = ::mlx::core::astype(::mlx::core::array(H_in * W_in), ::mlx::core::int64);
        auto sy = ::mlx::core::astype(::mlx::core::array(W_in), ::mlx::core::int64);
        auto n_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int64), {N, 1, 1, 1});
        auto c_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int64), {1, C, 1, 1});

        // Build (yi_int, xi_int) -> flat_idx[N, C, H_out, W_out] (broadcast over C),
        // and a mask of in-bounds positions if zeros padding.
        auto build_flat_idx = [&](const ::mlx::core::array& yi, const ::mlx::core::array& xi) {
            auto y_b = ::mlx::core::reshape(yi, {N, 1, H_out, W_out});
            auto x_b = ::mlx::core::reshape(xi, {N, 1, H_out, W_out});
            auto flat = ::mlx::core::add(::mlx::core::add(::mlx::core::multiply(n_idx, sn),
                                                          ::mlx::core::multiply(c_idx, sc)),
                                         ::mlx::core::add(::mlx::core::multiply(y_b, sy), x_b));
            return ::mlx::core::broadcast_to(flat, {N, C, H_out, W_out});
        };
        auto build_in_bounds_mask = [&](const ::mlx::core::array& yi_pre_clip,
                                        const ::mlx::core::array& xi_pre_clip) {
            auto zero_i = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
            auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
            auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
            auto in_y = ::mlx::core::logical_and(::mlx::core::greater_equal(yi_pre_clip, zero_i),
                                                 ::mlx::core::less_equal(yi_pre_clip, Hm1_i));
            auto in_x = ::mlx::core::logical_and(::mlx::core::greater_equal(xi_pre_clip, zero_i),
                                                 ::mlx::core::less_equal(xi_pre_clip, Wm1_i));
            return ::mlx::core::logical_and(in_y, in_x);
        };

        // Initial dinput buffer (flat). Each corner scatter accumulates into it.
        auto dinput_flat = ::mlx::core::zeros({static_cast<int>(total_in)}, mlx_dt);

        ::mlx::core::array dgrid_x = zero;
        ::mlx::core::array dgrid_y = zero;
        dgrid_x = ::mlx::core::broadcast_to(dgrid_x, gx_n.shape());
        dgrid_y = ::mlx::core::broadcast_to(dgrid_y, gy_n.shape());

        if (mode_ == 1) {
            // Nearest: single corner; dgrid = 0.
            auto ixr_f = ::mlx::core::round(ix, 0);
            auto iyr_f = ::mlx::core::round(iy, 0);
            auto ixr = ::mlx::core::astype(ixr_f, ::mlx::core::int64);
            auto iyr = ::mlx::core::astype(iyr_f, ::mlx::core::int64);
            auto in_bounds = build_in_bounds_mask(iyr, ixr);
            auto zero_i = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
            auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
            auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
            ixr = ::mlx::core::clip(ixr, std::optional<::mlx::core::array>(zero_i),
                                    std::optional<::mlx::core::array>(Wm1_i));
            iyr = ::mlx::core::clip(iyr, std::optional<::mlx::core::array>(zero_i),
                                    std::optional<::mlx::core::array>(Hm1_i));
            auto flat_idx = build_flat_idx(iyr, ixr);  // [N, C, H_out, W_out]
            auto contrib = *go_arr.arr;                // [N, C, H_out, W_out]
            if (padding_mode_ == 0) {
                auto mask = ::mlx::core::astype(
                    ::mlx::core::reshape(in_bounds, {N, 1, H_out, W_out}), mlx_dt);
                auto mask_b = ::mlx::core::broadcast_to(mask, {N, C, H_out, W_out});
                contrib = ::mlx::core::multiply(contrib, mask_b);
            }
            // Flatten to 1-D and scatter_add into dinput_flat.
            // MLX contract: updates.ndim == a.ndim + indices.ndim. With
            // a [K] (rank 1) and indices [M] (rank 1), updates must be
            // [M, 1] — the trailing 1 is the "slice into a" dim.
            auto flat_idx_1d = ::mlx::core::reshape(flat_idx, {N * C * H_out * W_out});
            auto contrib_2d = ::mlx::core::reshape(contrib, {N * C * H_out * W_out, 1});
            dinput_flat =
                ::mlx::core::scatter_add(dinput_flat, flat_idx_1d, contrib_2d, /*axis=*/0);
        } else {
            // Bilinear: 4 corners.
            auto x0 = ::mlx::core::astype(::mlx::core::floor(ix), ::mlx::core::int64);
            auto y0 = ::mlx::core::astype(::mlx::core::floor(iy), ::mlx::core::int64);
            auto one_i = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
            auto x1 = ::mlx::core::add(x0, one_i);
            auto y1 = ::mlx::core::add(y0, one_i);

            auto x0_f = ::mlx::core::astype(x0, mlx_dt);
            auto y0_f = ::mlx::core::astype(y0, mlx_dt);
            auto x1_f = ::mlx::core::astype(x1, mlx_dt);
            auto y1_f = ::mlx::core::astype(y1, mlx_dt);
            auto wa = ::mlx::core::multiply(::mlx::core::subtract(x1_f, ix),
                                            ::mlx::core::subtract(y1_f, iy));
            auto wb = ::mlx::core::multiply(::mlx::core::subtract(x1_f, ix),
                                            ::mlx::core::subtract(iy, y0_f));
            auto wc = ::mlx::core::multiply(::mlx::core::subtract(ix, x0_f),
                                            ::mlx::core::subtract(y1_f, iy));
            auto wd = ::mlx::core::multiply(::mlx::core::subtract(ix, x0_f),
                                            ::mlx::core::subtract(iy, y0_f));

            auto zero_i = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
            auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
            auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
            auto clip_y = [&](const ::mlx::core::array& v) {
                return ::mlx::core::clip(v, std::optional<::mlx::core::array>(zero_i),
                                         std::optional<::mlx::core::array>(Hm1_i));
            };
            auto clip_x = [&](const ::mlx::core::array& v) {
                return ::mlx::core::clip(v, std::optional<::mlx::core::array>(zero_i),
                                         std::optional<::mlx::core::array>(Wm1_i));
            };
            auto y0c = clip_y(y0), y1c = clip_y(y1);
            auto x0c = clip_x(x0), x1c = clip_x(x1);
            auto in00 = build_in_bounds_mask(y0, x0);
            auto in01 = build_in_bounds_mask(y0, x1);
            auto in10 = build_in_bounds_mask(y1, x0);
            auto in11 = build_in_bounds_mask(y1, x1);

            // Helper: scatter_add(go * w * mask) into dinput_flat for one corner.
            auto scatter_one_corner = [&](const ::mlx::core::array& yi,
                                          const ::mlx::core::array& xi, const ::mlx::core::array& w,
                                          const ::mlx::core::array& in_bounds) {
                auto flat_idx = build_flat_idx(yi, xi);
                auto w_b = ::mlx::core::broadcast_to(::mlx::core::reshape(w, {N, 1, H_out, W_out}),
                                                     {N, C, H_out, W_out});
                ::mlx::core::array contrib = ::mlx::core::multiply(*go_arr.arr, w_b);
                if (padding_mode_ == 0) {
                    auto mask = ::mlx::core::astype(
                        ::mlx::core::reshape(in_bounds, {N, 1, H_out, W_out}), mlx_dt);
                    auto mask_b = ::mlx::core::broadcast_to(mask, {N, C, H_out, W_out});
                    contrib = ::mlx::core::multiply(contrib, mask_b);
                }
                auto flat_idx_1d = ::mlx::core::reshape(flat_idx, {N * C * H_out * W_out});
                auto contrib_2d = ::mlx::core::reshape(contrib, {N * C * H_out * W_out, 1});
                dinput_flat =
                    ::mlx::core::scatter_add(dinput_flat, flat_idx_1d, contrib_2d, /*axis=*/0);
            };
            scatter_one_corner(y0c, x0c, wa, in00);
            scatter_one_corner(y1c, x0c, wb, in10);
            scatter_one_corner(y0c, x1c, wc, in01);
            scatter_one_corner(y1c, x1c, wd, in11);

            // dgrid contributions — analytic chain through bilinear.
            // For each corner, fetch I_corner across channels (gather from input).
            auto fetch_corner = [&](const ::mlx::core::array& yi, const ::mlx::core::array& xi,
                                    const ::mlx::core::array& in_bounds) {
                auto flat_idx = build_flat_idx(yi, xi);
                auto vals = ::mlx::core::take(in_flat, flat_idx);
                if (padding_mode_ == 0) {
                    auto mask = ::mlx::core::astype(
                        ::mlx::core::reshape(in_bounds, {N, 1, H_out, W_out}), mlx_dt);
                    auto mask_b = ::mlx::core::broadcast_to(mask, {N, C, H_out, W_out});
                    vals = ::mlx::core::multiply(vals, mask_b);
                }
                return vals;
            };
            auto Ia = fetch_corner(y0c, x0c, in00);
            auto Ib = fetch_corner(y1c, x0c, in10);
            auto Ic = fetch_corner(y0c, x1c, in01);
            auto Id = fetch_corner(y1c, x1c, in11);

            // dy_term1 = y1_f - iy ; dy_term2 = iy - y0_f
            auto dy_t1 = ::mlx::core::subtract(y1_f, iy);
            auto dy_t2 = ::mlx::core::subtract(iy, y0_f);
            auto dx_t1 = ::mlx::core::subtract(x1_f, ix);
            auto dx_t2 = ::mlx::core::subtract(ix, x0_f);
            auto dy_t1_b = ::mlx::core::broadcast_to(
                ::mlx::core::reshape(dy_t1, {N, 1, H_out, W_out}), {N, C, H_out, W_out});
            auto dy_t2_b = ::mlx::core::broadcast_to(
                ::mlx::core::reshape(dy_t2, {N, 1, H_out, W_out}), {N, C, H_out, W_out});
            auto dx_t1_b = ::mlx::core::broadcast_to(
                ::mlx::core::reshape(dx_t1, {N, 1, H_out, W_out}), {N, C, H_out, W_out});
            auto dx_t2_b = ::mlx::core::broadcast_to(
                ::mlx::core::reshape(dx_t2, {N, 1, H_out, W_out}), {N, C, H_out, W_out});
            // dix_per_chan = go * ((Ic - Ia) * dy_t1 + (Id - Ib) * dy_t2)
            auto dix_per = ::mlx::core::multiply(
                *go_arr.arr,
                ::mlx::core::add(::mlx::core::multiply(::mlx::core::subtract(Ic, Ia), dy_t1_b),
                                 ::mlx::core::multiply(::mlx::core::subtract(Id, Ib), dy_t2_b)));
            // diy_per = go * ((Ib - Ia) * dx_t1 + (Id - Ic) * dx_t2)
            auto diy_per = ::mlx::core::multiply(
                *go_arr.arr,
                ::mlx::core::add(::mlx::core::multiply(::mlx::core::subtract(Ib, Ia), dx_t1_b),
                                 ::mlx::core::multiply(::mlx::core::subtract(Id, Ic), dx_t2_b)));
            // Sum over channel axis to get [N, H_out, W_out].
            auto dix_sum = ::mlx::core::sum(dix_per, std::vector<int>{1}, /*keepdims=*/false);
            auto diy_sum = ::mlx::core::sum(diy_per, std::vector<int>{1}, /*keepdims=*/false);
            auto sx =
                mlx_scalar_dt(static_cast<double>(W_in - (align_corners_ ? 1 : 0)) / 2.0, mlx_dt);
            // align: (in - 1) / 2;  not align: in / 2.
            sx = align_corners_ ? mlx_scalar_dt(static_cast<double>(W_in - 1) / 2.0, mlx_dt)
                                : mlx_scalar_dt(static_cast<double>(W_in) / 2.0, mlx_dt);
            auto sy_arr = align_corners_
                              ? mlx_scalar_dt(static_cast<double>(H_in - 1) / 2.0, mlx_dt)
                              : mlx_scalar_dt(static_cast<double>(H_in) / 2.0, mlx_dt);
            dgrid_x = ::mlx::core::multiply(dix_sum, sx);
            dgrid_y = ::mlx::core::multiply(diy_sum, sy_arr);
            // border + clipped → zero out dgrid contribution.
            if (padding_mode_ == 1) {
                auto inv_x = ::mlx::core::logical_not(clipped_x);
                auto inv_y = ::mlx::core::logical_not(clipped_y);
                auto mx = ::mlx::core::astype(inv_x, mlx_dt);
                auto my = ::mlx::core::astype(inv_y, mlx_dt);
                dgrid_x = ::mlx::core::multiply(dgrid_x, mx);
                dgrid_y = ::mlx::core::multiply(dgrid_y, my);
            }
        }

        // Reshape dinput back to [N, C, H_in, W_in].
        auto dx_arr = ::mlx::core::reshape(dinput_flat, gpu::to_mlx_shape(input_shape_));

        // Stack dgrid_x, dgrid_y into [N, H_out, W_out, 2].
        auto dgx_e = ::mlx::core::expand_dims(dgrid_x, /*axis=*/-1);
        auto dgy_e = ::mlx::core::expand_dims(dgrid_y, /*axis=*/-1);
        auto dg_arr = ::mlx::core::concatenate(std::vector<::mlx::core::array>{dgx_e, dgy_e}, -1);
        return {Storage{gpu::wrap_mlx_array(std::move(dx_arr), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dg_arr), dtype_)}};
    }

    // -----------------------------------------------------------------
    // CPU branch (analytic forward+backward, unchanged).
    // -----------------------------------------------------------------
    auto dx = allocate_size(static_cast<std::size_t>(N) * C * H_in * W_in, dtype_);
    auto dg = allocate_size(static_cast<std::size_t>(N) * H_out * W_out * 2, dtype_);
    std::memset(dx.ptr.get(), 0, dx.nbytes);
    std::memset(dg.ptr.get(), 0, dg.nbytes);

    const auto& xs = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& gs = std::get<CpuStorage>(saved_inputs_[1]);
    const auto& gout = std::get<CpuStorage>(grad_out);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
        const T* op = reinterpret_cast<const T*>(gout.ptr.get());
        T* dxp = reinterpret_cast<T*>(dx.ptr.get());
        T* dgp = reinterpret_cast<T*>(dg.ptr.get());
        const std::size_t in_chan = static_cast<std::size_t>(H_in) * W_in;
        const std::size_t out_chan = static_cast<std::size_t>(H_out) * W_out;
        const T sx = denorm_grad_factor<T>(W_in, align_corners_);
        const T sy = denorm_grad_factor<T>(H_in, align_corners_);

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    const std::size_t gidx =
                        ((static_cast<std::size_t>(n) * H_out + h) * W_out + w) * 2;
                    const T gx_norm = gp[gidx + 0];
                    const T gy_norm = gp[gidx + 1];
                    T ix = denorm<T>(gx_norm, W_in, align_corners_);
                    T iy = denorm<T>(gy_norm, H_in, align_corners_);

                    bool clipped_x = false, clipped_y = false;
                    if (mode_ == 0 && padding_mode_ == 1) {
                        // bilinear + border: clamp the pre-floor coords.
                        if (ix < T{0}) {
                            ix = T{0};
                            clipped_x = true;
                        }
                        if (ix > static_cast<T>(W_in - 1)) {
                            ix = static_cast<T>(W_in - 1);
                            clipped_x = true;
                        }
                        if (iy < T{0}) {
                            iy = T{0};
                            clipped_y = true;
                        }
                        if (iy > static_cast<T>(H_in - 1)) {
                            iy = static_cast<T>(H_in - 1);
                            clipped_y = true;
                        }
                    }

                    if (mode_ == 1) {
                        // Nearest: dgrid is identically zero (round is non-differentiable).
                        // dinput: scatter dout to the chosen neighbor.
                        int ixr = static_cast<int>(std::nearbyint(static_cast<double>(ix)));
                        int iyr = static_cast<int>(std::nearbyint(static_cast<double>(iy)));
                        bool oob = false;
                        if (padding_mode_ == 1) {
                            ixr = std::clamp(ixr, 0, W_in - 1);
                            iyr = std::clamp(iyr, 0, H_in - 1);
                        } else {
                            if (ixr < 0 || ixr > W_in - 1 || iyr < 0 || iyr > H_in - 1)
                                oob = true;
                            ixr = std::clamp(ixr, 0, W_in - 1);
                            iyr = std::clamp(iyr, 0, H_in - 1);
                        }
                        if (!oob) {
                            for (int c = 0; c < C; ++c) {
                                const T go = op[((static_cast<std::size_t>(n) * C + c) * out_chan) +
                                                h * W_out + w];
                                dxp[((static_cast<std::size_t>(n) * C + c) * in_chan) + iyr * W_in +
                                    ixr] += go;
                            }
                        }
                        continue;
                    }

                    // Bilinear.
                    const T x0f = std::floor(ix);
                    const T y0f = std::floor(iy);
                    const int x0 = static_cast<int>(x0f);
                    const int y0 = static_cast<int>(y0f);
                    const int x1 = x0 + 1;
                    const int y1 = y0 + 1;
                    const T wa = (static_cast<T>(x1) - ix) * (static_cast<T>(y1) - iy);
                    const T wb = (static_cast<T>(x1) - ix) * (iy - static_cast<T>(y0));
                    const T wc = (ix - static_cast<T>(x0)) * (static_cast<T>(y1) - iy);
                    const T wd = (ix - static_cast<T>(x0)) * (iy - static_cast<T>(y0));

                    auto in_bounds = [&](int yi, int xi) -> bool {
                        return xi >= 0 && xi <= W_in - 1 && yi >= 0 && yi <= H_in - 1;
                    };
                    auto fetch_for_dgrid = [&](int yi, int xi, int c) -> T {
                        const bool oob = !in_bounds(yi, xi);
                        if (oob && padding_mode_ == 0)
                            return T{0};
                        const int ycl = std::clamp(yi, 0, H_in - 1);
                        const int xcl = std::clamp(xi, 0, W_in - 1);
                        return xp[((static_cast<std::size_t>(n) * C + c) * in_chan) + ycl * W_in +
                                  xcl];
                    };
                    auto scatter_dx = [&](int yi, int xi, int c, T contrib) {
                        const bool oob = !in_bounds(yi, xi);
                        if (oob && padding_mode_ == 0)
                            return;
                        const int ycl = std::clamp(yi, 0, H_in - 1);
                        const int xcl = std::clamp(xi, 0, W_in - 1);
                        dxp[((static_cast<std::size_t>(n) * C + c) * in_chan) + ycl * W_in + xcl] +=
                            contrib;
                    };

                    T dix_acc = T{0};
                    T diy_acc = T{0};
                    for (int c = 0; c < C; ++c) {
                        const T go =
                            op[((static_cast<std::size_t>(n) * C + c) * out_chan) + h * W_out + w];
                        // dinput contributions (4 corners).
                        scatter_dx(y0, x0, c, go * wa);
                        scatter_dx(y1, x0, c, go * wb);
                        scatter_dx(y0, x1, c, go * wc);
                        scatter_dx(y1, x1, c, go * wd);
                        // dgrid contributions (analytic gradient through bilinear).
                        const T Ia = fetch_for_dgrid(y0, x0, c);
                        const T Ib = fetch_for_dgrid(y1, x0, c);
                        const T Ic = fetch_for_dgrid(y0, x1, c);
                        const T Id = fetch_for_dgrid(y1, x1, c);
                        const T dy_term1 = static_cast<T>(y1) - iy;
                        const T dy_term2 = iy - static_cast<T>(y0);
                        const T dx_term1 = static_cast<T>(x1) - ix;
                        const T dx_term2 = ix - static_cast<T>(x0);
                        dix_acc += go * ((Ic - Ia) * dy_term1 + (Id - Ib) * dy_term2);
                        diy_acc += go * ((Ib - Ia) * dx_term1 + (Id - Ic) * dx_term2);
                    }
                    if (clipped_x)
                        dix_acc = T{0};
                    if (clipped_y)
                        diy_acc = T{0};
                    dgp[gidx + 0] = dix_acc * sx;
                    dgp[gidx + 1] = diy_acc * sy;
                }
            }
        }
    };
    if (dtype_ == Dtype::F32)
        run(float{});
    else if (dtype_ == Dtype::F64)
        run(double{});
    else
        throw NotImplementedError("grid_sample backward: dtype not supported");

    // GPU branch returns earlier via the native MLX path; this code only runs
    // for CPU input.
    return {Storage{std::move(dx)}, Storage{std::move(dg)}};
}

TensorImplPtr grid_sample_op(const TensorImplPtr& input,
                             const TensorImplPtr& grid,
                             int mode,
                             int padding_mode,
                             bool align_corners) {
    return GridSampleBackward::forward(input, grid, mode, padding_mode, align_corners);
}
LUCID_REGISTER_OP(GridSampleBackward)

}  // namespace lucid
