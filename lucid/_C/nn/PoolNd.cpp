#include "PoolNd.h"

#include <cstring>
#include <limits>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/cpu/Pool.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

template <>
const OpSchema MaxPool1dBackward::schema_v1{"max_pool1d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema MaxPool2dBackward::schema_v1{"max_pool2d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema MaxPool3dBackward::schema_v1{"max_pool3d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema AvgPool1dBackward::schema_v1{"avg_pool1d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema AvgPool2dBackward::schema_v1{"avg_pool2d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema AvgPool3dBackward::schema_v1{"avg_pool3d", 1, AmpPolicy::KeepInput, true};

namespace {

CpuStorage alloc_bytes(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}
CpuStorage alloc_i32(std::size_t numel) {
    CpuStorage s;
    s.dtype = Dtype::I32;
    s.nbytes = numel * sizeof(std::int32_t);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

inline int compute_out(int S, int K, int stride, int pad) {
    return (S + 2 * pad - K) / stride + 1;
}

template <int N>
void validate_input(const TensorImplPtr& x, std::string_view op_name) {
    Validator::input(x, std::string(op_name) + ".x").non_null();
    if (static_cast<int>(x->shape().size()) != N + 2)
        throw ShapeMismatch(x->shape(), Shape{}, std::string(op_name) + ": x rank mismatch");
}

// CPU forward dispatch by N — picks the right backend kernel.
template <int N, typename T>
void max_pool_forward_dispatch(const T* x,
                               T* y,
                               std::int32_t* a,
                               int B,
                               int C,
                               const int* S,
                               const int* K,
                               const int* O,
                               const int* stride,
                               const int* pad);
template <>
void max_pool_forward_dispatch<1, float>(const float* x,
                                         float* y,
                                         std::int32_t* a,
                                         int B,
                                         int C,
                                         const int* S,
                                         const int* K,
                                         const int* O,
                                         const int* stride,
                                         const int* pad) {
    backend::cpu::max_pool1d_forward_f32(x, y, a, B, C, S[0], K[0], O[0], stride[0], pad[0]);
}
template <>
void max_pool_forward_dispatch<1, double>(const double* x,
                                          double* y,
                                          std::int32_t* a,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::max_pool1d_forward_f64(x, y, a, B, C, S[0], K[0], O[0], stride[0], pad[0]);
}
template <>
void max_pool_forward_dispatch<2, float>(const float* x,
                                         float* y,
                                         std::int32_t* a,
                                         int B,
                                         int C,
                                         const int* S,
                                         const int* K,
                                         const int* O,
                                         const int* stride,
                                         const int* pad) {
    backend::cpu::max_pool2d_forward_f32(x, y, a, B, C, S[0], S[1], K[0], K[1], O[0], O[1],
                                         stride[0], stride[1], pad[0], pad[1]);
}
template <>
void max_pool_forward_dispatch<2, double>(const double* x,
                                          double* y,
                                          std::int32_t* a,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::max_pool2d_forward_f64(x, y, a, B, C, S[0], S[1], K[0], K[1], O[0], O[1],
                                         stride[0], stride[1], pad[0], pad[1]);
}
template <>
void max_pool_forward_dispatch<3, float>(const float* x,
                                         float* y,
                                         std::int32_t* a,
                                         int B,
                                         int C,
                                         const int* S,
                                         const int* K,
                                         const int* O,
                                         const int* stride,
                                         const int* pad) {
    backend::cpu::max_pool3d_forward_f32(x, y, a, B, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0],
                                         O[1], O[2], stride[0], stride[1], stride[2], pad[0],
                                         pad[1], pad[2]);
}
template <>
void max_pool_forward_dispatch<3, double>(const double* x,
                                          double* y,
                                          std::int32_t* a,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::max_pool3d_forward_f64(x, y, a, B, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0],
                                         O[1], O[2], stride[0], stride[1], stride[2], pad[0],
                                         pad[1], pad[2]);
}

template <int N, typename T>
void max_pool_backward_dispatch(
    const T* g, const std::int32_t* a, T* dx, int B, int C, const int* S, const int* O);
template <>
void max_pool_backward_dispatch<1, float>(
    const float* g, const std::int32_t* a, float* dx, int B, int C, const int* S, const int* O) {
    backend::cpu::max_pool1d_backward_f32(g, a, dx, B, C, S[0], O[0]);
}
template <>
void max_pool_backward_dispatch<1, double>(
    const double* g, const std::int32_t* a, double* dx, int B, int C, const int* S, const int* O) {
    backend::cpu::max_pool1d_backward_f64(g, a, dx, B, C, S[0], O[0]);
}
template <>
void max_pool_backward_dispatch<2, float>(
    const float* g, const std::int32_t* a, float* dx, int B, int C, const int* S, const int* O) {
    backend::cpu::max_pool2d_backward_f32(g, a, dx, B, C, S[0], S[1], O[0], O[1]);
}
template <>
void max_pool_backward_dispatch<2, double>(
    const double* g, const std::int32_t* a, double* dx, int B, int C, const int* S, const int* O) {
    backend::cpu::max_pool2d_backward_f64(g, a, dx, B, C, S[0], S[1], O[0], O[1]);
}
template <>
void max_pool_backward_dispatch<3, float>(
    const float* g, const std::int32_t* a, float* dx, int B, int C, const int* S, const int* O) {
    backend::cpu::max_pool3d_backward_f32(g, a, dx, B, C, S[0], S[1], S[2], O[0], O[1], O[2]);
}
template <>
void max_pool_backward_dispatch<3, double>(
    const double* g, const std::int32_t* a, double* dx, int B, int C, const int* S, const int* O) {
    backend::cpu::max_pool3d_backward_f64(g, a, dx, B, C, S[0], S[1], S[2], O[0], O[1], O[2]);
}

template <int N, typename T>
void avg_pool_forward_dispatch(const T* x,
                               T* y,
                               int B,
                               int C,
                               const int* S,
                               const int* K,
                               const int* O,
                               const int* stride,
                               const int* pad);
template <>
void avg_pool_forward_dispatch<1, float>(const float* x,
                                         float* y,
                                         int B,
                                         int C,
                                         const int* S,
                                         const int* K,
                                         const int* O,
                                         const int* stride,
                                         const int* pad) {
    backend::cpu::avg_pool1d_forward_f32(x, y, B, C, S[0], K[0], O[0], stride[0], pad[0]);
}
template <>
void avg_pool_forward_dispatch<1, double>(const double* x,
                                          double* y,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::avg_pool1d_forward_f64(x, y, B, C, S[0], K[0], O[0], stride[0], pad[0]);
}
template <>
void avg_pool_forward_dispatch<2, float>(const float* x,
                                         float* y,
                                         int B,
                                         int C,
                                         const int* S,
                                         const int* K,
                                         const int* O,
                                         const int* stride,
                                         const int* pad) {
    backend::cpu::avg_pool2d_forward_f32(x, y, B, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0],
                                         stride[1], pad[0], pad[1]);
}
template <>
void avg_pool_forward_dispatch<2, double>(const double* x,
                                          double* y,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::avg_pool2d_forward_f64(x, y, B, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0],
                                         stride[1], pad[0], pad[1]);
}
template <>
void avg_pool_forward_dispatch<3, float>(const float* x,
                                         float* y,
                                         int B,
                                         int C,
                                         const int* S,
                                         const int* K,
                                         const int* O,
                                         const int* stride,
                                         const int* pad) {
    backend::cpu::avg_pool3d_forward_f32(x, y, B, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1],
                                         O[2], stride[0], stride[1], stride[2], pad[0], pad[1],
                                         pad[2]);
}
template <>
void avg_pool_forward_dispatch<3, double>(const double* x,
                                          double* y,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::avg_pool3d_forward_f64(x, y, B, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1],
                                         O[2], stride[0], stride[1], stride[2], pad[0], pad[1],
                                         pad[2]);
}

template <int N, typename T>
void avg_pool_backward_dispatch(const T* g,
                                T* dx,
                                int B,
                                int C,
                                const int* S,
                                const int* K,
                                const int* O,
                                const int* stride,
                                const int* pad);
template <>
void avg_pool_backward_dispatch<1, float>(const float* g,
                                          float* dx,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::avg_pool1d_backward_f32(g, dx, B, C, S[0], K[0], O[0], stride[0], pad[0]);
}
template <>
void avg_pool_backward_dispatch<1, double>(const double* g,
                                           double* dx,
                                           int B,
                                           int C,
                                           const int* S,
                                           const int* K,
                                           const int* O,
                                           const int* stride,
                                           const int* pad) {
    backend::cpu::avg_pool1d_backward_f64(g, dx, B, C, S[0], K[0], O[0], stride[0], pad[0]);
}
template <>
void avg_pool_backward_dispatch<2, float>(const float* g,
                                          float* dx,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::avg_pool2d_backward_f32(g, dx, B, C, S[0], S[1], K[0], K[1], O[0], O[1],
                                          stride[0], stride[1], pad[0], pad[1]);
}
template <>
void avg_pool_backward_dispatch<2, double>(const double* g,
                                           double* dx,
                                           int B,
                                           int C,
                                           const int* S,
                                           const int* K,
                                           const int* O,
                                           const int* stride,
                                           const int* pad) {
    backend::cpu::avg_pool2d_backward_f64(g, dx, B, C, S[0], S[1], K[0], K[1], O[0], O[1],
                                          stride[0], stride[1], pad[0], pad[1]);
}
template <>
void avg_pool_backward_dispatch<3, float>(const float* g,
                                          float* dx,
                                          int B,
                                          int C,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const int* stride,
                                          const int* pad) {
    backend::cpu::avg_pool3d_backward_f32(g, dx, B, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0],
                                          O[1], O[2], stride[0], stride[1], stride[2], pad[0],
                                          pad[1], pad[2]);
}
template <>
void avg_pool_backward_dispatch<3, double>(const double* g,
                                           double* dx,
                                           int B,
                                           int C,
                                           const int* S,
                                           const int* K,
                                           const int* O,
                                           const int* stride,
                                           const int* pad) {
    backend::cpu::avg_pool3d_backward_f64(g, dx, B, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0],
                                          O[1], O[2], stride[0], stride[1], stride[2], pad[0],
                                          pad[1], pad[2]);
}

// Build the 6-D (or generally (N+1)*2-d) windowed view via as_strided.
// Padded input: (B, C, *Sp), strides assume row-major.
// Windowed:  (B, C, *O, *K), with strides:
//   B → C * prod(Sp)
//   C → prod(Sp)
//   O[i] → stride_in[i] * prod(Sp[i+1:])
//   K[i] → prod(Sp[i+1:])
template <int N>
::mlx::core::array build_window_view(const ::mlx::core::array& padded,
                                     int B,
                                     int C,
                                     const int* Sp,
                                     const int* O,
                                     const int* K,
                                     const int* stride) {
    using SE = ::mlx::core::ShapeElem;
    using SS = std::int64_t;  // mlx Strides element

    // Shape: (B, C, *O, *K)
    ::mlx::core::Shape windowed;
    windowed.reserve(2 + 2 * N);
    windowed.push_back(static_cast<SE>(B));
    windowed.push_back(static_cast<SE>(C));
    for (int i = 0; i < N; ++i)
        windowed.push_back(static_cast<SE>(O[i]));
    for (int i = 0; i < N; ++i)
        windowed.push_back(static_cast<SE>(K[i]));

    // Compute Sp suffix products.
    SS suffix[N + 1];
    suffix[N] = 1;
    for (int i = N - 1; i >= 0; --i)
        suffix[i] = suffix[i + 1] * static_cast<SS>(Sp[i]);

    ::mlx::core::Strides strides_v;
    strides_v.reserve(2 + 2 * N);
    // B
    strides_v.push_back(static_cast<SS>(C) * suffix[0]);
    // C
    strides_v.push_back(suffix[0]);
    // O[i]: stride_in[i] * suffix[i+1]
    for (int i = 0; i < N; ++i)
        strides_v.push_back(static_cast<SS>(stride[i]) * suffix[i + 1]);
    // K[i]: suffix[i+1]
    for (int i = 0; i < N; ++i)
        strides_v.push_back(suffix[i + 1]);

    return ::mlx::core::as_strided(padded, windowed, strides_v, 0);
}

}  // namespace

// =====================================================================
// MaxPoolNd
// =====================================================================

template <int N>
TensorImplPtr MaxPoolNdBackward<N>::forward(const TensorImplPtr& x,
                                            const int (&K)[N],
                                            const int (&stride_in)[N],
                                            const int (&pad)[N]) {
    validate_input<N>(x, MaxPoolNdBackward<N>::schema_v1.name);
    int stride[N];
    for (int i = 0; i < N; ++i)
        stride[i] = (stride_in[i] == 0) ? K[i] : stride_in[i];

    const int B = static_cast<int>(x->shape()[0]);
    const int C = static_cast<int>(x->shape()[1]);
    int S[N], O[N];
    int O_total = 1, S_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape()[2 + i]);
        O[i] = compute_out(S[i], K[i], stride[i], pad[i]);
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape(), Shape{}, "max_pool: output non-positive");
        O_total *= O[i];
        S_total *= S[i];
    }
    int K_total = 1;
    for (int i = 0; i < N; ++i)
        K_total *= K[i];

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(C));
    for (int i = 0; i < N; ++i)
        out_shape.push_back(static_cast<std::int64_t>(O[i]));

    OpScopeFull scope{MaxPoolNdBackward<N>::schema_v1.name, x->device(), x->dtype(), out_shape};

    Storage out_storage;
    Storage saved_argmax;

    if (x->device() == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage());
        if (!gx.arr)
            ErrorBuilder("max_pool").fail("null GPU input");

        ::mlx::core::array neg_inf(-std::numeric_limits<double>::infinity(),
                                   gpu::to_mlx_dtype(x->dtype()));
        std::vector<std::pair<int, int>> pad_widths;
        pad_widths.reserve(N + 2);
        pad_widths.emplace_back(0, 0);
        pad_widths.emplace_back(0, 0);
        for (int i = 0; i < N; ++i)
            pad_widths.emplace_back(pad[i], pad[i]);
        auto x_pad = ::mlx::core::pad(*gx.arr, pad_widths, neg_inf);

        int Sp[N];
        for (int i = 0; i < N; ++i)
            Sp[i] = S[i] + 2 * pad[i];
        auto wins = build_window_view<N>(x_pad, B, C, Sp, O, K, stride);

        // y = max over the trailing N kernel axes
        std::vector<int> kernel_axes;
        kernel_axes.reserve(N);
        for (int i = 0; i < N; ++i)
            kernel_axes.push_back(2 + N + i);
        auto y = ::mlx::core::max(wins, kernel_axes, /*keepdims=*/false);
        // argmax over flattened kernel
        ::mlx::core::Shape flat_win;
        flat_win.reserve(N + 3);
        flat_win.push_back(static_cast<::mlx::core::ShapeElem>(B));
        flat_win.push_back(static_cast<::mlx::core::ShapeElem>(C));
        for (int i = 0; i < N; ++i)
            flat_win.push_back(static_cast<::mlx::core::ShapeElem>(O[i]));
        flat_win.push_back(static_cast<::mlx::core::ShapeElem>(K_total));
        auto wins_flat = ::mlx::core::reshape(wins, flat_win);
        auto argmax = ::mlx::core::argmax(wins_flat, /*axis=*/2 + N,
                                          /*keepdims=*/false);
        argmax = ::mlx::core::astype(argmax, ::mlx::core::int32);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype())};
        saved_argmax = Storage{gpu::wrap_mlx_array(std::move(argmax), Dtype::I32)};
    } else {
        auto y_cpu = alloc_bytes(static_cast<std::size_t>(B) * C * O_total, x->dtype());
        auto am_cpu = alloc_i32(static_cast<std::size_t>(B) * C * O_total);
        const auto& x_cpu = std::get<CpuStorage>(x->storage());
        switch (x->dtype()) {
            case Dtype::F32:
                max_pool_forward_dispatch<N, float>(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<float*>(y_cpu.ptr.get()),
                    reinterpret_cast<std::int32_t*>(am_cpu.ptr.get()), B, C, S, K, O, stride, pad);
                break;
            case Dtype::F64:
                max_pool_forward_dispatch<N, double>(
                    reinterpret_cast<const double*>(x_cpu.ptr.get()),
                    reinterpret_cast<double*>(y_cpu.ptr.get()),
                    reinterpret_cast<std::int32_t*>(am_cpu.ptr.get()), B, C, S, K, O, stride, pad);
                break;
            default:
                ErrorBuilder("max_pool").not_implemented("dtype not supported (F32/F64)");
        }
        out_storage = Storage{std::move(y_cpu)};
        saved_argmax = Storage{std::move(am_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);
    if (!GradMode::is_enabled() || !x->requires_grad())
        return out;

    auto x_edge = detail::ensure_grad_fn(x);
    auto bwd = std::make_shared<MaxPoolNdBackward<N>>();
    bwd->input_shapes_ = {x->shape()};
    bwd->out_shape_ = out->shape();
    bwd->dtype_ = x->dtype();
    bwd->device_ = x->device();
    bwd->input_tensors_ = {x};
    bwd->saved_argmax_ = std::move(saved_argmax);
    for (int i = 0; i < N; ++i) {
        bwd->K_[i] = K[i];
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
    }
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions({x->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

template <int N>
std::vector<Storage> MaxPoolNdBackward<N>::apply(Storage grad_out) {
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int C = static_cast<int>(this->input_shapes_[0][1]);
    int S[N], O[N];
    int O_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        O[i] = static_cast<int>(this->out_shape_[2 + i]);
        O_total *= O[i];
    }

    if (this->device_ == Device::GPU) {
        const auto& gG = std::get<GpuStorage>(grad_out);
        const auto& gA = std::get<GpuStorage>(this->saved_argmax_);
        if (!gG.arr || !gA.arr) {
            ErrorBuilder("max_pool backward").fail("null GPU array");
        }
        using SE = ::mlx::core::ShapeElem;
        const auto idt = ::mlx::core::int32;

        // Compute (k_i) multi-index from flat argmax: k_i = (argmax / suffix_K[i+1]) % K_i
        // For N=1: ki = argmax; for N=2: ki=argmax/KW, kj=argmax%KW; etc.
        // Then ih_i = oh_i * stride_i + k_i ; flat into padded (Sp)
        int Sp[N];
        for (int i = 0; i < N; ++i)
            Sp[i] = S[i] + 2 * this->pad_[i];

        // Build per-spatial-dim (k_i) tensors of shape (1, 1, *broadcasted O dims).
        // We'll compute ih_i from oh_i (range tensor) and k_i.
        // K-suffix products to decompose argmax.
        int K_suffix[N + 1];
        K_suffix[N] = 1;
        for (int i = N - 1; i >= 0; --i)
            K_suffix[i] = K_suffix[i + 1] * this->K_[i];

        // For each spatial dim i, build:
        //   k_i = (argmax / K_suffix[i+1]) % K_i
        //   ih_i = oh_i * stride_i + ki_i  (broadcast)
        //   flat_idx = ((ih_0 * Sp_1) + ih_1) * Sp_2 + ih_2 ...
        auto compute_ih = [&](int i) -> ::mlx::core::array {
            ::mlx::core::array div_arr(static_cast<std::int32_t>(K_suffix[i + 1]), idt);
            ::mlx::core::array mod_arr(static_cast<std::int32_t>(this->K_[i]), idt);
            auto ki = ::mlx::core::remainder(::mlx::core::floor_divide(*gA.arr, div_arr), mod_arr);
            ::mlx::core::Shape range_shape(N + 2, 1);
            range_shape[2 + i] = static_cast<SE>(O[i]);
            auto o_range = ::mlx::core::reshape(::mlx::core::arange(0, O[i], 1, idt), range_shape);
            ::mlx::core::array stride_arr(static_cast<std::int32_t>(this->stride_[i]), idt);
            return ::mlx::core::add(::mlx::core::multiply(o_range, stride_arr), ki);
        };

        ::mlx::core::array flat_idx = compute_ih(0);
        for (int i = 1; i < N; ++i) {
            ::mlx::core::array Sp_arr(static_cast<std::int32_t>(Sp[i]), idt);
            flat_idx = ::mlx::core::add(::mlx::core::multiply(flat_idx, Sp_arr), compute_ih(i));
        }

        const SE BC = static_cast<SE>(B) * static_cast<SE>(C);
        std::int64_t Sp_total = 1;
        for (int i = 0; i < N; ++i)
            Sp_total *= Sp[i];

        ::mlx::core::Shape flat_2d{BC, static_cast<SE>(O_total)};
        ::mlx::core::Shape pad_2d{BC, static_cast<SE>(Sp_total)};
        // Broadcast flat_idx to (B, C, *O) before reshape to (BC, O_total).
        ::mlx::core::Shape full_O;
        full_O.reserve(N + 2);
        full_O.push_back(static_cast<SE>(B));
        full_O.push_back(static_cast<SE>(C));
        for (int i = 0; i < N; ++i)
            full_O.push_back(static_cast<SE>(O[i]));
        flat_idx = ::mlx::core::broadcast_to(flat_idx, full_O);

        auto idx_2d = ::mlx::core::reshape(flat_idx, flat_2d);
        auto g_2d = ::mlx::core::reshape(*gG.arr, flat_2d);
        ::mlx::core::array zero_pad = ::mlx::core::zeros(pad_2d, gpu::to_mlx_dtype(this->dtype_));
        auto dx_pad_2d = ::mlx::core::scatter_add_axis(zero_pad, idx_2d, g_2d, /*axis=*/1);

        // Reshape to (B, C, *Sp) then crop padding.
        ::mlx::core::Shape full_Sp;
        full_Sp.reserve(N + 2);
        full_Sp.push_back(static_cast<SE>(B));
        full_Sp.push_back(static_cast<SE>(C));
        for (int i = 0; i < N; ++i)
            full_Sp.push_back(static_cast<SE>(Sp[i]));
        auto dx_pad = ::mlx::core::reshape(dx_pad_2d, full_Sp);

        ::mlx::core::Shape crop_lo(N + 2, 0);
        ::mlx::core::Shape crop_hi;
        crop_hi.reserve(N + 2);
        crop_hi.push_back(static_cast<SE>(B));
        crop_hi.push_back(static_cast<SE>(C));
        for (int i = 0; i < N; ++i) {
            crop_lo[2 + i] = static_cast<SE>(this->pad_[i]);
            crop_hi.push_back(static_cast<SE>(this->pad_[i] + S[i]));
        }
        auto dx = ::mlx::core::slice(dx_pad, crop_lo, crop_hi);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), this->dtype_)}};
    }

    int S_total = 1;
    for (int i = 0; i < N; ++i)
        S_total *= S[i];
    auto dx_cpu = alloc_bytes(static_cast<std::size_t>(B) * C * S_total, this->dtype_);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);

    const auto& g_cpu = std::get<CpuStorage>(grad_out);
    const auto& a_cpu = std::get<CpuStorage>(this->saved_argmax_);
    switch (this->dtype_) {
        case Dtype::F32:
            max_pool_backward_dispatch<N, float>(
                reinterpret_cast<const float*>(g_cpu.ptr.get()),
                reinterpret_cast<const std::int32_t*>(a_cpu.ptr.get()),
                reinterpret_cast<float*>(dx_cpu.ptr.get()), B, C, S, O);
            break;
        case Dtype::F64:
            max_pool_backward_dispatch<N, double>(
                reinterpret_cast<const double*>(g_cpu.ptr.get()),
                reinterpret_cast<const std::int32_t*>(a_cpu.ptr.get()),
                reinterpret_cast<double*>(dx_cpu.ptr.get()), B, C, S, O);
            break;
        default:
            ErrorBuilder("max_pool backward").not_implemented("dtype not supported");
    }
    return {Storage{std::move(dx_cpu)}};
}

// =====================================================================
// AvgPoolNd
// =====================================================================

template <int N>
TensorImplPtr AvgPoolNdBackward<N>::forward(const TensorImplPtr& x,
                                            const int (&K)[N],
                                            const int (&stride_in)[N],
                                            const int (&pad)[N]) {
    validate_input<N>(x, AvgPoolNdBackward<N>::schema_v1.name);
    int stride[N];
    for (int i = 0; i < N; ++i)
        stride[i] = (stride_in[i] == 0) ? K[i] : stride_in[i];

    const int B = static_cast<int>(x->shape()[0]);
    const int C = static_cast<int>(x->shape()[1]);
    int S[N], O[N];
    int O_total = 1, S_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape()[2 + i]);
        O[i] = compute_out(S[i], K[i], stride[i], pad[i]);
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape(), Shape{}, "avg_pool: output non-positive");
        O_total *= O[i];
        S_total *= S[i];
    }

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(C));
    for (int i = 0; i < N; ++i)
        out_shape.push_back(static_cast<std::int64_t>(O[i]));

    OpScopeFull scope{AvgPoolNdBackward<N>::schema_v1.name, x->device(), x->dtype(), out_shape};

    Storage out_storage;

    if (x->device() == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage());
        if (!gx.arr)
            ErrorBuilder("avg_pool").fail("null GPU input");
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(x->dtype()));
        std::vector<std::pair<int, int>> pad_widths;
        pad_widths.reserve(N + 2);
        pad_widths.emplace_back(0, 0);
        pad_widths.emplace_back(0, 0);
        for (int i = 0; i < N; ++i)
            pad_widths.emplace_back(pad[i], pad[i]);
        auto x_pad = ::mlx::core::pad(*gx.arr, pad_widths, zero);
        int Sp[N];
        for (int i = 0; i < N; ++i)
            Sp[i] = S[i] + 2 * pad[i];
        auto wins = build_window_view<N>(x_pad, B, C, Sp, O, K, stride);
        std::vector<int> kernel_axes;
        kernel_axes.reserve(N);
        for (int i = 0; i < N; ++i)
            kernel_axes.push_back(2 + N + i);
        auto y = ::mlx::core::mean(wins, kernel_axes, /*keepdims=*/false);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype())};
    } else {
        auto y_cpu = alloc_bytes(static_cast<std::size_t>(B) * C * O_total, x->dtype());
        const auto& x_cpu = std::get<CpuStorage>(x->storage());
        switch (x->dtype()) {
            case Dtype::F32:
                avg_pool_forward_dispatch<N, float>(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                                    reinterpret_cast<float*>(y_cpu.ptr.get()), B, C,
                                                    S, K, O, stride, pad);
                break;
            case Dtype::F64:
                avg_pool_forward_dispatch<N, double>(
                    reinterpret_cast<const double*>(x_cpu.ptr.get()),
                    reinterpret_cast<double*>(y_cpu.ptr.get()), B, C, S, K, O, stride, pad);
                break;
            default:
                ErrorBuilder("avg_pool").not_implemented("dtype not supported (F32/F64)");
        }
        out_storage = Storage{std::move(y_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);
    if (!GradMode::is_enabled() || !x->requires_grad())
        return out;

    auto x_edge = detail::ensure_grad_fn(x);
    auto bwd = std::make_shared<AvgPoolNdBackward<N>>();
    bwd->input_shapes_ = {x->shape()};
    bwd->out_shape_ = out->shape();
    bwd->dtype_ = x->dtype();
    bwd->device_ = x->device();
    bwd->input_tensors_ = {x};
    for (int i = 0; i < N; ++i) {
        bwd->K_[i] = K[i];
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
    }
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions({x->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

template <int N>
std::vector<Storage> AvgPoolNdBackward<N>::apply(Storage grad_out) {
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int C = static_cast<int>(this->input_shapes_[0][1]);
    int S[N], O[N];
    int O_total = 1, S_total = 1, K_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        O[i] = static_cast<int>(this->out_shape_[2 + i]);
        O_total *= O[i];
        S_total *= S[i];
        K_total *= this->K_[i];
    }

    if (this->device_ == Device::GPU) {
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gG.arr)
            ErrorBuilder("avg_pool backward").fail("null GPU array");
        using SE = ::mlx::core::ShapeElem;
        const auto idt = ::mlx::core::int32;
        const int K = K_total;
        int Sp[N];
        for (int i = 0; i < N; ++i)
            Sp[i] = S[i] + 2 * this->pad_[i];

        // Build (B, C, *O, K_total) updates = g/K_total broadcast.
        ::mlx::core::array inv_K(1.0 / static_cast<double>(K), gpu::to_mlx_dtype(this->dtype_));
        ::mlx::core::Shape g_with_k;
        g_with_k.reserve(N + 3);
        g_with_k.push_back(static_cast<SE>(B));
        g_with_k.push_back(static_cast<SE>(C));
        for (int i = 0; i < N; ++i)
            g_with_k.push_back(static_cast<SE>(O[i]));
        g_with_k.push_back(1);
        auto g_expanded = ::mlx::core::reshape(*gG.arr, g_with_k);
        auto g_per_cell = ::mlx::core::multiply(g_expanded, inv_K);
        ::mlx::core::Shape full_with_k = g_with_k;
        full_with_k[N + 2] = static_cast<SE>(K);
        auto updates = ::mlx::core::broadcast_to(g_per_cell, full_with_k);

        // Build flat_idx[..., k] = sum_i ((oh_i*stride_i + ki_i) cumulated over
        // padded spatial dims). Compute ki_i = (k / K_suffix[i+1]) % K_i.
        // k_range broadcastable to (B, C, *O, K) — store K in the last axis.
        ::mlx::core::Shape kr_shape(N + 3, 1);
        kr_shape[N + 2] = static_cast<SE>(K);
        ::mlx::core::array k_range =
            ::mlx::core::reshape(::mlx::core::arange(0, K, 1, idt), kr_shape);

        int K_suffix[N + 1];
        K_suffix[N] = 1;
        for (int i = N - 1; i >= 0; --i)
            K_suffix[i] = K_suffix[i + 1] * this->K_[i];

        auto compute_ih = [&](int i) -> ::mlx::core::array {
            ::mlx::core::array div_arr(static_cast<std::int32_t>(K_suffix[i + 1]), idt);
            ::mlx::core::array mod_arr(static_cast<std::int32_t>(this->K_[i]), idt);
            auto ki = ::mlx::core::remainder(::mlx::core::floor_divide(k_range, div_arr), mod_arr);
            ::mlx::core::Shape range_shape(N + 3, 1);
            range_shape[2 + i] = static_cast<SE>(O[i]);
            auto o_range = ::mlx::core::reshape(::mlx::core::arange(0, O[i], 1, idt), range_shape);
            ::mlx::core::array stride_arr(static_cast<std::int32_t>(this->stride_[i]), idt);
            return ::mlx::core::add(::mlx::core::multiply(o_range, stride_arr), ki);
        };
        ::mlx::core::array flat_idx = compute_ih(0);
        for (int i = 1; i < N; ++i) {
            ::mlx::core::array Sp_arr(static_cast<std::int32_t>(Sp[i]), idt);
            flat_idx = ::mlx::core::add(::mlx::core::multiply(flat_idx, Sp_arr), compute_ih(i));
        }
        flat_idx = ::mlx::core::broadcast_to(flat_idx, full_with_k);

        const SE BC = static_cast<SE>(B) * static_cast<SE>(C);
        std::int64_t Sp_total = 1;
        for (int i = 0; i < N; ++i)
            Sp_total *= Sp[i];
        ::mlx::core::Shape flat_2d{BC, static_cast<SE>(O_total) * static_cast<SE>(K)};
        ::mlx::core::Shape pad_2d{BC, static_cast<SE>(Sp_total)};
        auto idx_2d = ::mlx::core::reshape(flat_idx, flat_2d);
        auto upd_2d = ::mlx::core::reshape(updates, flat_2d);
        ::mlx::core::array zero_pad = ::mlx::core::zeros(pad_2d, gpu::to_mlx_dtype(this->dtype_));
        auto dx_pad_2d = ::mlx::core::scatter_add_axis(zero_pad, idx_2d, upd_2d, /*axis=*/1);

        ::mlx::core::Shape full_Sp;
        full_Sp.reserve(N + 2);
        full_Sp.push_back(static_cast<SE>(B));
        full_Sp.push_back(static_cast<SE>(C));
        for (int i = 0; i < N; ++i)
            full_Sp.push_back(static_cast<SE>(Sp[i]));
        auto dx_pad = ::mlx::core::reshape(dx_pad_2d, full_Sp);

        ::mlx::core::Shape crop_lo(N + 2, 0);
        ::mlx::core::Shape crop_hi;
        crop_hi.reserve(N + 2);
        crop_hi.push_back(static_cast<SE>(B));
        crop_hi.push_back(static_cast<SE>(C));
        for (int i = 0; i < N; ++i) {
            crop_lo[2 + i] = static_cast<SE>(this->pad_[i]);
            crop_hi.push_back(static_cast<SE>(this->pad_[i] + S[i]));
        }
        auto dx = ::mlx::core::slice(dx_pad, crop_lo, crop_hi);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), this->dtype_)}};
    }

    auto dx_cpu = alloc_bytes(static_cast<std::size_t>(B) * C * S_total, this->dtype_);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);
    switch (this->dtype_) {
        case Dtype::F32:
            avg_pool_backward_dispatch<N, float>(reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                                 reinterpret_cast<float*>(dx_cpu.ptr.get()), B, C,
                                                 S, this->K_, O, this->stride_, this->pad_);
            break;
        case Dtype::F64:
            avg_pool_backward_dispatch<N, double>(reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                                  reinterpret_cast<double*>(dx_cpu.ptr.get()), B, C,
                                                  S, this->K_, O, this->stride_, this->pad_);
            break;
        default:
            ErrorBuilder("avg_pool backward").not_implemented("dtype not supported");
    }
    return {Storage{std::move(dx_cpu)}};
}

// Explicit instantiations
template class MaxPoolNdBackward<1>;
template class MaxPoolNdBackward<2>;
template class MaxPoolNdBackward<3>;
template class AvgPoolNdBackward<1>;
template class AvgPoolNdBackward<2>;
template class AvgPoolNdBackward<3>;

// Factories
TensorImplPtr max_pool1d_op(const TensorImplPtr& x, int KL, int sl, int pl) {
    int K[1]{KL};
    int s[1]{sl};
    int p[1]{pl};
    return MaxPool1dBackward::forward(x, K, s, p);
}
TensorImplPtr max_pool2d_op(
    const TensorImplPtr& x, int KH, int KW, int sh, int sw, int ph, int pw) {
    int K[2]{KH, KW};
    int s[2]{sh, sw};
    int p[2]{ph, pw};
    return MaxPool2dBackward::forward(x, K, s, p);
}
TensorImplPtr max_pool3d_op(const TensorImplPtr& x,
                            int KD,
                            int KH,
                            int KW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw) {
    int K[3]{KD, KH, KW};
    int s[3]{sd, sh, sw};
    int p[3]{pd, ph, pw};
    return MaxPool3dBackward::forward(x, K, s, p);
}
TensorImplPtr avg_pool1d_op(const TensorImplPtr& x, int KL, int sl, int pl) {
    int K[1]{KL};
    int s[1]{sl};
    int p[1]{pl};
    return AvgPool1dBackward::forward(x, K, s, p);
}
TensorImplPtr avg_pool2d_op(
    const TensorImplPtr& x, int KH, int KW, int sh, int sw, int ph, int pw) {
    int K[2]{KH, KW};
    int s[2]{sh, sw};
    int p[2]{ph, pw};
    return AvgPool2dBackward::forward(x, K, s, p);
}
TensorImplPtr avg_pool3d_op(const TensorImplPtr& x,
                            int KD,
                            int KH,
                            int KW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw) {
    int K[3]{KD, KH, KW};
    int s[3]{sd, sh, sw};
    int p[3]{pd, ph, pw};
    return AvgPool3dBackward::forward(x, K, s, p);
}

LUCID_REGISTER_OP(MaxPool1dBackward)
LUCID_REGISTER_OP(MaxPool2dBackward)
LUCID_REGISTER_OP(MaxPool3dBackward)
LUCID_REGISTER_OP(AvgPool1dBackward)
LUCID_REGISTER_OP(AvgPool2dBackward)
LUCID_REGISTER_OP(AvgPool3dBackward)

}  // namespace lucid
