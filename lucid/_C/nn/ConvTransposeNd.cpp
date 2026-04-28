#include "ConvTransposeNd.h"

#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/cpu/Blas.h"
#include "../backend/cpu/Im2Col.h"
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

template <>
const OpSchema ConvTranspose1dBackward::schema_v1{"conv_transpose1d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema ConvTranspose2dBackward::schema_v1{"conv_transpose2d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema ConvTranspose3dBackward::schema_v1{"conv_transpose3d", 1, AmpPolicy::Promote, true};

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

template <typename T>
void add_bias_chw(T* y, const T* bias, int C_out, int O_total) {
    for (int c = 0; c < C_out; ++c) {
        const T bv = bias[c];
        T* row = y + c * O_total;
        for (int i = 0; i < O_total; ++i)
            row[i] += bv;
    }
}

// im2col / col2im dispatch by spatial rank (same as ConvNd).
template <int N, typename T>
void im2col_dispatch(const T* x,
                     T* cols,
                     int C,
                     const int* S,
                     const int* K,
                     const int* O,
                     const int* stride,
                     const int* pad);

template <>
void im2col_dispatch<1, float>(const float* x,
                               float* cols,
                               int C,
                               const int* S,
                               const int* K,
                               const int* O,
                               const int* stride,
                               const int* pad) {
    backend::cpu::im2col_1d_f32(x, cols, C, S[0], K[0], O[0], stride[0], pad[0], 1);
}
template <>
void im2col_dispatch<1, double>(const double* x,
                                double* cols,
                                int C,
                                const int* S,
                                const int* K,
                                const int* O,
                                const int* stride,
                                const int* pad) {
    backend::cpu::im2col_1d_f64(x, cols, C, S[0], K[0], O[0], stride[0], pad[0], 1);
}
template <>
void im2col_dispatch<2, float>(const float* x,
                               float* cols,
                               int C,
                               const int* S,
                               const int* K,
                               const int* O,
                               const int* stride,
                               const int* pad) {
    backend::cpu::im2col_f32(x, cols, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                             pad[0], pad[1], 1, 1);
}
template <>
void im2col_dispatch<2, double>(const double* x,
                                double* cols,
                                int C,
                                const int* S,
                                const int* K,
                                const int* O,
                                const int* stride,
                                const int* pad) {
    backend::cpu::im2col_f64(x, cols, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                             pad[0], pad[1], 1, 1);
}
template <>
void im2col_dispatch<3, float>(const float* x,
                               float* cols,
                               int C,
                               const int* S,
                               const int* K,
                               const int* O,
                               const int* stride,
                               const int* pad) {
    backend::cpu::im2col_3d_f32(x, cols, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                                stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1);
}
template <>
void im2col_dispatch<3, double>(const double* x,
                                double* cols,
                                int C,
                                const int* S,
                                const int* K,
                                const int* O,
                                const int* stride,
                                const int* pad) {
    backend::cpu::im2col_3d_f64(x, cols, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                                stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1);
}

template <int N, typename T>
void col2im_dispatch(const T* cols,
                     T* dx,
                     int C,
                     const int* S,
                     const int* K,
                     const int* O,
                     const int* stride,
                     const int* pad);

template <>
void col2im_dispatch<1, float>(const float* cols,
                               float* dx,
                               int C,
                               const int* S,
                               const int* K,
                               const int* O,
                               const int* stride,
                               const int* pad) {
    backend::cpu::col2im_1d_f32(cols, dx, C, S[0], K[0], O[0], stride[0], pad[0], 1);
}
template <>
void col2im_dispatch<1, double>(const double* cols,
                                double* dx,
                                int C,
                                const int* S,
                                const int* K,
                                const int* O,
                                const int* stride,
                                const int* pad) {
    backend::cpu::col2im_1d_f64(cols, dx, C, S[0], K[0], O[0], stride[0], pad[0], 1);
}
template <>
void col2im_dispatch<2, float>(const float* cols,
                               float* dx,
                               int C,
                               const int* S,
                               const int* K,
                               const int* O,
                               const int* stride,
                               const int* pad) {
    backend::cpu::col2im_f32(cols, dx, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                             pad[0], pad[1], 1, 1);
}
template <>
void col2im_dispatch<2, double>(const double* cols,
                                double* dx,
                                int C,
                                const int* S,
                                const int* K,
                                const int* O,
                                const int* stride,
                                const int* pad) {
    backend::cpu::col2im_f64(cols, dx, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                             pad[0], pad[1], 1, 1);
}
template <>
void col2im_dispatch<3, float>(const float* cols,
                               float* dx,
                               int C,
                               const int* S,
                               const int* K,
                               const int* O,
                               const int* stride,
                               const int* pad) {
    backend::cpu::col2im_3d_f32(cols, dx, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                                stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1);
}
template <>
void col2im_dispatch<3, double>(const double* cols,
                                double* dx,
                                int C,
                                const int* S,
                                const int* K,
                                const int* O,
                                const int* stride,
                                const int* pad) {
    backend::cpu::col2im_3d_f64(cols, dx, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                                stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1);
}

template <int N>
std::vector<int> nchw_to_nhwc_perm() {
    std::vector<int> p;
    p.reserve(N + 2);
    p.push_back(0);
    for (int i = 0; i < N; ++i)
        p.push_back(2 + i);
    p.push_back(1);
    return p;
}
template <int N>
std::vector<int> nhwc_to_nchw_perm() {
    std::vector<int> p;
    p.reserve(N + 2);
    p.push_back(0);
    p.push_back(N + 1);
    for (int i = 0; i < N; ++i)
        p.push_back(1 + i);
    return p;
}
// Our W is (Cin, Cout, *K). MLX conv_transpose2d wants (Cout_t, *K, Cin_t)
// where Cin_t = our Cin (input to transpose) and Cout_t = our Cout (output
// of transpose). So permute (Cin, Cout, *K) → (Cout, *K, Cin) via
// {1, 2..N+1, 0}.
template <int N>
std::vector<int> w_to_mlx_transpose_perm() {
    std::vector<int> p;
    p.reserve(N + 2);
    p.push_back(1);
    for (int i = 0; i < N; ++i)
        p.push_back(2 + i);
    p.push_back(0);
    return p;
}

template <int N>
::mlx::core::array mlx_conv_transpose_nd(const ::mlx::core::array& x_nhwc,
                                         const ::mlx::core::array& W_nhwc,
                                         const int (&stride)[N],
                                         const int (&pad)[N],
                                         const int (&opad)[N]);

template <>
::mlx::core::array mlx_conv_transpose_nd<1>(const ::mlx::core::array& x,
                                            const ::mlx::core::array& W,
                                            const int (&s)[1],
                                            const int (&p)[1],
                                            const int (&op)[1]) {
    return ::mlx::core::conv_transpose1d(x, W, s[0], p[0], 1, op[0]);
}
template <>
::mlx::core::array mlx_conv_transpose_nd<2>(const ::mlx::core::array& x,
                                            const ::mlx::core::array& W,
                                            const int (&s)[2],
                                            const int (&p)[2],
                                            const int (&op)[2]) {
    return ::mlx::core::conv_transpose2d(x, W, std::pair<int, int>{s[0], s[1]},
                                         std::pair<int, int>{p[0], p[1]}, std::pair<int, int>{1, 1},
                                         std::pair<int, int>{op[0], op[1]});
}
template <>
::mlx::core::array mlx_conv_transpose_nd<3>(const ::mlx::core::array& x,
                                            const ::mlx::core::array& W,
                                            const int (&s)[3],
                                            const int (&p)[3],
                                            const int (&op)[3]) {
    return ::mlx::core::conv_transpose3d(x, W, std::tuple<int, int, int>{s[0], s[1], s[2]},
                                         std::tuple<int, int, int>{p[0], p[1], p[2]},
                                         std::tuple<int, int, int>{1, 1, 1},
                                         std::tuple<int, int, int>{op[0], op[1], op[2]});
}

template <int N>
::mlx::core::array mlx_conv_nd(const ::mlx::core::array& x_nhwc,
                               const ::mlx::core::array& W_nhwc,
                               const int (&stride)[N],
                               const int (&pad)[N]);

template <>
::mlx::core::array mlx_conv_nd<1>(const ::mlx::core::array& x,
                                  const ::mlx::core::array& W,
                                  const int (&s)[1],
                                  const int (&p)[1]) {
    return ::mlx::core::conv1d(x, W, s[0], p[0]);
}
template <>
::mlx::core::array mlx_conv_nd<2>(const ::mlx::core::array& x,
                                  const ::mlx::core::array& W,
                                  const int (&s)[2],
                                  const int (&p)[2]) {
    return ::mlx::core::conv2d(x, W, std::pair<int, int>{s[0], s[1]},
                               std::pair<int, int>{p[0], p[1]});
}
template <>
::mlx::core::array mlx_conv_nd<3>(const ::mlx::core::array& x,
                                  const ::mlx::core::array& W,
                                  const int (&s)[3],
                                  const int (&p)[3]) {
    return ::mlx::core::conv3d(x, W, std::tuple<int, int, int>{s[0], s[1], s[2]},
                               std::tuple<int, int, int>{p[0], p[1], p[2]});
}

}  // namespace

template <int N>
TensorImplPtr ConvTransposeNdBackward<N>::forward(const TensorImplPtr& x,
                                                  const TensorImplPtr& W,
                                                  const TensorImplPtr& b,
                                                  const int (&stride)[N],
                                                  const int (&pad)[N],
                                                  const int (&opad)[N]) {
    if (!x || !W || !b)
        ErrorBuilder("conv_transpose").fail("null input");
    if (x->dtype_ != W->dtype_ || x->dtype_ != b->dtype_)
        throw DtypeMismatch(std::string(dtype_name(x->dtype_)), std::string(dtype_name(W->dtype_)),
                            "conv_transpose");
    if (x->device_ != W->device_ || x->device_ != b->device_)
        throw DeviceMismatch(std::string(device_name(x->device_)),
                             std::string(device_name(W->device_)), "conv_transpose");
    if (x->device_ == Device::CPU &&
        (!x->is_contiguous() || !W->is_contiguous() || !b->is_contiguous()))
        ErrorBuilder("conv_transpose")
            .not_implemented("non-contiguous input not supported (call .contiguous() first)");
    if (static_cast<int>(x->shape_.size()) != N + 2)
        throw ShapeMismatch(x->shape_, Shape{}, "conv_transpose: x rank mismatch");
    if (static_cast<int>(W->shape_.size()) != N + 2)
        throw ShapeMismatch(W->shape_, Shape{}, "conv_transpose: W rank mismatch");
    if (b->shape_.size() != 1)
        throw ShapeMismatch(b->shape_, Shape{}, "conv_transpose: b must be 1-D");

    const int B = static_cast<int>(x->shape_[0]);
    const int Cin = static_cast<int>(x->shape_[1]);
    const int Cw = static_cast<int>(W->shape_[0]);
    const int Cout = static_cast<int>(W->shape_[1]);
    if (Cw != Cin)
        throw ShapeMismatch(W->shape_, x->shape_, "conv_transpose: C_in mismatch");
    if (b->shape_[0] != Cout)
        throw ShapeMismatch(b->shape_, W->shape_, "conv_transpose: bias C_out mismatch");

    int S[N], K[N], O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape_[2 + i]);
        K[i] = static_cast<int>(W->shape_[2 + i]);
        O[i] = (S[i] - 1) * stride[i] - 2 * pad[i] + K[i] + opad[i];
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape_, W->shape_, "conv_transpose: output shape non-positive");
    }

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(Cout));
    for (int i = 0; i < N; ++i)
        out_shape.push_back(static_cast<std::int64_t>(O[i]));

    int O_total = 1, S_total = 1, K_total = 1;
    for (int i = 0; i < N; ++i) {
        O_total *= O[i];
        S_total *= S[i];
        K_total *= K[i];
    }

    OpScopeFull scope{ConvTransposeNdBackward<N>::schema_v1.name, x->device_, x->dtype_, out_shape};
    scope.set_flops(static_cast<std::int64_t>(2) * B * Cout * O_total * Cin * K_total);

    Storage out_storage;

    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto& gW = std::get<GpuStorage>(W->storage_);
        const auto& gb = std::get<GpuStorage>(b->storage_);
        if (!gx.arr || !gW.arr || !gb.arr) {
            ErrorBuilder("conv_transpose").fail("null GPU input");
        }
        auto x_nhwc = ::mlx::core::transpose(*gx.arr, nchw_to_nhwc_perm<N>());
        auto W_nhwc = ::mlx::core::transpose(*gW.arr, w_to_mlx_transpose_perm<N>());
        auto y_nhwc = mlx_conv_transpose_nd<N>(x_nhwc, W_nhwc, stride, pad, opad);
        ::mlx::core::Shape b_brd(N + 2, 1);
        b_brd[N + 1] = static_cast<::mlx::core::ShapeElem>(Cout);
        auto b_view = ::mlx::core::reshape(*gb.arr, b_brd);
        y_nhwc = ::mlx::core::add(y_nhwc, b_view);
        auto y = ::mlx::core::contiguous(::mlx::core::transpose(y_nhwc, nhwc_to_nchw_perm<N>()));
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype_)};
    } else {
        // CPU forward: per-batch (W^T @ x_2d) → cols; col2im → y[b]; add bias.
        // x_2d : (Cin, prod(S))
        // W_2d : (Cin, Cout * prod(K))     — W has shape (Cin, Cout, *K)
        // cols : (Cout * prod(K), prod(S)) = W_2d^T @ x_2d
        //                                 — sgemm(transA=true, transB=false, M=Cout*K, N=S, K=Cin)
        // col2im: y_b (Cout, *O) ← cols, with stride/pad params (S↔O reversed
        //         in the col2im signature: it takes "kernel iterates in
        //         (KH,KW)" and "spatial iterates in (OH,OW)"; for our case
        //         we iterate input position s ∈ [0, S), kernel ∈ [0, K),
        //         and write y_b at o = s·stride + k − p).
        const int K_flat = Cout * K_total;

        auto out_cpu = allocate_size(static_cast<std::size_t>(B) * Cout * O_total, x->dtype_);
        if (out_cpu.nbytes)
            std::memset(out_cpu.ptr.get(), 0, out_cpu.nbytes);
        auto cols_cpu = allocate_size(static_cast<std::size_t>(K_flat) * S_total, x->dtype_);

        const auto& x_cpu = std::get<CpuStorage>(x->storage_);
        const auto& W_cpu = std::get<CpuStorage>(W->storage_);
        const auto& b_cpu = std::get<CpuStorage>(b->storage_);

        for (int bi = 0; bi < B; ++bi) {
            switch (x->dtype_) {
                case Dtype::F32: {
                    auto* xp = reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                               static_cast<std::size_t>(bi) * Cin * S_total;
                    auto* Wp = reinterpret_cast<const float*>(W_cpu.ptr.get());
                    auto* cp = reinterpret_cast<float*>(cols_cpu.ptr.get());
                    auto* yp = reinterpret_cast<float*>(out_cpu.ptr.get()) +
                               static_cast<std::size_t>(bi) * Cout * O_total;
                    // cols[K_flat, S_total] = W_2d^T (Cin, K_flat)^T @ x_2d (Cin, S_total)
                    backend::cpu::sgemm(
                        /*transA=*/true, /*transB=*/false, K_flat, S_total, Cin, 1.0f, Wp, K_flat,
                        xp, S_total, 0.0f, cp, S_total);
                    col2im_dispatch<N, float>(cp, yp, Cout, O, K, S, stride, pad);
                    add_bias_chw<float>(yp, reinterpret_cast<const float*>(b_cpu.ptr.get()), Cout,
                                        O_total);
                    break;
                }
                case Dtype::F64: {
                    auto* xp = reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                               static_cast<std::size_t>(bi) * Cin * S_total;
                    auto* Wp = reinterpret_cast<const double*>(W_cpu.ptr.get());
                    auto* cp = reinterpret_cast<double*>(cols_cpu.ptr.get());
                    auto* yp = reinterpret_cast<double*>(out_cpu.ptr.get()) +
                               static_cast<std::size_t>(bi) * Cout * O_total;
                    backend::cpu::dgemm(true, false, K_flat, S_total, Cin, 1.0, Wp, K_flat, xp,
                                        S_total, 0.0, cp, S_total);
                    col2im_dispatch<N, double>(cp, yp, Cout, O, K, S, stride, pad);
                    add_bias_chw<double>(yp, reinterpret_cast<const double*>(b_cpu.ptr.get()), Cout,
                                         O_total);
                    break;
                }
                default:
                    ErrorBuilder("conv_transpose").not_implemented("dtype not supported (F32/F64)");
            }
        }
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape), x->dtype_,
                                            x->device_, false);

    if (!GradMode::is_enabled() || !(x->requires_grad_ || W->requires_grad_ || b->requires_grad_)) {
        return out;
    }

    auto x_edge = detail::ensure_grad_fn(x);
    auto W_edge = detail::ensure_grad_fn(W);
    auto b_edge = detail::ensure_grad_fn(b);

    auto bwd = std::make_shared<ConvTransposeNdBackward<N>>();
    bwd->input_shapes_ = {x->shape_, W->shape_, b->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = x->dtype_;
    bwd->device_ = x->device_;
    bwd->input_tensors_ = {x, W, b};
    bwd->saved_inputs_ = {x->storage_, W->storage_, b->storage_};
    for (int i = 0; i < N; ++i) {
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
        bwd->opad_[i] = opad[i];
    }
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(W_edge, 0), Edge(b_edge, 0)});
    bwd->set_saved_versions({x->version_, W->version_, b->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

template <int N>
std::vector<Storage> ConvTransposeNdBackward<N>::apply(Storage grad_out) {
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int Cin = static_cast<int>(this->input_shapes_[0][1]);
    const int Cout = static_cast<int>(this->input_shapes_[1][1]);
    int S[N], K[N], O[N];
    int S_total = 1, K_total = 1, O_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        K[i] = static_cast<int>(this->input_shapes_[1][2 + i]);
        O[i] = static_cast<int>(this->out_shape_[2 + i]);
        S_total *= S[i];
        K_total *= K[i];
        O_total *= O[i];
    }

    if (this->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(this->saved_inputs_[0]);
        const auto& gW = std::get<GpuStorage>(this->saved_inputs_[1]);
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gx.arr || !gW.arr || !gG.arr) {
            ErrorBuilder("conv_transpose backward").fail("null GPU array");
        }

        // db = sum(grad, axes={0, 2..N+1})
        std::vector<int> db_axes;
        db_axes.reserve(N + 1);
        db_axes.push_back(0);
        for (int i = 0; i < N; ++i)
            db_axes.push_back(2 + i);
        auto db = ::mlx::core::sum(*gG.arr, db_axes, /*keepdims=*/false);

        // dx = regular conv(grad, W_perm, stride, pad). For mlx::conv_nd, the
        // weight format is (Cout_of_conv, *K, Cin_of_conv). Here Cin_of_conv
        // = our Co (grad channels), Cout_of_conv = our Ci (dx output channels).
        // Our W is (Ci, Co, *K). To get (Ci, *K, Co) we permute via
        // {0, 2..N+1, 1}.
        std::vector<int> w_dx_perm;
        w_dx_perm.push_back(0);
        for (int i = 0; i < N; ++i)
            w_dx_perm.push_back(2 + i);
        w_dx_perm.push_back(1);
        auto grad_nhwc = ::mlx::core::transpose(*gG.arr, nchw_to_nhwc_perm<N>());
        auto W_dx_nhwc = ::mlx::core::transpose(*gW.arr, w_dx_perm);
        auto dx_nhwc = mlx_conv_nd<N>(grad_nhwc, W_dx_nhwc, this->stride_, this->pad_);
        auto dx = ::mlx::core::contiguous(::mlx::core::transpose(dx_nhwc, nhwc_to_nchw_perm<N>()));

        // dW via conv_general dilation trick (x and grad swapped vs ConvForward).
        //   input  = grad permuted  (B, Co, *O) → (Co, *O, B)
        //   kernel = x    permuted  (B, Ci, *S) → (Ci, *S, B)
        //   stride=1, padding=p, kernel_dilation=stride
        //   output: (Co, *K, Ci)  → permute to (Cin, Cout, *K) NCHW
        std::vector<int> perm_axes;
        perm_axes.push_back(1);
        for (int i = 0; i < N; ++i)
            perm_axes.push_back(2 + i);
        perm_axes.push_back(0);
        auto g_perm = ::mlx::core::transpose(*gG.arr, perm_axes);
        auto x_perm = ::mlx::core::transpose(*gx.arr, perm_axes);
        std::vector<int> conv_stride(N, 1);
        std::vector<int> conv_pad(N);
        std::vector<int> conv_kdil(N);
        std::vector<int> conv_idil(N, 1);
        for (int i = 0; i < N; ++i) {
            conv_pad[i] = this->pad_[i];
            conv_kdil[i] = this->stride_[i];
        }
        auto dW_perm = ::mlx::core::conv_general(g_perm, x_perm, conv_stride, conv_pad, conv_pad,
                                                 conv_kdil, conv_idil);
        // Crop to (Co, *K, Ci).
        using SE = ::mlx::core::ShapeElem;
        ::mlx::core::Shape crop_lo(N + 2, 0);
        ::mlx::core::Shape crop_hi;
        crop_hi.push_back(static_cast<SE>(Cout));
        for (int i = 0; i < N; ++i)
            crop_hi.push_back(static_cast<SE>(K[i]));
        crop_hi.push_back(static_cast<SE>(Cin));
        dW_perm = ::mlx::core::slice(dW_perm, crop_lo, crop_hi);
        // (Co, *K, Ci) → (Cin, Cout, *K). axes [N+1, 0, 1..N]
        std::vector<int> dW_back;
        dW_back.push_back(N + 1);  // Cin
        dW_back.push_back(0);      // Cout
        for (int i = 0; i < N; ++i)
            dW_back.push_back(1 + i);
        auto dW = ::mlx::core::contiguous(::mlx::core::transpose(dW_perm, dW_back));

        return {Storage{gpu::wrap_mlx_array(std::move(dx), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dW), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(db), this->dtype_)}};
    }

    // ---- CPU backward ----
    // dx: regular conv-forward of grad with W (axes 0/1 swapped) — i.e.
    //     dx[b, ci, *s] = sum_{co, *k} W[ci, co, *k] * grad[b, co, s*stride + k - p]
    //     We compute via per-batch im2col on grad + sgemm.
    //   im2col(grad[b], grad_cols, Cout, *O, *K, *S, stride, pad)
    //     → cols (Cout · prod(K), prod(S))
    //   W_2d (Cin, Cout · prod(K)) — W is (Cin, Cout, *K)
    //   dx_2d (Cin, prod(S)) = W_2d @ cols
    const int K_flat = Cout * K_total;

    auto dx_cpu = allocate_size(static_cast<std::size_t>(B) * Cin * S_total, this->dtype_);
    auto dW_cpu = allocate_size(static_cast<std::size_t>(Cin) * K_flat, this->dtype_);
    auto db_cpu = allocate_size(static_cast<std::size_t>(Cout), this->dtype_);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    if (dW_cpu.nbytes)
        std::memset(dW_cpu.ptr.get(), 0, dW_cpu.nbytes);
    if (db_cpu.nbytes)
        std::memset(db_cpu.ptr.get(), 0, db_cpu.nbytes);

    auto cols_cpu = allocate_size(static_cast<std::size_t>(K_flat) * S_total, this->dtype_);

    const auto& x_cpu = std::get<CpuStorage>(this->saved_inputs_[0]);
    const auto& W_cpu = std::get<CpuStorage>(this->saved_inputs_[1]);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);

    for (int bi = 0; bi < B; ++bi) {
        switch (this->dtype_) {
            case Dtype::F32: {
                auto* xp = reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                           static_cast<std::size_t>(bi) * Cin * S_total;
                auto* gp = reinterpret_cast<const float*>(g_cpu.ptr.get()) +
                           static_cast<std::size_t>(bi) * Cout * O_total;
                auto* dxp = reinterpret_cast<float*>(dx_cpu.ptr.get()) +
                            static_cast<std::size_t>(bi) * Cin * S_total;
                auto* cp = reinterpret_cast<float*>(cols_cpu.ptr.get());

                // im2col on grad: kernel iterates *K, spatial iterates *S.
                // Layout: cols[(co * prod(K) + flat(k)) * prod(S) + flat(s)].
                im2col_dispatch<N, float>(gp, cp, Cout, O, K, S, this->stride_, this->pad_);

                // dx_2d (Cin, prod(S)) = W_2d (Cin, K_flat) @ cols (K_flat, prod(S))
                backend::cpu::sgemm(false, false, Cin, S_total, K_flat, 1.0f,
                                    reinterpret_cast<const float*>(W_cpu.ptr.get()), K_flat, cp,
                                    S_total, 0.0f, dxp, S_total);

                // dW_2d (Cin, K_flat) += x_2d (Cin, S_total) @ cols^T (S_total, K_flat)
                backend::cpu::sgemm(false, true, Cin, K_flat, S_total, 1.0f, xp, S_total, cp,
                                    S_total, 1.0f, reinterpret_cast<float*>(dW_cpu.ptr.get()),
                                    K_flat);

                {
                    auto* dbp = reinterpret_cast<float*>(db_cpu.ptr.get());
                    for (int co = 0; co < Cout; ++co) {
                        const float* row = gp + co * O_total;
                        float s = 0.f;
                        for (int j = 0; j < O_total; ++j)
                            s += row[j];
                        dbp[co] += s;
                    }
                }
                break;
            }
            case Dtype::F64: {
                auto* xp = reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                           static_cast<std::size_t>(bi) * Cin * S_total;
                auto* gp = reinterpret_cast<const double*>(g_cpu.ptr.get()) +
                           static_cast<std::size_t>(bi) * Cout * O_total;
                auto* dxp = reinterpret_cast<double*>(dx_cpu.ptr.get()) +
                            static_cast<std::size_t>(bi) * Cin * S_total;
                auto* cp = reinterpret_cast<double*>(cols_cpu.ptr.get());

                im2col_dispatch<N, double>(gp, cp, Cout, O, K, S, this->stride_, this->pad_);
                backend::cpu::dgemm(false, false, Cin, S_total, K_flat, 1.0,
                                    reinterpret_cast<const double*>(W_cpu.ptr.get()), K_flat, cp,
                                    S_total, 0.0, dxp, S_total);
                backend::cpu::dgemm(false, true, Cin, K_flat, S_total, 1.0, xp, S_total, cp,
                                    S_total, 1.0, reinterpret_cast<double*>(dW_cpu.ptr.get()),
                                    K_flat);
                {
                    auto* dbp = reinterpret_cast<double*>(db_cpu.ptr.get());
                    for (int co = 0; co < Cout; ++co) {
                        const double* row = gp + co * O_total;
                        double s = 0.0;
                        for (int j = 0; j < O_total; ++j)
                            s += row[j];
                        dbp[co] += s;
                    }
                }
                break;
            }
            default:
                ErrorBuilder("conv_transpose backward").not_implemented("dtype not supported");
        }
    }
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dW_cpu)}, Storage{std::move(db_cpu)}};
}

template class ConvTransposeNdBackward<1>;
template class ConvTransposeNdBackward<2>;
template class ConvTransposeNdBackward<3>;

TensorImplPtr conv_transpose1d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int sl,
                                  int pl,
                                  int opl) {
    int s[1]{sl};
    int p[1]{pl};
    int op[1]{opl};
    return ConvTranspose1dBackward::forward(x, W, b, s, p, op);
}
TensorImplPtr conv_transpose2d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int sh,
                                  int sw,
                                  int ph,
                                  int pw,
                                  int oph,
                                  int opw) {
    int s[2]{sh, sw};
    int p[2]{ph, pw};
    int op[2]{oph, opw};
    return ConvTranspose2dBackward::forward(x, W, b, s, p, op);
}
TensorImplPtr conv_transpose3d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int sd,
                                  int sh,
                                  int sw,
                                  int pd,
                                  int ph,
                                  int pw,
                                  int opd,
                                  int oph,
                                  int opw) {
    int s[3]{sd, sh, sw};
    int p[3]{pd, ph, pw};
    int op[3]{opd, oph, opw};
    return ConvTranspose3dBackward::forward(x, W, b, s, p, op);
}

LUCID_REGISTER_OP(ConvTranspose1dBackward)
LUCID_REGISTER_OP(ConvTranspose2dBackward)
LUCID_REGISTER_OP(ConvTranspose3dBackward)

}  // namespace lucid
