#include "ConvNd.h"

#include <cstring>
#include <numeric>
#include <vector>

#include <mlx/ops.h>

#include "../backend/cpu/Blas.h"
#include "../backend/cpu/Im2Col.h"
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

template <> const OpSchema Conv1dBackward::schema_v1{
    "conv1d", 1, AmpPolicy::Promote, true};
template <> const OpSchema Conv2dBackward::schema_v1{
    "conv2d", 1, AmpPolicy::Promote, true};
template <> const OpSchema Conv3dBackward::schema_v1{
    "conv3d", 1, AmpPolicy::Promote, true};

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype  = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr    = allocate_aligned_bytes(s.nbytes);
    return s;
}

inline int compute_out(int S, int K, int stride, int pad, int dilation) {
    const int eff = dilation * (K - 1) + 1;
    return (S + 2 * pad - eff) / stride + 1;
}

template <typename T>
void add_bias_chw(T* y, const T* bias, int C_out, int O_total) {
    for (int c = 0; c < C_out; ++c) {
        const T bv = bias[c];
        T* row = y + c * O_total;
        for (int i = 0; i < O_total; ++i) row[i] += bv;
    }
}

template <int N, typename T>
void im2col_dispatch(const T* x, T* cols,
                     int C,
                     const int* S_in,
                     const int* K_in,
                     const int* O_in,
                     const int* stride_in,
                     const int* pad_in,
                     const int* dilation_in);

template <> void im2col_dispatch<1, float>(
    const float* x, float* cols, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::im2col_1d_f32(x, cols, C, S[0], K[0], O[0],
                                stride[0], pad[0], dil[0]);
}
template <> void im2col_dispatch<1, double>(
    const double* x, double* cols, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::im2col_1d_f64(x, cols, C, S[0], K[0], O[0],
                                stride[0], pad[0], dil[0]);
}
template <> void im2col_dispatch<2, float>(
    const float* x, float* cols, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::im2col_f32(x, cols, C, S[0], S[1], K[0], K[1], O[0], O[1],
                             stride[0], stride[1], pad[0], pad[1],
                             dil[0], dil[1]);
}
template <> void im2col_dispatch<2, double>(
    const double* x, double* cols, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::im2col_f64(x, cols, C, S[0], S[1], K[0], K[1], O[0], O[1],
                             stride[0], stride[1], pad[0], pad[1],
                             dil[0], dil[1]);
}
template <> void im2col_dispatch<3, float>(
    const float* x, float* cols, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::im2col_3d_f32(x, cols, C, S[0], S[1], S[2],
                                K[0], K[1], K[2], O[0], O[1], O[2],
                                stride[0], stride[1], stride[2],
                                pad[0], pad[1], pad[2],
                                dil[0], dil[1], dil[2]);
}
template <> void im2col_dispatch<3, double>(
    const double* x, double* cols, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::im2col_3d_f64(x, cols, C, S[0], S[1], S[2],
                                K[0], K[1], K[2], O[0], O[1], O[2],
                                stride[0], stride[1], stride[2],
                                pad[0], pad[1], pad[2],
                                dil[0], dil[1], dil[2]);
}

template <int N, typename T>
void col2im_dispatch(const T* cols, T* dx, int C,
                     const int* S, const int* K, const int* O,
                     const int* stride, const int* pad, const int* dil);

template <> void col2im_dispatch<1, float>(
    const float* cols, float* dx, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::col2im_1d_f32(cols, dx, C, S[0], K[0], O[0],
                                 stride[0], pad[0], dil[0]);
}
template <> void col2im_dispatch<1, double>(
    const double* cols, double* dx, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::col2im_1d_f64(cols, dx, C, S[0], K[0], O[0],
                                 stride[0], pad[0], dil[0]);
}
template <> void col2im_dispatch<2, float>(
    const float* cols, float* dx, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::col2im_f32(cols, dx, C, S[0], S[1], K[0], K[1], O[0], O[1],
                             stride[0], stride[1], pad[0], pad[1],
                             dil[0], dil[1]);
}
template <> void col2im_dispatch<2, double>(
    const double* cols, double* dx, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::col2im_f64(cols, dx, C, S[0], S[1], K[0], K[1], O[0], O[1],
                             stride[0], stride[1], pad[0], pad[1],
                             dil[0], dil[1]);
}
template <> void col2im_dispatch<3, float>(
    const float* cols, float* dx, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::col2im_3d_f32(cols, dx, C, S[0], S[1], S[2],
                                K[0], K[1], K[2], O[0], O[1], O[2],
                                stride[0], stride[1], stride[2],
                                pad[0], pad[1], pad[2],
                                dil[0], dil[1], dil[2]);
}
template <> void col2im_dispatch<3, double>(
    const double* cols, double* dx, int C,
    const int* S, const int* K, const int* O,
    const int* stride, const int* pad, const int* dil) {
    backend::cpu::col2im_3d_f64(cols, dx, C, S[0], S[1], S[2],
                                K[0], K[1], K[2], O[0], O[1], O[2],
                                stride[0], stride[1], stride[2],
                                pad[0], pad[1], pad[2],
                                dil[0], dil[1], dil[2]);
}

template <int N>
std::vector<int> nchw_to_nhwc_perm() {
    std::vector<int> p;
    p.reserve(N + 2);
    p.push_back(0);
    for (int i = 0; i < N; ++i) p.push_back(2 + i);
    p.push_back(1);
    return p;
}

template <int N>
std::vector<int> nhwc_to_nchw_perm() {
    std::vector<int> p;
    p.reserve(N + 2);
    p.push_back(0);
    p.push_back(N + 1);
    for (int i = 0; i < N; ++i) p.push_back(1 + i);
    return p;
}

template <int N>
std::vector<int> w_nchw_to_nhwc_perm() { return nchw_to_nhwc_perm<N>(); }

template <int N>
std::vector<int> w_to_transpose_perm() {
    std::vector<int> p;
    p.reserve(N + 2);
    p.push_back(1);
    for (int i = 0; i < N; ++i) p.push_back(2 + i);
    p.push_back(0);
    return p;
}

}  // namespace

template <int N>
TensorImplPtr ConvNdBackward<N>::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& W,
                                         const TensorImplPtr& b,
                                         const int (&stride)[N],
                                         const int (&pad)[N],
                                         const int (&dilation)[N],
                                         int groups) {
    if (!x || !W || !b) throw LucidError("conv: null input");
    if (x->dtype_ != W->dtype_ || x->dtype_ != b->dtype_)
        throw DtypeMismatch(std::string(dtype_name(x->dtype_)),
                            std::string(dtype_name(W->dtype_)), "conv");
    if (x->device_ != W->device_ || x->device_ != b->device_)
        throw DeviceMismatch(std::string(device_name(x->device_)),
                             std::string(device_name(W->device_)), "conv");
    if (x->device_ == Device::CPU &&
        (!x->is_contiguous() || !W->is_contiguous() || !b->is_contiguous()))
        throw NotImplementedError(
            "conv: non-contiguous input not supported (call .contiguous() first)");
    if (static_cast<int>(x->shape_.size()) != N + 2)
        throw ShapeMismatch(x->shape_, Shape{}, "conv: x rank mismatch");
    if (static_cast<int>(W->shape_.size()) != N + 2)
        throw ShapeMismatch(W->shape_, Shape{}, "conv: W rank mismatch");
    if (b->shape_.size() != 1)
        throw ShapeMismatch(b->shape_, Shape{}, "conv: b must be 1-D (C_out,)");
    if (groups < 1)
        throw LucidError("conv: groups must be >= 1");

    const int B    = static_cast<int>(x->shape_[0]);
    const int Cin  = static_cast<int>(x->shape_[1]);
    const int Cout = static_cast<int>(W->shape_[0]);
    const int Cw   = static_cast<int>(W->shape_[1]);

    if (Cin % groups != 0)
        throw ShapeMismatch(x->shape_, W->shape_,
                             "conv: C_in must be divisible by groups");
    if (Cout % groups != 0)
        throw ShapeMismatch(W->shape_, x->shape_,
                             "conv: C_out must be divisible by groups");
    const int Cin_g  = Cin  / groups;
    const int Cout_g = Cout / groups;
    if (Cw != Cin_g)
        throw ShapeMismatch(W->shape_, x->shape_,
                             "conv: W.shape[1] must equal C_in / groups");
    if (b->shape_[0] != Cout)
        throw ShapeMismatch(b->shape_, W->shape_, "conv: bias C_out mismatch");

    int S[N];
    int K[N];
    int O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape_[2 + i]);
        K[i] = static_cast<int>(W->shape_[2 + i]);
        if (dilation[i] < 1)
            throw LucidError("conv: dilation must be >= 1");
        O[i] = compute_out(S[i], K[i], stride[i], pad[i], dilation[i]);
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape_, W->shape_,
                                "conv: output shape non-positive");
    }

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(Cout));
    for (int i = 0; i < N; ++i) out_shape.push_back(static_cast<std::int64_t>(O[i]));

    int O_total = 1; for (int i = 0; i < N; ++i) O_total *= O[i];
    int K_total = 1; for (int i = 0; i < N; ++i) K_total *= K[i];
    int S_total = 1; for (int i = 0; i < N; ++i) S_total *= S[i];

    OpScope scope{ConvNdBackward<N>::schema_v1.name, x->device_, x->dtype_, out_shape};
    scope.set_flops(static_cast<std::int64_t>(2) * B * Cout * O_total * Cin_g * K_total);

    Storage out_storage;

    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto& gW = std::get<GpuStorage>(W->storage_);
        const auto& gb = std::get<GpuStorage>(b->storage_);
        if (!gx.arr || !gW.arr || !gb.arr) {
            throw LucidError("conv: null GPU input");
        }
        auto x_nhwc = ::mlx::core::transpose(*gx.arr, nchw_to_nhwc_perm<N>());
        auto W_nhwc = ::mlx::core::transpose(*gW.arr, w_nchw_to_nhwc_perm<N>());
        std::vector<int> sv(stride, stride + N);
        std::vector<int> pv(pad, pad + N);
        std::vector<int> dv(dilation, dilation + N);
        std::vector<int> idv(N, 1);
        auto y_nhwc = ::mlx::core::conv_general(
            x_nhwc, W_nhwc, sv, pv, pv, dv, idv,
            /*groups=*/groups,
            /*flip=*/false);
        ::mlx::core::Shape b_brd(N + 2, 1);
        b_brd[N + 1] = static_cast<::mlx::core::ShapeElem>(Cout);
        auto b_view = ::mlx::core::reshape(*gb.arr, b_brd);
        y_nhwc = ::mlx::core::add(y_nhwc, b_view);
        auto y = ::mlx::core::contiguous(::mlx::core::transpose(
            y_nhwc, nhwc_to_nchw_perm<N>()));
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y), x->dtype_)};
    } else {
        auto out_cpu = allocate_size(static_cast<std::size_t>(B) * Cout * O_total,
                                     x->dtype_);
        auto cols_cpu = allocate_size(
            static_cast<std::size_t>(Cin_g) * K_total * O_total, x->dtype_);

        const auto& x_cpu = std::get<CpuStorage>(x->storage_);
        const auto& W_cpu = std::get<CpuStorage>(W->storage_);
        const auto& b_cpu = std::get<CpuStorage>(b->storage_);
        const int K_flat = Cin_g * K_total;
        const int M_out  = O_total;
        const int W_per_group = Cout_g * K_flat;

        for (int bi = 0; bi < B; ++bi) {
            for (int g = 0; g < groups; ++g) {
                switch (x->dtype_) {
                    case Dtype::F32: {
                        auto* xp = reinterpret_cast<const float*>(x_cpu.ptr.get())
                                   + (static_cast<std::size_t>(bi) * Cin
                                      + static_cast<std::size_t>(g) * Cin_g) * S_total;
                        auto* cp = reinterpret_cast<float*>(cols_cpu.ptr.get());
                        auto* wp = reinterpret_cast<const float*>(W_cpu.ptr.get())
                                   + static_cast<std::size_t>(g) * W_per_group;
                        auto* yp = reinterpret_cast<float*>(out_cpu.ptr.get())
                                   + (static_cast<std::size_t>(bi) * Cout
                                      + static_cast<std::size_t>(g) * Cout_g) * O_total;
                        im2col_dispatch<N, float>(xp, cp, Cin_g, S, K, O,
                                                   stride, pad, dilation);
                        backend::cpu::sgemm(false, false, Cout_g, M_out, K_flat, 1.0f,
                                            wp, K_flat, cp, M_out, 0.0f, yp, M_out);
                        break;
                    }
                    case Dtype::F64: {
                        auto* xp = reinterpret_cast<const double*>(x_cpu.ptr.get())
                                   + (static_cast<std::size_t>(bi) * Cin
                                      + static_cast<std::size_t>(g) * Cin_g) * S_total;
                        auto* cp = reinterpret_cast<double*>(cols_cpu.ptr.get());
                        auto* wp = reinterpret_cast<const double*>(W_cpu.ptr.get())
                                   + static_cast<std::size_t>(g) * W_per_group;
                        auto* yp = reinterpret_cast<double*>(out_cpu.ptr.get())
                                   + (static_cast<std::size_t>(bi) * Cout
                                      + static_cast<std::size_t>(g) * Cout_g) * O_total;
                        im2col_dispatch<N, double>(xp, cp, Cin_g, S, K, O,
                                                    stride, pad, dilation);
                        backend::cpu::dgemm(false, false, Cout_g, M_out, K_flat, 1.0,
                                            wp, K_flat, cp, M_out, 0.0, yp, M_out);
                        break;
                    }
                    default:
                        throw NotImplementedError("conv: dtype not supported (F32/F64)");
                }
            }
            switch (x->dtype_) {
                case Dtype::F32: {
                    auto* yp = reinterpret_cast<float*>(out_cpu.ptr.get())
                               + static_cast<std::size_t>(bi) * Cout * O_total;
                    add_bias_chw<float>(
                        yp, reinterpret_cast<const float*>(b_cpu.ptr.get()),
                        Cout, O_total);
                    break;
                }
                case Dtype::F64: {
                    auto* yp = reinterpret_cast<double*>(out_cpu.ptr.get())
                               + static_cast<std::size_t>(bi) * Cout * O_total;
                    add_bias_chw<double>(
                        yp, reinterpret_cast<const double*>(b_cpu.ptr.get()),
                        Cout, O_total);
                    break;
                }
                default: break;
            }
        }
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            std::move(out_shape), x->dtype_,
                                            x->device_, false);

    if (!GradMode::is_enabled() ||
        !(x->requires_grad_ || W->requires_grad_ || b->requires_grad_)) {
        return out;
    }

    auto x_edge = detail::ensure_grad_fn(x);
    auto W_edge = detail::ensure_grad_fn(W);
    auto b_edge = detail::ensure_grad_fn(b);

    auto bwd = std::make_shared<ConvNdBackward<N>>();
    bwd->input_shapes_  = {x->shape_, W->shape_, b->shape_};
    bwd->out_shape_     = out->shape_;
    bwd->dtype_         = x->dtype_;
    bwd->device_        = x->device_;
    bwd->input_tensors_ = {x, W, b};
    bwd->saved_inputs_  = {x->storage_, W->storage_, b->storage_};
    for (int i = 0; i < N; ++i) {
        bwd->stride_[i]   = stride[i];
        bwd->pad_[i]      = pad[i];
        bwd->dilation_[i] = dilation[i];
    }
    bwd->groups_ = groups;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(W_edge, 0), Edge(b_edge, 0)});
    bwd->set_saved_versions({x->version_, W->version_, b->version_});

    out->grad_fn_       = std::move(bwd);
    out->is_leaf_       = false;
    out->requires_grad_ = true;
    return out;
}

template <int N>
std::vector<Storage> ConvNdBackward<N>::apply(Storage grad_out) {
    const int B    = static_cast<int>(this->input_shapes_[0][0]);
    const int Cin  = static_cast<int>(this->input_shapes_[0][1]);
    const int Cout = static_cast<int>(this->input_shapes_[1][0]);
    const int G    = this->groups_;
    const int Cin_g  = Cin  / G;
    const int Cout_g = Cout / G;
    int S[N], K[N], O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        K[i] = static_cast<int>(this->input_shapes_[1][2 + i]);
        O[i] = static_cast<int>(this->out_shape_[2 + i]);
    }
    int O_total = 1; for (int i = 0; i < N; ++i) O_total *= O[i];
    int K_total = 1; for (int i = 0; i < N; ++i) K_total *= K[i];
    int S_total = 1; for (int i = 0; i < N; ++i) S_total *= S[i];

    if (this->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(this->saved_inputs_[0]);
        const auto& gW = std::get<GpuStorage>(this->saved_inputs_[1]);
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gx.arr || !gW.arr || !gG.arr) {
            throw LucidError("conv backward: null GPU array");
        }
        std::vector<int> db_axes;
        db_axes.reserve(N + 1);
        db_axes.push_back(0);
        for (int i = 0; i < N; ++i) db_axes.push_back(2 + i);
        auto db = ::mlx::core::sum(*gG.arr, db_axes, /*keepdims=*/false);

        std::vector<int> sv(this->stride_, this->stride_ + N);
        std::vector<int> pv(this->pad_, this->pad_ + N);
        std::vector<int> dv(this->dilation_, this->dilation_ + N);

        // Transposed-conv pad_hi for dx: derived from forward
        //   O = (I + 2P - K')/S + 1, K' = (K-1)*D + 1
        // so the TC inverse needs pad_lo + pad_hi = I + (K-1)*D - 1 - (O-1)*S.
        // Setting pad_lo = forward pad gives the asymmetric pad_hi below.
        std::vector<int> opad(N);
        for (int i = 0; i < N; ++i) {
            opad[i] = S[i] + (K[i] - 1) * dv[i] - 1
                      - (O[i] - 1) * sv[i] - pv[i];
        }
        auto grad_nhwc = ::mlx::core::transpose(*gG.arr, nchw_to_nhwc_perm<N>());
        auto W_t_nhwc = ::mlx::core::transpose(*gW.arr, w_to_transpose_perm<N>());
        std::vector<int> ones_n(N, 1);
        // kernel_dilation must be the forward dilation `dv` so the dilated
        // kernel pattern is reproduced; input_dilation = forward stride `sv`.
        auto dx_nhwc = ::mlx::core::conv_general(
            grad_nhwc, W_t_nhwc, ones_n, pv, opad, dv, sv,
            /*groups=*/G, /*flip=*/true);
        auto dx = ::mlx::core::contiguous(::mlx::core::transpose(
            dx_nhwc, nhwc_to_nchw_perm<N>()));

        std::vector<int> x_perm_axes;
        x_perm_axes.push_back(1);
        for (int i = 0; i < N; ++i) x_perm_axes.push_back(2 + i);
        x_perm_axes.push_back(0);
        auto x_perm = ::mlx::core::transpose(*gx.arr, x_perm_axes);
        auto g_perm = ::mlx::core::transpose(*gG.arr, x_perm_axes);
        std::vector<int> conv_stride(N, 1);
        std::vector<int> conv_pad = pv;
        std::vector<int> conv_kdil = sv;
        std::vector<int> conv_idil = dv;
        auto dW_perm = ::mlx::core::conv_general(
            x_perm, g_perm, conv_stride, conv_pad, conv_pad,
            conv_kdil, conv_idil, /*groups=*/G, /*flip=*/false);
        ::mlx::core::Shape crop_lo(N + 2, 0);
        ::mlx::core::Shape crop_hi;
        crop_hi.push_back(static_cast<::mlx::core::ShapeElem>(Cin_g));
        for (int i = 0; i < N; ++i)
            crop_hi.push_back(static_cast<::mlx::core::ShapeElem>(K[i]));
        crop_hi.push_back(static_cast<::mlx::core::ShapeElem>(Cout));
        dW_perm = ::mlx::core::slice(dW_perm, crop_lo, crop_hi);
        std::vector<int> dW_perm_back;
        dW_perm_back.push_back(N + 1);
        dW_perm_back.push_back(0);
        for (int i = 0; i < N; ++i) dW_perm_back.push_back(1 + i);
        auto dW = ::mlx::core::contiguous(::mlx::core::transpose(
            dW_perm, dW_perm_back));

        return {Storage{gpu::wrap_mlx_array(std::move(dx), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dW), this->dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(db), this->dtype_)}};
    }

    const int K_flat = Cin_g * K_total;
    const int M_out  = O_total;

    auto dx_cpu = allocate_size(static_cast<std::size_t>(B) * Cin * S_total, this->dtype_);
    auto dW_cpu = allocate_size(static_cast<std::size_t>(Cout) * K_flat, this->dtype_);
    auto db_cpu = allocate_size(static_cast<std::size_t>(Cout), this->dtype_);
    if (dx_cpu.nbytes) std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    if (dW_cpu.nbytes) std::memset(dW_cpu.ptr.get(), 0, dW_cpu.nbytes);
    if (db_cpu.nbytes) std::memset(db_cpu.ptr.get(), 0, db_cpu.nbytes);

    auto cols_cpu     = allocate_size(static_cast<std::size_t>(K_flat) * M_out, this->dtype_);
    auto col_grad_cpu = allocate_size(static_cast<std::size_t>(K_flat) * M_out, this->dtype_);

    const auto& x_cpu = std::get<CpuStorage>(this->saved_inputs_[0]);
    const auto& W_cpu = std::get<CpuStorage>(this->saved_inputs_[1]);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);
    const int W_per_group = Cout_g * K_flat;

    for (int bi = 0; bi < B; ++bi) {
        for (int g = 0; g < G; ++g) {
            switch (this->dtype_) {
                case Dtype::F32: {
                    auto* xp = reinterpret_cast<const float*>(x_cpu.ptr.get())
                               + (static_cast<std::size_t>(bi) * Cin
                                  + static_cast<std::size_t>(g) * Cin_g) * S_total;
                    auto* gp = reinterpret_cast<const float*>(g_cpu.ptr.get())
                               + (static_cast<std::size_t>(bi) * Cout
                                  + static_cast<std::size_t>(g) * Cout_g) * O_total;
                    auto* dxp = reinterpret_cast<float*>(dx_cpu.ptr.get())
                                + (static_cast<std::size_t>(bi) * Cin
                                   + static_cast<std::size_t>(g) * Cin_g) * S_total;
                    auto* wp = reinterpret_cast<const float*>(W_cpu.ptr.get())
                               + static_cast<std::size_t>(g) * W_per_group;
                    auto* dwp = reinterpret_cast<float*>(dW_cpu.ptr.get())
                                + static_cast<std::size_t>(g) * W_per_group;
                    auto* cp  = reinterpret_cast<float*>(cols_cpu.ptr.get());
                    auto* cgp = reinterpret_cast<float*>(col_grad_cpu.ptr.get());

                    im2col_dispatch<N, float>(xp, cp, Cin_g, S, K, O,
                                               this->stride_, this->pad_,
                                               this->dilation_);
                    backend::cpu::sgemm(false, true, Cout_g, K_flat, M_out, 1.0f,
                                        gp, M_out, cp, M_out, 1.0f,
                                        dwp, K_flat);
                    backend::cpu::sgemm(true, false, K_flat, M_out, Cout_g, 1.0f,
                                        wp, K_flat, gp, M_out, 0.0f, cgp, M_out);
                    col2im_dispatch<N, float>(cgp, dxp, Cin_g, S, K, O,
                                               this->stride_, this->pad_,
                                               this->dilation_);
                    {
                        auto* dbp = reinterpret_cast<float*>(db_cpu.ptr.get())
                                    + static_cast<std::size_t>(g) * Cout_g;
                        for (int co = 0; co < Cout_g; ++co) {
                            const float* row = gp + co * O_total;
                            float s = 0.f;
                            for (int j = 0; j < O_total; ++j) s += row[j];
                            dbp[co] += s;
                        }
                    }
                    break;
                }
                case Dtype::F64: {
                    auto* xp = reinterpret_cast<const double*>(x_cpu.ptr.get())
                               + (static_cast<std::size_t>(bi) * Cin
                                  + static_cast<std::size_t>(g) * Cin_g) * S_total;
                    auto* gp = reinterpret_cast<const double*>(g_cpu.ptr.get())
                               + (static_cast<std::size_t>(bi) * Cout
                                  + static_cast<std::size_t>(g) * Cout_g) * O_total;
                    auto* dxp = reinterpret_cast<double*>(dx_cpu.ptr.get())
                                + (static_cast<std::size_t>(bi) * Cin
                                   + static_cast<std::size_t>(g) * Cin_g) * S_total;
                    auto* wp = reinterpret_cast<const double*>(W_cpu.ptr.get())
                               + static_cast<std::size_t>(g) * W_per_group;
                    auto* dwp = reinterpret_cast<double*>(dW_cpu.ptr.get())
                                + static_cast<std::size_t>(g) * W_per_group;
                    auto* cp  = reinterpret_cast<double*>(cols_cpu.ptr.get());
                    auto* cgp = reinterpret_cast<double*>(col_grad_cpu.ptr.get());

                    im2col_dispatch<N, double>(xp, cp, Cin_g, S, K, O,
                                                this->stride_, this->pad_,
                                                this->dilation_);
                    backend::cpu::dgemm(false, true, Cout_g, K_flat, M_out, 1.0,
                                        gp, M_out, cp, M_out, 1.0,
                                        dwp, K_flat);
                    backend::cpu::dgemm(true, false, K_flat, M_out, Cout_g, 1.0,
                                        wp, K_flat, gp, M_out, 0.0, cgp, M_out);
                    col2im_dispatch<N, double>(cgp, dxp, Cin_g, S, K, O,
                                                this->stride_, this->pad_,
                                                this->dilation_);
                    {
                        auto* dbp = reinterpret_cast<double*>(db_cpu.ptr.get())
                                    + static_cast<std::size_t>(g) * Cout_g;
                        for (int co = 0; co < Cout_g; ++co) {
                            const double* row = gp + co * O_total;
                            double s = 0.0;
                            for (int j = 0; j < O_total; ++j) s += row[j];
                            dbp[co] += s;
                        }
                    }
                    break;
                }
                default:
                    throw NotImplementedError("conv backward: dtype not supported");
            }
        }
    }
    return {Storage{std::move(dx_cpu)}, Storage{std::move(dW_cpu)},
            Storage{std::move(db_cpu)}};
}

template class ConvNdBackward<1>;
template class ConvNdBackward<2>;
template class ConvNdBackward<3>;

TensorImplPtr conv1d_op(const TensorImplPtr& x, const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sl, int pl, int dl, int groups) {
    int stride[1]{sl};
    int pad[1]{pl};
    int dilation[1]{dl};
    return Conv1dBackward::forward(x, W, b, stride, pad, dilation, groups);
}
TensorImplPtr conv2d_op(const TensorImplPtr& x, const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sh, int sw, int ph, int pw,
                        int dh, int dw, int groups) {
    int stride[2]{sh, sw};
    int pad[2]{ph, pw};
    int dilation[2]{dh, dw};
    return Conv2dBackward::forward(x, W, b, stride, pad, dilation, groups);
}
TensorImplPtr conv3d_op(const TensorImplPtr& x, const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sd, int sh, int sw,
                        int pd, int ph, int pw,
                        int dd, int dh, int dw, int groups) {
    int stride[3]{sd, sh, sw};
    int pad[3]{pd, ph, pw};
    int dilation[3]{dd, dh, dw};
    return Conv3dBackward::forward(x, W, b, stride, pad, dilation, groups);
}

LUCID_REGISTER_OP(Conv1dBackward)
LUCID_REGISTER_OP(Conv2dBackward)
LUCID_REGISTER_OP(Conv3dBackward)

// ===================================================================
// Unfold (im2col exposed as a standalone op)
// ===================================================================

const OpSchema UnfoldBackward::schema_v1{"unfold", 1, AmpPolicy::KeepInput, true};

TensorImplPtr UnfoldBackward::forward(const TensorImplPtr& x,
                                       const std::vector<int>& kernel,
                                       const std::vector<int>& stride,
                                       const std::vector<int>& pad,
                                       const std::vector<int>& dilation) {
    if (!x) throw LucidError("unfold: null input");
    if (x->device_ == Device::CPU && !x->is_contiguous())
        throw NotImplementedError("unfold: non-contiguous input not supported");

    const int N = static_cast<int>(kernel.size());
    if (N < 1 || N > 3)
        throw LucidError("unfold: only 1D / 2D / 3D supported");
    if (static_cast<int>(stride.size()) != N
        || static_cast<int>(pad.size()) != N
        || static_cast<int>(dilation.size()) != N)
        throw LucidError("unfold: stride/pad/dilation length must match kernel");
    if (static_cast<int>(x->shape_.size()) != N + 2)
        throw ShapeMismatch(x->shape_, Shape{}, "unfold: x rank mismatch");

    const int B   = static_cast<int>(x->shape_[0]);
    const int C   = static_cast<int>(x->shape_[1]);
    std::vector<int> S(N), K = kernel, O(N);
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape_[2 + i]);
        const int eff = dilation[i] * (K[i] - 1) + 1;
        O[i] = (S[i] + 2 * pad[i] - eff) / stride[i] + 1;
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape_, Shape{}, "unfold: non-positive output dim");
    }
    int O_total = 1; for (int i = 0; i < N; ++i) O_total *= O[i];
    int K_total = 1; for (int i = 0; i < N; ++i) K_total *= K[i];
    int S_total = 1; for (int i = 0; i < N; ++i) S_total *= S[i];

    Shape out_shape{static_cast<std::int64_t>(B),
                     static_cast<std::int64_t>(C * K_total),
                     static_cast<std::int64_t>(O_total)};

    OpScope scope{schema_v1.name, x->device_, x->dtype_, out_shape};

    // Native MLX path for unfold: build N-D source-coordinate maps via
    // index arithmetic, mask out-of-bounds positions to zero, then take.
    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(x->dtype_);
        // Spatial-dim index arrays per axis: shape patterns
        //   k_d[d]: (1, 1, K_0, K_1, ..., O_0, O_1, ...) — 1 at non-d axes
        //   o_d[d]: similarly
        // Final composite index has shape (B, C, K_0..K_{N-1}, O_0..O_{N-1}).
        ::mlx::core::Shape composite_shape;
        composite_shape.push_back(B);
        composite_shape.push_back(C);
        for (int d = 0; d < N; ++d) composite_shape.push_back(K[d]);
        for (int d = 0; d < N; ++d) composite_shape.push_back(O[d]);

        auto i32 = ::mlx::core::int32;
        ::mlx::core::Shape b_shape(composite_shape.size(), 1);
        b_shape[0] = B;
        auto b_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, B, 1), i32), b_shape);

        ::mlx::core::Shape c_shape(composite_shape.size(), 1);
        c_shape[1] = C;
        auto c_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), i32), c_shape);

        // For each spatial dim d:
        //   in_d = stride[d] * o + dilation[d] * k - pad[d]
        // valid_d = (in_d >= 0) & (in_d < S[d])
        std::optional<::mlx::core::array> valid_opt;
        std::vector<::mlx::core::array> in_d_clipped;
        for (int d = 0; d < N; ++d) {
            auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, K[d], 1), i32);
            auto o_arr = ::mlx::core::astype(::mlx::core::arange(0, O[d], 1), i32);
            ::mlx::core::Shape k_shape(composite_shape.size(), 1);
            k_shape[2 + d] = K[d];
            ::mlx::core::Shape o_shape(composite_shape.size(), 1);
            o_shape[2 + N + d] = O[d];
            k_arr = ::mlx::core::reshape(k_arr, k_shape);
            o_arr = ::mlx::core::reshape(o_arr, o_shape);
            auto sd = ::mlx::core::array(stride[d], i32);
            auto dd = ::mlx::core::array(dilation[d], i32);
            auto pd = ::mlx::core::array(pad[d], i32);
            auto in_d = ::mlx::core::subtract(
                ::mlx::core::add(::mlx::core::multiply(sd, o_arr),
                                  ::mlx::core::multiply(dd, k_arr)),
                pd);
            auto zero_i = ::mlx::core::array(0, i32);
            auto cap_i = ::mlx::core::array(S[d] - 1, i32);
            auto v = ::mlx::core::logical_and(
                ::mlx::core::greater_equal(in_d, zero_i),
                ::mlx::core::less_equal(in_d, cap_i));
            valid_opt = valid_opt.has_value()
                ? ::mlx::core::logical_and(*valid_opt, v) : v;
            in_d_clipped.push_back(::mlx::core::clip(in_d,
                std::optional<::mlx::core::array>(zero_i),
                std::optional<::mlx::core::array>(cap_i)));
        }
        auto valid = *valid_opt;

        // flat = b * C*S_total + c * S_total + sum_d in_d_clipped * (prod of trailing S[k>d])
        auto flat = ::mlx::core::add(
            ::mlx::core::multiply(b_idx,
                ::mlx::core::array(C * S_total, i32)),
            ::mlx::core::multiply(c_idx,
                ::mlx::core::array(S_total, i32)));
        for (int d = 0; d < N; ++d) {
            int trailing = 1;
            for (int e = d + 1; e < N; ++e) trailing *= S[e];
            flat = ::mlx::core::add(flat,
                ::mlx::core::multiply(in_d_clipped[d],
                    ::mlx::core::array(trailing, i32)));
        }
        flat = ::mlx::core::broadcast_to(flat, composite_shape);

        auto x_flat = ::mlx::core::reshape(*gx.arr,
            ::mlx::core::Shape{B * C * S_total});
        auto sampled = ::mlx::core::take(x_flat, flat);
        auto valid_b = ::mlx::core::astype(
            ::mlx::core::broadcast_to(valid, composite_shape), mlx_dt);
        auto masked = ::mlx::core::multiply(sampled, valid_b);

        // Reshape (B, C, K_0..K_{N-1}, O_0..O_{N-1}) → (B, C*K_total, O_total).
        auto reshaped = ::mlx::core::reshape(masked,
            ::mlx::core::Shape{B, C * K_total, O_total});
        auto out_storage_gpu = Storage{gpu::wrap_mlx_array(std::move(reshaped),
                                                            x->dtype_)};
        auto out = std::make_shared<TensorImpl>(std::move(out_storage_gpu),
                                                 out_shape, x->dtype_,
                                                 x->device_, false);
        if (!GradMode::is_enabled() || !x->requires_grad_) return out;
        auto x_edge = detail::ensure_grad_fn(x);
        auto bwd = std::make_shared<UnfoldBackward>();
        bwd->input_shapes_ = {x->shape_};
        bwd->out_shape_    = out->shape_;
        bwd->dtype_        = x->dtype_;
        bwd->device_       = x->device_;
        bwd->input_tensors_ = {x};
        bwd->kernel_   = kernel;
        bwd->stride_   = stride;
        bwd->pad_      = pad;
        bwd->dilation_ = dilation;
        bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
        bwd->set_saved_versions({x->version_});
        out->grad_fn_       = std::move(bwd);
        out->is_leaf_       = false;
        out->requires_grad_ = true;
        return out;
    }

    // CPU path — Apple Accelerate im2col helpers.
    auto out_cpu = allocate_size(static_cast<std::size_t>(B) * C * K_total * O_total,
                                  x->dtype_);
    const auto& x_cpu = std::get<CpuStorage>(x->storage_);

    for (int bi = 0; bi < B; ++bi) {
        switch (x->dtype_) {
            case Dtype::F32: {
                auto* xp = reinterpret_cast<const float*>(x_cpu.ptr.get())
                           + static_cast<std::size_t>(bi) * C * S_total;
                auto* yp = reinterpret_cast<float*>(out_cpu.ptr.get())
                           + static_cast<std::size_t>(bi) * C * K_total * O_total;
                if (N == 1)
                    backend::cpu::im2col_1d_f32(xp, yp, C, S[0], K[0], O[0],
                                                  stride[0], pad[0], dilation[0]);
                else if (N == 2)
                    backend::cpu::im2col_f32(xp, yp, C, S[0], S[1], K[0], K[1],
                                               O[0], O[1], stride[0], stride[1],
                                               pad[0], pad[1], dilation[0], dilation[1]);
                else
                    backend::cpu::im2col_3d_f32(xp, yp, C, S[0], S[1], S[2],
                                                  K[0], K[1], K[2], O[0], O[1], O[2],
                                                  stride[0], stride[1], stride[2],
                                                  pad[0], pad[1], pad[2],
                                                  dilation[0], dilation[1], dilation[2]);
                break;
            }
            case Dtype::F64: {
                auto* xp = reinterpret_cast<const double*>(x_cpu.ptr.get())
                           + static_cast<std::size_t>(bi) * C * S_total;
                auto* yp = reinterpret_cast<double*>(out_cpu.ptr.get())
                           + static_cast<std::size_t>(bi) * C * K_total * O_total;
                if (N == 1)
                    backend::cpu::im2col_1d_f64(xp, yp, C, S[0], K[0], O[0],
                                                  stride[0], pad[0], dilation[0]);
                else if (N == 2)
                    backend::cpu::im2col_f64(xp, yp, C, S[0], S[1], K[0], K[1],
                                               O[0], O[1], stride[0], stride[1],
                                               pad[0], pad[1], dilation[0], dilation[1]);
                else
                    backend::cpu::im2col_3d_f64(xp, yp, C, S[0], S[1], S[2],
                                                  K[0], K[1], K[2], O[0], O[1], O[2],
                                                  stride[0], stride[1], stride[2],
                                                  pad[0], pad[1], pad[2],
                                                  dilation[0], dilation[1], dilation[2]);
                break;
            }
            default:
                throw NotImplementedError("unfold: dtype not supported");
        }
    }

    Storage out_storage = Storage{std::move(out_cpu)};
    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                             std::move(out_shape), x->dtype_,
                                             x->device_, false);

    if (!GradMode::is_enabled() || !x->requires_grad_) return out;

    auto x_edge = detail::ensure_grad_fn(x);
    auto bwd = std::make_shared<UnfoldBackward>();
    bwd->input_shapes_  = {x->shape_};
    bwd->out_shape_     = out->shape_;
    bwd->dtype_         = x->dtype_;
    bwd->device_        = x->device_;
    bwd->input_tensors_ = {x};
    bwd->kernel_   = kernel;
    bwd->stride_   = stride;
    bwd->pad_      = pad;
    bwd->dilation_ = dilation;
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions({x->version_});
    out->grad_fn_       = std::move(bwd);
    out->is_leaf_       = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> UnfoldBackward::apply(Storage grad_out) {
    const int N = static_cast<int>(kernel_.size());
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int C = static_cast<int>(this->input_shapes_[0][1]);
    std::vector<int> S(N), K = kernel_, O(N);
    int S_total = 1, K_total = 1, O_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        const int eff = dilation_[i] * (K[i] - 1) + 1;
        O[i] = (S[i] + 2 * pad_[i] - eff) / stride_[i] + 1;
        S_total *= S[i];
        K_total *= K[i];
        O_total *= O[i];
    }
    if (this->device_ == Device::GPU) {
        // Native MLX path — scatter_add into dx, mirroring the unfold forward
        // index arithmetic. Build the same N-D index arrays and add grad_out
        // values into the corresponding positions of a zero base.
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(this->dtype_);
        auto i32 = ::mlx::core::int32;

        ::mlx::core::Shape composite_shape;
        composite_shape.push_back(B);
        composite_shape.push_back(C);
        for (int d = 0; d < N; ++d) composite_shape.push_back(K[d]);
        for (int d = 0; d < N; ++d) composite_shape.push_back(O[d]);

        // batch / channel index arrays
        ::mlx::core::Shape b_shape(composite_shape.size(), 1);
        b_shape[0] = B;
        auto b_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, B, 1), i32), b_shape);
        ::mlx::core::Shape c_shape(composite_shape.size(), 1);
        c_shape[1] = C;
        auto c_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), i32), c_shape);

        std::optional<::mlx::core::array> valid_opt;
        std::vector<::mlx::core::array> in_d_clipped;
        for (int d = 0; d < N; ++d) {
            ::mlx::core::Shape k_shape(composite_shape.size(), 1);
            k_shape[2 + d] = K[d];
            ::mlx::core::Shape o_shape(composite_shape.size(), 1);
            o_shape[2 + N + d] = O[d];
            auto k_arr = ::mlx::core::reshape(
                ::mlx::core::astype(::mlx::core::arange(0, K[d], 1), i32), k_shape);
            auto o_arr = ::mlx::core::reshape(
                ::mlx::core::astype(::mlx::core::arange(0, O[d], 1), i32), o_shape);
            auto in_d = ::mlx::core::subtract(
                ::mlx::core::add(
                    ::mlx::core::multiply(::mlx::core::array(stride_[d], i32), o_arr),
                    ::mlx::core::multiply(::mlx::core::array(dilation_[d], i32), k_arr)),
                ::mlx::core::array(pad_[d], i32));
            auto zero_i = ::mlx::core::array(0, i32);
            auto cap_i = ::mlx::core::array(S[d] - 1, i32);
            auto v = ::mlx::core::logical_and(
                ::mlx::core::greater_equal(in_d, zero_i),
                ::mlx::core::less_equal(in_d, cap_i));
            valid_opt = valid_opt.has_value()
                ? ::mlx::core::logical_and(*valid_opt, v) : v;
            in_d_clipped.push_back(::mlx::core::clip(in_d,
                std::optional<::mlx::core::array>(zero_i),
                std::optional<::mlx::core::array>(cap_i)));
        }
        auto valid = *valid_opt;

        // Broadcast all index arrays to composite_shape and zero out
        // contributions where valid==false (by multiplying grad with valid mask).
        std::vector<::mlx::core::array> idxs;
        idxs.push_back(::mlx::core::broadcast_to(b_idx, composite_shape));
        idxs.push_back(::mlx::core::broadcast_to(c_idx, composite_shape));
        for (int d = 0; d < N; ++d)
            idxs.push_back(::mlx::core::broadcast_to(in_d_clipped[d],
                                                       composite_shape));

        // Reshape grad_out (B, C*K_total, O_total) → composite_shape, mask invalids.
        auto grad_comp = ::mlx::core::reshape(*gg.arr, composite_shape);
        auto valid_b = ::mlx::core::astype(
            ::mlx::core::broadcast_to(valid, composite_shape), mlx_dt);
        auto grad_masked = ::mlx::core::multiply(grad_comp, valid_b);

        // updates.shape = composite_shape + (1,)*input_ndim because we cover
        // all input axes (input_ndim = 2 + N: B, C, S_0..S_{N-1}).
        const int input_ndim = 2 + N;
        ::mlx::core::Shape upd_shape = composite_shape;
        for (int i = 0; i < input_ndim; ++i) upd_shape.push_back(1);
        auto updates = ::mlx::core::reshape(grad_masked, upd_shape);

        ::mlx::core::Shape base_shape;
        base_shape.push_back(B);
        base_shape.push_back(C);
        for (int d = 0; d < N; ++d) base_shape.push_back(S[d]);
        auto base = ::mlx::core::zeros(base_shape, mlx_dt);
        std::vector<int> axes_v;
        for (int i = 0; i < input_ndim; ++i) axes_v.push_back(i);
        auto out = ::mlx::core::scatter_add(base, idxs, updates, axes_v);
        return {Storage{gpu::wrap_mlx_array(std::move(out), this->dtype_)}};
    }

    auto dx_cpu = allocate_size(static_cast<std::size_t>(B) * C * S_total, this->dtype_);
    if (dx_cpu.nbytes) std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);

    for (int bi = 0; bi < B; ++bi) {
        switch (this->dtype_) {
            case Dtype::F32: {
                auto* gp = reinterpret_cast<const float*>(g_cpu.ptr.get())
                           + static_cast<std::size_t>(bi) * C * K_total * O_total;
                auto* dxp = reinterpret_cast<float*>(dx_cpu.ptr.get())
                            + static_cast<std::size_t>(bi) * C * S_total;
                if (N == 1)
                    backend::cpu::col2im_1d_f32(gp, dxp, C, S[0], K[0], O[0],
                                                 stride_[0], pad_[0], dilation_[0]);
                else if (N == 2)
                    backend::cpu::col2im_f32(gp, dxp, C, S[0], S[1], K[0], K[1],
                                               O[0], O[1], stride_[0], stride_[1],
                                               pad_[0], pad_[1],
                                               dilation_[0], dilation_[1]);
                else
                    backend::cpu::col2im_3d_f32(gp, dxp, C, S[0], S[1], S[2],
                                                 K[0], K[1], K[2], O[0], O[1], O[2],
                                                 stride_[0], stride_[1], stride_[2],
                                                 pad_[0], pad_[1], pad_[2],
                                                 dilation_[0], dilation_[1], dilation_[2]);
                break;
            }
            case Dtype::F64: {
                auto* gp = reinterpret_cast<const double*>(g_cpu.ptr.get())
                           + static_cast<std::size_t>(bi) * C * K_total * O_total;
                auto* dxp = reinterpret_cast<double*>(dx_cpu.ptr.get())
                            + static_cast<std::size_t>(bi) * C * S_total;
                if (N == 1)
                    backend::cpu::col2im_1d_f64(gp, dxp, C, S[0], K[0], O[0],
                                                 stride_[0], pad_[0], dilation_[0]);
                else if (N == 2)
                    backend::cpu::col2im_f64(gp, dxp, C, S[0], S[1], K[0], K[1],
                                               O[0], O[1], stride_[0], stride_[1],
                                               pad_[0], pad_[1],
                                               dilation_[0], dilation_[1]);
                else
                    backend::cpu::col2im_3d_f64(gp, dxp, C, S[0], S[1], S[2],
                                                 K[0], K[1], K[2], O[0], O[1], O[2],
                                                 stride_[0], stride_[1], stride_[2],
                                                 pad_[0], pad_[1], pad_[2],
                                                 dilation_[0], dilation_[1], dilation_[2]);
                break;
            }
            default:
                throw NotImplementedError("unfold backward: dtype not supported");
        }
    }
    return {Storage{std::move(dx_cpu)}};
}

TensorImplPtr unfold_op(const TensorImplPtr& x,
                         const std::vector<int>& kernel,
                         const std::vector<int>& stride,
                         const std::vector<int>& pad,
                         const std::vector<int>& dilation) {
    return UnfoldBackward::forward(x, kernel, stride, pad, dilation);
}

LUCID_REGISTER_OP(UnfoldBackward)

}  // namespace lucid
