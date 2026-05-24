// lucid/_C/nn/ConvNd.cpp
//
// N-dimensional convolution and Unfold implementation.
//
// Forward shape formula for each spatial axis i:
//   O[i] = (S[i] + 2*pad[i] - dilation[i]*(K[i]-1) - 1) / stride[i] + 1
//
// FLOP estimate (multiply-add pairs):
//   2 * B * C_out * prod(O) * C_in_per_group * prod(K)
//
// The forward delegates to IBackend::conv_nd_forward (im2col + GEMM on CPU,
// native MLX conv on GPU).  The backward delegates to
// IBackend::conv_nd_backward, which returns [dx, dW, db].
//
// Unfold (im2col) delegates to IBackend::unfold_forward; its backward
// (fold / col2im) delegates to IBackend::unfold_backward.

#include "ConvNd.h"

#include <vector>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/BinaryKernel.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/ufunc/Astype.h"

namespace lucid {

template <>
const OpSchema Conv1dBackward::schema_v1{"conv1d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema Conv2dBackward::schema_v1{"conv2d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema Conv3dBackward::schema_v1{"conv3d", 1, AmpPolicy::Promote, true};

namespace {

// Compute the output size for one spatial dimension.
// S – input size, K – kernel size, stride, pad, dilation are scalars.
inline int compute_out(int S, int K, int stride, int pad, int dilation) {
    const int eff = dilation * (K - 1) + 1;  // Effective (dilated) kernel size.
    return (S + 2 * pad - eff) / stride + 1;
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
    if (!x || !W || !b)
        ErrorBuilder("conv").fail("null input");
    if (x->device() != W->device() || x->device() != b->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(W->device())), "conv");
    // x must be (B, C_in, S_0...) and W must be (C_out, C_in/g, K_0...).
    if (static_cast<int>(x->shape().size()) != N + 2)
        throw ShapeMismatch(x->shape(), Shape{}, "conv: x rank mismatch");
    if (static_cast<int>(W->shape().size()) != N + 2)
        throw ShapeMismatch(W->shape(), Shape{}, "conv: W rank mismatch");
    if (b->shape().size() != 1)
        throw ShapeMismatch(b->shape(), Shape{}, "conv: b must be 1-D (C_out,)");
    if (groups < 1)
        ErrorBuilder("conv").fail("groups must be >= 1");

    // 3.3 AMP plumbing: schema_v1.amp_policy == Promote.  Under an
    // ``AutocastGuard(F16)`` scope, SchemaGuard resolves ``eff_dt`` to
    // the autocast target (F32 on CPU per the Accelerate-no-F16 carve-out,
    // F16 on GPU where ``mlx::core::conv_general`` accepts native F16
    // and Apple Silicon GPUs run the conv at half throughput → speedup).
    // Without this plumbing, ``with amp.autocast(F16):`` had no effect on
    // Conv2d — F32 was used regardless.  Must run before the
    // dtype-match check: under autocast, the surrounding Conv has cast
    // x to F16 while W / b on the Parameter slots are still F32 — the
    // pre-AMP strict equality check fired before SchemaGuard could
    // unify them.
    //
    // Use ``astype_op`` (not ``maybe_cast_for_kernel``) so the cast
    // tensors carry an ``AstypeBackward`` grad_fn pointing back to the
    // original Parameter / activation.  Without this the cast tensors
    // would have requires_grad=false → ``NaryKernel::wire_autograd`` would
    // short-circuit with any_grad=false → the entire conv backward chain
    // would be dropped under AMP.  ``astype_op`` returns the input
    // unchanged when dtypes already match, so the F32 baseline incurs
    // zero overhead.
    SchemaGuard sg{ConvNdBackward<N>::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr W_eff = astype_op(W, eff_dt);
    const TensorImplPtr b_eff = astype_op(b, eff_dt);
    if (x_eff->dtype() != W_eff->dtype() || x_eff->dtype() != b_eff->dtype())
        throw DtypeMismatch(std::string(dtype_name(x_eff->dtype())),
                            std::string(dtype_name(W_eff->dtype())), "conv");

    const int B = static_cast<int>(x_eff->shape()[0]);
    const int Cin = static_cast<int>(x_eff->shape()[1]);
    const int Cout = static_cast<int>(W_eff->shape()[0]);
    const int Cw = static_cast<int>(W_eff->shape()[1]);

    if (Cin % groups != 0)
        throw ShapeMismatch(x_eff->shape(), W_eff->shape(),
                            "conv: C_in must be divisible by groups");
    if (Cout % groups != 0)
        throw ShapeMismatch(W_eff->shape(), x_eff->shape(),
                            "conv: C_out must be divisible by groups");
    const int Cin_g = Cin / groups;
    const int Cout_g = Cout / groups;
    if (Cw != Cin_g)
        throw ShapeMismatch(W_eff->shape(), x_eff->shape(),
                            "conv: W.shape[1] must equal C_in / groups");
    if (b_eff->shape()[0] != Cout)
        throw ShapeMismatch(b_eff->shape(), W_eff->shape(), "conv: bias C_out mismatch");

    int S[N];
    int K[N];
    int O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        K[i] = static_cast<int>(W_eff->shape()[2 + i]);
        if (dilation[i] < 1)
            ErrorBuilder("conv").fail("dilation must be >= 1");
        O[i] = compute_out(S[i], K[i], stride[i], pad[i], dilation[i]);
        if (O[i] <= 0)
            throw ShapeMismatch(x_eff->shape(), W_eff->shape(),
                                "conv: output shape non-positive");
    }

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(Cout));
    for (int i = 0; i < N; ++i)
        out_shape.push_back(static_cast<std::int64_t>(O[i]));

    int O_total = 1;
    for (int i = 0; i < N; ++i)
        O_total *= O[i];
    int K_total = 1;
    for (int i = 0; i < N; ++i)
        K_total *= K[i];

    OpScopeFull scope{ConvNdBackward<N>::schema_v1.name, x_eff->device(), eff_dt, out_shape};
    // 2 ops (mul+add) per element of the output per input-channel-kernel position.
    scope.set_flops(static_cast<std::int64_t>(2) * B * Cout * O_total * Cin_g * K_total);
    // 3.5 Phase 1.2: report conv params for the compile-path emitter.
    {
        std::vector<std::int64_t> Sv(stride, stride + N);
        std::vector<std::int64_t> Pv(pad, pad + N);
        std::vector<std::int64_t> Dv(dilation, dilation + N);
        scope.set_attr("stride", std::move(Sv));
        scope.set_attr("padding", std::move(Pv));
        scope.set_attr("dilation", std::move(Dv));
        scope.set_attr("groups", static_cast<std::int64_t>(groups));
    }

    backend::IBackend::ConvNdOpts opts{};
    opts.N = N;
    opts.groups = groups;
    for (int i = 0; i < N; ++i) {
        opts.stride[i] = stride[i];
        opts.pad[i] = pad[i];
        opts.dilation[i] = dilation[i];
    }
    auto& be = backend::Dispatcher::for_device(x_eff->device());
    Storage out_storage =
        be.conv_nd_forward(x_eff->storage(), W_eff->storage(), b_eff->storage(), B, Cin, Cout,
                           Cin_g, Cout_g, S, K, O, opts, out_shape, eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape), eff_dt,
                                            x_eff->device(), false);
    // wire_autograd (below) calls ``on_op_io`` internally — recording it
    // here too would double-feed the trace (6 inputs instead of 3) and
    // abort the compile.  Conv1d/3d still rely on the wire_autograd
    // path; the explicit hook is intentionally omitted.

    auto bwd = std::make_shared<ConvNdBackward<N>>();
    for (int i = 0; i < N; ++i) {
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
        bwd->dilation_[i] = dilation[i];
    }
    bwd->groups_ = groups;
    // saved_inputs_[0..2] hold the *cast* (effective-dtype) {x, W, b} so
    // the backward conv_general dispatches at eff_dt to match the forward.
    kernel::NaryKernel<ConvNdBackward<N>, 3>::wire_autograd(std::move(bwd),
                                                            {x_eff, W_eff, b_eff}, out);
    return out;
}

template <int N>
std::vector<Storage> ConvNdBackward<N>::apply(Storage grad_out) {
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int Cin = static_cast<int>(this->input_shapes_[0][1]);
    const int Cout = static_cast<int>(this->input_shapes_[1][0]);
    const int Cin_g = Cin / this->groups_;
    const int Cout_g = Cout / this->groups_;
    int S[N], K[N], O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        K[i] = static_cast<int>(this->input_shapes_[1][2 + i]);
        O[i] = static_cast<int>(this->out_shape_[2 + i]);
    }
    backend::IBackend::ConvNdOpts opts{};
    opts.N = N;
    opts.groups = this->groups_;
    for (int i = 0; i < N; ++i) {
        opts.stride[i] = this->stride_[i];
        opts.pad[i] = this->pad_[i];
        opts.dilation[i] = this->dilation_[i];
    }
    auto& be = backend::Dispatcher::for_device(this->device_);
    // Returns [dx, dW, db].
    return be.conv_nd_backward(grad_out, this->saved_inputs_[0], this->saved_inputs_[1], B, Cin,
                               Cout, Cin_g, Cout_g, S, K, O, opts, this->dtype_);
}

template class ConvNdBackward<1>;
template class ConvNdBackward<2>;
template class ConvNdBackward<3>;

// Entry points flatten individual scalar parameters into fixed-size arrays.
TensorImplPtr conv1d_op(const TensorImplPtr& x,
                        const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sl,
                        int pl,
                        int dl,
                        int groups) {
    int stride[1]{sl};
    int pad[1]{pl};
    int dilation[1]{dl};
    return Conv1dBackward::forward(x, W, b, stride, pad, dilation, groups);
}
TensorImplPtr conv2d_op(const TensorImplPtr& x,
                        const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sh,
                        int sw,
                        int ph,
                        int pw,
                        int dh,
                        int dw,
                        int groups) {
    int stride[2]{sh, sw};
    int pad[2]{ph, pw};
    int dilation[2]{dh, dw};
    return Conv2dBackward::forward(x, W, b, stride, pad, dilation, groups);
}
TensorImplPtr conv3d_op(const TensorImplPtr& x,
                        const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sd,
                        int sh,
                        int sw,
                        int pd,
                        int ph,
                        int pw,
                        int dd,
                        int dh,
                        int dw,
                        int groups) {
    int stride[3]{sd, sh, sw};
    int pad[3]{pd, ph, pw};
    int dilation[3]{dd, dh, dw};
    return Conv3dBackward::forward(x, W, b, stride, pad, dilation, groups);
}

LUCID_REGISTER_OP(Conv1dBackward)
LUCID_REGISTER_OP(Conv2dBackward)
LUCID_REGISTER_OP(Conv3dBackward)

const OpSchema UnfoldBackward::schema_v1{"unfold", 1, AmpPolicy::KeepInput, true};

TensorImplPtr UnfoldBackward::forward(const TensorImplPtr& x,
                                      const std::vector<int>& kernel,
                                      const std::vector<int>& stride,
                                      const std::vector<int>& pad,
                                      const std::vector<int>& dilation) {
    Validator::input(x, "unfold.x").non_null();

    const int N = static_cast<int>(kernel.size());
    if (N < 1 || N > 3)
        ErrorBuilder("unfold").fail("only 1D / 2D / 3D supported");
    if (static_cast<int>(stride.size()) != N || static_cast<int>(pad.size()) != N ||
        static_cast<int>(dilation.size()) != N)
        ErrorBuilder("unfold").fail("stride/pad/dilation length must match kernel");
    if (static_cast<int>(x->shape().size()) != N + 2)
        throw ShapeMismatch(x->shape(), Shape{}, "unfold: x rank mismatch");

    const int B = static_cast<int>(x->shape()[0]);
    const int C = static_cast<int>(x->shape()[1]);
    std::vector<int> S(N), O(N);
    const std::vector<int>& K = kernel;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape()[2 + i]);
        const int eff = dilation[i] * (K[i] - 1) + 1;
        O[i] = (S[i] + 2 * pad[i] - eff) / stride[i] + 1;
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape(), Shape{}, "unfold: non-positive output dim");
    }
    int O_total = 1;
    for (int i = 0; i < N; ++i)
        O_total *= O[i];
    int K_total = 1;
    for (int i = 0; i < N; ++i)
        K_total *= K[i];

    // Output: (B, C * prod(K), prod(O)) — the im2col column matrix.
    Shape out_shape{static_cast<std::int64_t>(B), static_cast<std::int64_t>(C * K_total),
                    static_cast<std::int64_t>(O_total)};

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), out_shape};
    {
        std::vector<std::int64_t> Kv(kernel.begin(), kernel.end());
        std::vector<std::int64_t> Sv(stride.begin(), stride.end());
        std::vector<std::int64_t> Pv(pad.begin(), pad.end());
        std::vector<std::int64_t> Dv(dilation.begin(), dilation.end());
        scope.set_attr("kernel_size", std::move(Kv));
        scope.set_attr("stride", std::move(Sv));
        scope.set_attr("padding", std::move(Pv));
        scope.set_attr("dilation", std::move(Dv));
    }

    auto& be = backend::Dispatcher::for_device(x->device());
    Storage out_storage = be.unfold_forward(x->storage(), B, C, S, K, O, stride, pad, dilation,
                                            out_shape, x->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({x}, out);
    }

    {
        auto bwd = std::make_shared<UnfoldBackward>();
        bwd->kernel_ = kernel;
        bwd->stride_ = stride;
        bwd->pad_ = pad;
        bwd->dilation_ = dilation;
        kernel::NaryKernel<UnfoldBackward, 1>::wire_autograd(std::move(bwd), {x}, out);
    }
    return out;
}

std::vector<Storage> UnfoldBackward::apply(Storage grad_out) {
    const int N = static_cast<int>(kernel_.size());
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int C = static_cast<int>(this->input_shapes_[0][1]);
    std::vector<int> S(N), K = kernel_, O(N);
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        const int eff = dilation_[i] * (K[i] - 1) + 1;
        O[i] = (S[i] + 2 * pad_[i] - eff) / stride_[i] + 1;
    }
    auto& be = backend::Dispatcher::for_device(this->device_);
    // fold / col2im to recover the original input gradient.
    return {be.unfold_backward(grad_out, B, C, S, K, O, stride_, pad_, dilation_, this->dtype_)};
}

TensorImplPtr unfold_op(const TensorImplPtr& x,
                        const std::vector<int>& kernel,
                        const std::vector<int>& stride,
                        const std::vector<int>& pad,
                        const std::vector<int>& dilation) {
    return UnfoldBackward::forward(x, kernel, stride, pad, dilation);
}

LUCID_REGISTER_OP(UnfoldBackward)

// ── Fold (col2im) ─────────────────────────────────────────────────────────────

TensorImplPtr fold_op(const TensorImplPtr& x,
                      const std::vector<int>& output_size,
                      const std::vector<int>& kernel_size,
                      const std::vector<int>& stride,
                      const std::vector<int>& padding,
                      const std::vector<int>& dilation) {
    Validator::input(x, "fold.x").non_null();
    if (x->shape().size() != 3)
        ErrorBuilder("fold").fail("input must be 3-D (N, C*kH*kW, L)");

    const int kH = kernel_size[0], kW = kernel_size[1];
    const int outH = output_size[0], outW = output_size[1];
    const int N = static_cast<int>(x->shape()[0]);
    const int C = static_cast<int>(x->shape()[1]) / (kH * kW);

    Shape out_shape = {static_cast<std::int64_t>(N), static_cast<std::int64_t>(C),
                       static_cast<std::int64_t>(outH), static_cast<std::int64_t>(outW)};
    OpScopeFull scope{"fold", x->device(), x->dtype(), out_shape};
    {
        std::vector<std::int64_t> Sv(output_size.begin(), output_size.end());
        std::vector<std::int64_t> Kv(kernel_size.begin(), kernel_size.end());
        std::vector<std::int64_t> Tv(stride.begin(), stride.end());
        std::vector<std::int64_t> Pv(padding.begin(), padding.end());
        std::vector<std::int64_t> Dv(dilation.begin(), dilation.end());
        scope.set_attr("output_size", std::move(Sv));
        scope.set_attr("kernel_size", std::move(Kv));
        scope.set_attr("stride", std::move(Tv));
        scope.set_attr("padding", std::move(Pv));
        scope.set_attr("dilation", std::move(Dv));
    }

    auto& be = backend::Dispatcher::for_device(x->device());
    Storage out = be.nn_fold(x->storage(), x->shape(), out_shape, kernel_size, stride, padding,
                             dilation, x->dtype());
    auto result = std::make_shared<TensorImpl>(std::move(out), out_shape, x->dtype(), x->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({x}, result);
    }
    return result;
}

}  // namespace lucid
