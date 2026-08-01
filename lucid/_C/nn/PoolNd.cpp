// lucid/_C/nn/PoolNd.cpp
//
// N-dimensional MaxPool and AvgPool implementations.
//
// Output size formula (same as conv, no dilation):
//   O[i] = (S[i] + 2*pad[i] - K[i]) / stride[i] + 1
//
// MaxPool forward: IBackend::max_pool_nd_forward returns [out, argmax].
//   Backward: IBackend::max_pool_nd_backward scatters grad_out via argmax.
//
// AvgPool forward: IBackend::avg_pool_nd_forward returns out.
//   Backward: IBackend::avg_pool_nd_backward distributes grad uniformly.
//
// Both forward paths skip wiring the autograd node when grad is not needed
// (GradMode disabled or x.requires_grad == false).

#include "PoolNd.h"

#include <cstring>
#include <limits>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Div.h"
#include "../ops/bfunc/Maximum.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/utils/Layout.h"
#include "../ops/utils/View.h"

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

// Standard pooling output-size formula (no dilation).
inline int compute_out(int S, int K, int stride, int pad) {
    return (S + 2 * pad - K) / stride + 1;
}

// Validate that x is non-null and has rank N+2.
template <int N>
void validate_input(const TensorImplPtr& x, std::string_view op_name) {
    Validator::input(x, std::string(op_name) + ".x").non_null();
    if (static_cast<int>(x->shape().size()) != N + 2)
        throw ShapeMismatch(x->shape(), Shape{}, std::string(op_name) + ": x rank mismatch");
}

}  // namespace

template <int N>
TensorImplPtr MaxPoolNdBackward<N>::forward(const TensorImplPtr& x,
                                            const int (&K)[N],
                                            const int (&stride_in)[N],
                                            const int (&pad)[N]) {
    validate_input<N>(x, MaxPoolNdBackward<N>::schema_v1.name);
    int stride[N];
    // stride == 0 is a sentinel meaning "use kernel size" (non-overlapping).
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
    // 3.5 Phase 1.2: report pool params so the compile-path emitter can
    // rebuild the descriptor.
    {
        std::vector<std::int64_t> Kv(K, K + N), Sv(stride, stride + N), Pv(pad, pad + N);
        scope.set_attr("kernel_size", std::move(Kv));
        scope.set_attr("stride", std::move(Sv));
        scope.set_attr("padding", std::move(Pv));
    }

    backend::IBackend::PoolOpts opts{};
    opts.N = N;
    for (int i = 0; i < N; ++i) {
        opts.K[i] = K[i];
        opts.stride[i] = stride[i];
        opts.pad[i] = pad[i];
    }
    auto& be = backend::Dispatcher::for_device(x->device());
    // max_pool_nd_forward returns [output, argmax].
    auto pool_out = be.max_pool_nd_forward(x->storage(), x->shape(), out_shape, opts, x->dtype());
    Storage out_storage = std::move(pool_out[0]);
    Storage saved_argmax = std::move(pool_out[1]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);
    // wire_autograd (below) records ``on_op_io`` internally — calling it
    // again here would double-feed the trace (2 inputs instead of 1) and
    // abort the compile.

    // wire_autograd is always invoked so the 3.5 compile-path trace
    // hook fires regardless of GradMode (autograd is gated inside).
    auto bwd = std::make_shared<MaxPoolNdBackward<N>>();
    bwd->saved_argmax_ = std::move(saved_argmax);
    for (int i = 0; i < N; ++i) {
        bwd->K_[i] = K[i];
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
    }
    kernel::NaryKernel<MaxPoolNdBackward<N>, 1>::wire_autograd(std::move(bwd), {x}, out, false);
    return out;
}

template <int N>
std::vector<Storage> MaxPoolNdBackward<N>::apply(Storage grad_out) {
    backend::IBackend::PoolOpts opts{};
    opts.N = N;
    for (int i = 0; i < N; ++i) {
        opts.K[i] = this->K_[i];
        opts.stride[i] = this->stride_[i];
        opts.pad[i] = this->pad_[i];
    }
    auto& be = backend::Dispatcher::for_device(this->device_);
    // Scatter grad to input positions using saved argmax indices.
    return {be.max_pool_nd_backward(grad_out, this->saved_argmax_, this->input_shapes_[0],
                                    this->out_shape_, opts, this->dtype_)};
}

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
    // 3.5 Phase 1.2: report pool params for the compile-path emitter.
    {
        std::vector<std::int64_t> Kv(K, K + N), Sv(stride, stride + N), Pv(pad, pad + N);
        scope.set_attr("kernel_size", std::move(Kv));
        scope.set_attr("stride", std::move(Sv));
        scope.set_attr("padding", std::move(Pv));
    }

    backend::IBackend::PoolOpts avg_opts{};
    avg_opts.N = N;
    for (int i = 0; i < N; ++i) {
        avg_opts.K[i] = K[i];
        avg_opts.stride[i] = stride[i];
        avg_opts.pad[i] = pad[i];
    }
    auto& avg_be = backend::Dispatcher::for_device(x->device());
    Storage out_storage =
        avg_be.avg_pool_nd_forward(x->storage(), x->shape(), out_shape, avg_opts, x->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);
    // wire_autograd (below) records ``on_op_io`` internally — calling it
    // again here would double-feed the trace.

    // wire_autograd always — trace-hook visibility under no-grad.
    auto bwd = std::make_shared<AvgPoolNdBackward<N>>();
    for (int i = 0; i < N; ++i) {
        bwd->K_[i] = K[i];
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
    }
    kernel::NaryKernel<AvgPoolNdBackward<N>, 1>::wire_autograd(std::move(bwd), {x}, out, false);
    return out;
}

template <int N>
std::vector<Storage> AvgPoolNdBackward<N>::apply(Storage grad_out) {
    backend::IBackend::PoolOpts opts{};
    opts.N = N;
    for (int i = 0; i < N; ++i) {
        opts.K[i] = this->K_[i];
        opts.stride[i] = this->stride_[i];
        opts.pad[i] = this->pad_[i];
    }
    auto& be = backend::Dispatcher::for_device(this->device_);
    // Distribute grad evenly across each pooling window.
    return {be.avg_pool_nd_backward(grad_out, this->input_shapes_[0], this->out_shape_, opts,
                                    this->dtype_)};
}

template <int N>
std::vector<TensorImplPtr> MaxPoolNdBackward<N>::apply_for_graph(const TensorImplPtr& grad_out) {
    // Max pooling sends each output's gradient back to the one input it came
    // from, so the adjoint is a scatter at the saved winners.  Those are
    // integers with no derivative, which is why reusing them costs nothing —
    // unlike a saved statistic, which depends on the input and has to be
    // recomputed.
    const Shape& xs = this->input_shapes_[0];
    std::int64_t planes = xs[0] * xs[1];
    std::int64_t in_plane = 1, out_plane = 1;
    for (std::size_t i = 2; i < xs.size(); ++i)
        in_plane *= xs[i];
    for (std::size_t i = 2; i < this->out_shape_.size(); ++i)
        out_plane *= this->out_shape_[i];

    const std::vector<std::int64_t> flat_out{planes, out_plane};

    // The index stays I32: scatter_add's backend reads it as int32_t and
    // never sees a dtype, so widening it here would be read back at half
    // width — the gradient lands on the wrong elements and nothing raises.
    auto idx = std::make_shared<TensorImpl>(this->saved_argmax_, this->out_shape_, Dtype::I32,
                                            this->device_, false);
    idx = reshape_op(idx, flat_out);
    auto g = reshape_op(grad_out, flat_out);

    // A window that saw only padding has no winner.  Clamping the index and
    // masking the gradient says that in dtypes both streams support; the CPU
    // backend has no integer ``where``.
    auto zeros_idx = zeros_like_op(idx);
    auto mask = astype_op(greater_equal_op(idx, zeros_idx), this->dtype_);
    auto safe_idx = maximum_op(idx, zeros_idx);
    auto src = mul_op(g, mask);

    std::vector<std::int64_t> in_shape(xs.begin(), xs.end());
    auto base = zeros_op(Shape{planes, in_plane}, this->dtype_, this->device_);
    auto dx = scatter_add_op(base, safe_idx, src, /*dim=*/1);
    return {reshape_op(dx, in_shape)};
}

template class MaxPoolNdBackward<1>;
template class MaxPoolNdBackward<2>;
template class MaxPoolNdBackward<3>;
template <int N>
std::vector<TensorImplPtr> AvgPoolNdBackward<N>::apply_for_graph(const TensorImplPtr& grad_out) {
    // Average pooling spreads each output's gradient evenly over its window,
    // so with non-overlapping unpadded windows the adjoint is exactly a
    // repeat: expand each output element across its K positions and divide
    // by the window size.  Overlapping or padded windows are a
    // conv-transpose against a constant kernel instead, and conv-transpose
    // has no graph-mode formula yet — so those are refused by name rather
    // than answered with the wrong tensor.
    const Shape& xs = this->input_shapes_[0];
    for (int i = 0; i < N; ++i) {
        const bool tiles = this->stride_[i] == this->K_[i] && this->pad_[i] == 0 &&
                           xs[static_cast<std::size_t>(2 + i)] ==
                               this->out_shape_[static_cast<std::size_t>(2 + i)] * this->K_[i];
        if (!tiles)
            ErrorBuilder("avg_pool")
                .not_implemented(
                    "create_graph=True needs non-overlapping, unpadded windows that tile "
                    "the input exactly (stride == kernel, padding == 0)");
    }

    // (B, C, O0, O1, ...) -> (B, C, O0, 1, O1, 1, ...) -> broadcast the 1s to
    // K -> (B, C, O0*K0, O1*K1, ...).
    std::vector<std::int64_t> split{xs[0], xs[1]};
    std::vector<std::int64_t> blown{xs[0], xs[1]};
    for (int i = 0; i < N; ++i) {
        split.push_back(this->out_shape_[static_cast<std::size_t>(2 + i)]);
        split.push_back(1);
        blown.push_back(this->out_shape_[static_cast<std::size_t>(2 + i)]);
        blown.push_back(this->K_[i]);
    }
    std::int64_t window = 1;
    for (int i = 0; i < N; ++i)
        window *= this->K_[i];

    auto spread = broadcast_to_op(reshape_op(grad_out, split), Shape(blown.begin(), blown.end()));
    std::vector<std::int64_t> in_shape(xs.begin(), xs.end());
    auto dx = reshape_op(spread, in_shape);
    return {div_op(dx, full_like_op(dx, static_cast<double>(window)))};
}

template class AvgPoolNdBackward<1>;
template class AvgPoolNdBackward<2>;
template class AvgPoolNdBackward<3>;

// Entry points pack scalar parameters into fixed-size arrays.
TensorImplPtr max_pool1d_op(const TensorImplPtr& x, int KL, int sl, int pl) {
    int K[1]{KL};
    int s[1]{sl};
    int p[1]{pl};
    return MaxPool1dBackward::forward(x, K, s, p);
}
TensorImplPtr
max_pool2d_op(const TensorImplPtr& x, int KH, int KW, int sh, int sw, int ph, int pw) {
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
TensorImplPtr
avg_pool2d_op(const TensorImplPtr& x, int KH, int KW, int sh, int sw, int ph, int pw) {
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
