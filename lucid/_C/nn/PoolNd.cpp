#include "PoolNd.h"

#include <cstring>
#include <limits>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
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

inline int compute_out(int S, int K, int stride, int pad) {
    return (S + 2 * pad - K) / stride + 1;
}

template <int N>
void validate_input(const TensorImplPtr& x, std::string_view op_name) {
    Validator::input(x, std::string(op_name) + ".x").non_null();
    if (static_cast<int>(x->shape().size()) != N + 2)
        throw ShapeMismatch(x->shape(), Shape{}, std::string(op_name) + ": x rank mismatch");
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

    backend::IBackend::PoolOpts opts{};
    opts.N = N;
    for (int i = 0; i < N; ++i) {
        opts.K[i] = K[i];
        opts.stride[i] = stride[i];
        opts.pad[i] = pad[i];
    }
    auto& be = backend::Dispatcher::for_device(x->device());
    auto pool_out = be.max_pool_nd_forward(x->storage(), x->shape(), out_shape, opts, x->dtype());
    Storage out_storage = std::move(pool_out[0]);
    Storage saved_argmax = std::move(pool_out[1]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);
    if (!GradMode::is_enabled() || !x->requires_grad())
        return out;

    auto bwd = std::make_shared<MaxPoolNdBackward<N>>();
    bwd->saved_argmax_ = std::move(saved_argmax);
    for (int i = 0; i < N; ++i) {
        bwd->K_[i] = K[i];
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
    }
    kernel::NaryKernel<MaxPoolNdBackward<N>, 1>::wire_autograd(std::move(bwd), {x}, out,
                                                               /*save_ins=*/false);
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
    return {be.max_pool_nd_backward(grad_out, this->saved_argmax_, this->input_shapes_[0],
                                    this->out_shape_, opts, this->dtype_)};
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
    if (!GradMode::is_enabled() || !x->requires_grad())
        return out;

    auto bwd = std::make_shared<AvgPoolNdBackward<N>>();
    for (int i = 0; i < N; ++i) {
        bwd->K_[i] = K[i];
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
    }
    kernel::NaryKernel<AvgPoolNdBackward<N>, 1>::wire_autograd(std::move(bwd), {x}, out,
                                                               /*save_ins=*/false);
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
    return {be.avg_pool_nd_backward(grad_out, this->input_shapes_[0], this->out_shape_, opts,
                                    this->dtype_)};
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
