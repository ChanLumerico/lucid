// lucid/_C/nn/ConvTransposeNd.cpp
//
// Transposed convolution (deconvolution) forward and backward.
//
// Output shape per spatial axis i:
//   O[i] = (S[i] - 1) * stride[i] - 2 * pad[i] + K[i] + opad[i]
//
// FLOP estimate: 2 * B * C_out * prod(O) * C_in * prod(K)
//
// The forward delegates to IBackend::conv_transpose_nd_forward (col2im + GEMM
// on CPU, MLX conv-transpose on GPU).  The backward delegates to
// IBackend::conv_transpose_nd_backward, which returns [dx, dW, db].

#include "ConvTransposeNd.h"

#include <vector>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

template <>
const OpSchema ConvTranspose1dBackward::schema_v1{"conv_transpose1d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema ConvTranspose2dBackward::schema_v1{"conv_transpose2d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema ConvTranspose3dBackward::schema_v1{"conv_transpose3d", 1, AmpPolicy::Promote, true};

template <int N>
TensorImplPtr ConvTransposeNdBackward<N>::forward(const TensorImplPtr& x,
                                                  const TensorImplPtr& W,
                                                  const TensorImplPtr& b,
                                                  const int (&stride)[N],
                                                  const int (&pad)[N],
                                                  const int (&opad)[N]) {
    if (!x || !W || !b)
        ErrorBuilder("conv_transpose").fail("null input");
    if (x->dtype() != W->dtype() || x->dtype() != b->dtype())
        throw DtypeMismatch(std::string(dtype_name(x->dtype())),
                            std::string(dtype_name(W->dtype())), "conv_transpose");
    if (x->device() != W->device() || x->device() != b->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(W->device())), "conv_transpose");
    if (static_cast<int>(x->shape().size()) != N + 2)
        throw ShapeMismatch(x->shape(), Shape{}, "conv_transpose: x rank mismatch");
    // Weight layout for transpose: (C_in, C_out, K_0, ..., K_{N-1}).
    if (static_cast<int>(W->shape().size()) != N + 2)
        throw ShapeMismatch(W->shape(), Shape{}, "conv_transpose: W rank mismatch");
    if (b->shape().size() != 1)
        throw ShapeMismatch(b->shape(), Shape{}, "conv_transpose: b must be 1-D");

    const int B = static_cast<int>(x->shape()[0]);
    const int Cin = static_cast<int>(x->shape()[1]);
    const int Cw = static_cast<int>(W->shape()[0]);
    const int Cout = static_cast<int>(W->shape()[1]);
    if (Cw != Cin)
        throw ShapeMismatch(W->shape(), x->shape(), "conv_transpose: C_in mismatch");
    if (b->shape()[0] != Cout)
        throw ShapeMismatch(b->shape(), W->shape(), "conv_transpose: bias C_out mismatch");

    int S[N], K[N], O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape()[2 + i]);
        K[i] = static_cast<int>(W->shape()[2 + i]);
        // Transposed-conv output formula.
        O[i] = (S[i] - 1) * stride[i] - 2 * pad[i] + K[i] + opad[i];
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape(), W->shape(),
                                "conv_transpose: output shape non-positive");
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

    OpScopeFull scope{ConvTransposeNdBackward<N>::schema_v1.name, x->device(), x->dtype(),
                      out_shape};
    scope.set_flops(static_cast<std::int64_t>(2) * B * Cout * O_total * Cin * K_total);

    Storage out_storage =
        backend::Dispatcher::for_device(x->device())
            .conv_transpose_nd_forward(x->storage(), W->storage(), b->storage(), B, Cin, Cout, S, K,
                                       O, stride, pad, opad, N, out_shape, x->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);

    auto bwd = std::make_shared<ConvTransposeNdBackward<N>>();
    for (int i = 0; i < N; ++i) {
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
        bwd->opad_[i] = opad[i];
    }
    // saved_inputs_[0..2] will hold {x, W, b}.
    kernel::NaryKernel<ConvTransposeNdBackward<N>, 3>::wire_autograd(std::move(bwd), {x, W, b},
                                                                     out);
    return out;
}

template <int N>
std::vector<Storage> ConvTransposeNdBackward<N>::apply(Storage grad_out) {
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int Cin = static_cast<int>(this->input_shapes_[0][1]);
    const int Cout = static_cast<int>(this->input_shapes_[1][1]);
    int S[N], K[N], O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        K[i] = static_cast<int>(this->input_shapes_[1][2 + i]);
        O[i] = static_cast<int>(this->out_shape_[2 + i]);
    }

    // Returns [dx, dW, db].
    return backend::Dispatcher::for_device(this->device_)
        .conv_transpose_nd_backward(grad_out, this->saved_inputs_[0], this->saved_inputs_[1], B,
                                    Cin, Cout, S, K, O, this->stride_, this->pad_, N, this->dtype_);
}

template class ConvTransposeNdBackward<1>;
template class ConvTransposeNdBackward<2>;
template class ConvTransposeNdBackward<3>;

// Entry points pack scalar parameters into fixed-size arrays.
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
