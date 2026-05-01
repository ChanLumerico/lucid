#include "Matmul.h"

#include <utility>
#include <variant>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "../../kernel/primitives/BatchedMatmul.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

using lucid::kernel::primitives::NdMatmulInfo;
using lucid::kernel::primitives::plan_nd_matmul;

namespace lucid {

const OpSchema MatmulBackward::schema_v1{"matmul", /*version=*/1, AmpPolicy::Promote,
                                         /*deterministic=*/true};

namespace {

backend::MatmulOpts matmul_opts(const NdMatmulInfo& info, bool transA, bool transB) {
    backend::MatmulOpts opts;
    opts.transA = transA;
    opts.transB = transB;
    opts.M = info.M;
    opts.K = info.K;
    opts.N = info.N;
    opts.batch = info.batch;
    return opts;
}

Storage broadcast_for_matmul(const Storage& storage,
                             const Shape& src_shape,
                             const Shape& dst_shape,
                             Dtype dt,
                             Device device) {
    if (src_shape == dst_shape)
        return storage;
    return backend::Dispatcher::for_device(device).broadcast(storage, src_shape, dst_shape, dt);
}

}  // namespace

std::vector<Storage> MatmulBackward::apply(Storage grad_out) {
    // For batched / broadcast matmul: dA = grad @ B^T, dB = A^T @ grad,
    // each computed at the broadcast shape (info.a_bcast_shape /
    // .b_bcast_shape) and then reduced back to the original input shapes
    // via reduce_grad_to_shape (handles broadcast-undo).
    const auto info = plan_nd_matmul(input_shapes_[0], input_shapes_[1]);

    Storage a_use = broadcast_for_matmul(saved_inputs_[0], input_shapes_[0], info.a_bcast_shape,
                                         dtype_, device_);
    Storage b_use = broadcast_for_matmul(saved_inputs_[1], input_shapes_[1], info.b_bcast_shape,
                                         dtype_, device_);

    // Build the "a-shaped" backward result: grad @ B^T (per slice).
    NdMatmulInfo dA_info = info;
    dA_info.M = info.M;
    dA_info.N = info.K;
    NdMatmulInfo dB_info = info;
    dB_info.M = info.K;
    dB_info.N = info.N;

    backend::MatmulOpts dA_opts;
    dA_opts.M = info.M;
    dA_opts.K = info.N;
    dA_opts.N = info.K;
    dA_opts.batch = info.batch;
    dA_opts.transB = true;
    backend::MatmulOpts dB_opts;
    dB_opts.M = info.K;
    dB_opts.K = info.M;
    dB_opts.N = info.N;
    dB_opts.batch = info.batch;
    dB_opts.transA = true;

    Storage dA_s =
        backend::Dispatcher::for_device(device_).matmul(grad_out, b_use, dA_opts, dtype_);
    Storage dB_s =
        backend::Dispatcher::for_device(device_).matmul(a_use, grad_out, dB_opts, dtype_);
    dA_s = reduce_grad_to_shape(dA_s, info.a_bcast_shape, input_shapes_[0], dtype_, device_);
    dB_s = reduce_grad_to_shape(dB_s, info.b_bcast_shape, input_shapes_[1], dtype_, device_);
    return {std::move(dA_s), std::move(dB_s)};
}

TensorImplPtr MatmulBackward::forward(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b)
        ErrorBuilder("matmul").fail("null input");
    if (a->dtype() != b->dtype())
        throw DtypeMismatch(std::string(dtype_name(a->dtype())),
                            std::string(dtype_name(b->dtype())), "matmul");
    if (a->device() != b->device())
        throw DeviceMismatch(std::string(device_name(a->device())),
                             std::string(device_name(b->device())), "matmul");
    if (a->shape().size() < 2 || b->shape().size() < 2) {
        throw ShapeMismatch(a->shape(), b->shape(), "matmul: both operands must be ≥2-D");
    }

    const auto info = plan_nd_matmul(a->shape(), b->shape());
    const int M = info.M, N = info.N, K = info.K;

    OpScopeFull scope{MatmulBackward::schema_v1.name, a->device(), a->dtype(), info.out_shape};
    scope.set_flops(static_cast<std::int64_t>(2) * info.batch * M * N * K);

    Storage a_use =
        broadcast_for_matmul(a->storage(), a->shape(), info.a_bcast_shape, a->dtype(), a->device());
    Storage b_use =
        broadcast_for_matmul(b->storage(), b->shape(), info.b_bcast_shape, a->dtype(), a->device());
    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
            .matmul(a_use, b_use, matmul_opts(info, /*transA=*/false, /*transB=*/false),
                    a->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), info.out_shape, a->dtype(),
                                            a->device(),
                                            /*requires_grad=*/false);

    kernel::NaryKernel<MatmulBackward, 2>::wire_autograd({a, b}, out);
    return out;
}

TensorImplPtr matmul_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MatmulBackward::forward(a, b);
}

LUCID_REGISTER_OP(MatmulBackward)

}  // namespace lucid
