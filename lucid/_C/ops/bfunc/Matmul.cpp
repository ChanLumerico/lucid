// lucid/_C/ops/bfunc/Matmul.cpp
//
// Implements MatmulBackward::forward, MatmulBackward::apply, and the matmul_op
// free function.

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
#include "../bfunc/_BinaryOp.h"

using lucid::kernel::primitives::NdMatmulInfo;
using lucid::kernel::primitives::plan_nd_matmul;

namespace lucid {

const OpSchema MatmulBackward::schema_v1{"matmul", 1, AmpPolicy::Promote, true};

namespace {

// Build a MatmulOpts struct for use with IBackend::matmul from a pre-computed
// NdMatmulInfo, optionally transposing either operand.
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

// Return storage broadcast to dst_shape if src_shape differs; otherwise return
// the original storage unchanged to avoid an unnecessary allocation.
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

// Compute dA and dB from the upstream gradient tensor.
//
// Given C = A @ B  (shapes [batch, M, K] and [batch, K, N]):
//   dA = grad_out @ B^T   → [batch, M, K]
//   dB = A^T @ grad_out   → [batch, K, N]
//
// Both saved inputs are first broadcast to the shapes that were used during the
// forward pass (info.a_bcast_shape and info.b_bcast_shape).  After the matmuls,
// reduce_grad_to_shape collapses any batch dimensions that were broadcast-
// expanded, recovering tensors with the original input shapes.
std::vector<Storage> MatmulBackward::apply(Storage grad_out) {
    const auto info = plan_nd_matmul(input_shapes_[0], input_shapes_[1]);

    Storage a_use = broadcast_for_matmul(saved_inputs_[0], input_shapes_[0], info.a_bcast_shape,
                                         dtype_, device_);
    Storage b_use = broadcast_for_matmul(saved_inputs_[1], input_shapes_[1], info.b_bcast_shape,
                                         dtype_, device_);

    NdMatmulInfo dA_info = info;
    dA_info.M = info.M;
    dA_info.N = info.K;
    NdMatmulInfo dB_info = info;
    dB_info.M = info.K;
    dB_info.N = info.N;

    // dA opts: grad_out [batch, M, N] @ B^T [batch, N, K] → [batch, M, K]
    backend::MatmulOpts dA_opts;
    dA_opts.M = info.M;
    dA_opts.K = info.N;   // grad_out inner dim is N
    dA_opts.N = info.K;   // output inner dim recovers K
    dA_opts.batch = info.batch;
    dA_opts.transB = true;

    // dB opts: A^T [batch, K, M] @ grad_out [batch, M, N] → [batch, K, N]
    backend::MatmulOpts dB_opts;
    dB_opts.M = info.K;   // output leading dim is K
    dB_opts.K = info.M;   // inner dim is M
    dB_opts.N = info.N;
    dB_opts.batch = info.batch;
    dB_opts.transA = true;

    Storage dA_s =
        backend::Dispatcher::for_device(device_).matmul(grad_out, b_use, dA_opts, dtype_);
    Storage dB_s =
        backend::Dispatcher::for_device(device_).matmul(a_use, grad_out, dB_opts, dtype_);

    // Sum over any batch dimensions that were broadcast-expanded.
    dA_s = reduce_grad_to_shape(dA_s, info.a_bcast_shape, input_shapes_[0], dtype_, device_);
    dB_s = reduce_grad_to_shape(dB_s, info.b_bcast_shape, input_shapes_[1], dtype_, device_);
    return {std::move(dA_s), std::move(dB_s)};
}

// Execute the forward matmul, open a profiler scope, compute FLOPs, and wire
// the backward node if gradient tracking is active.
//
// Validation requires:
//   - both inputs non-null, same dtype and device
//   - both inputs at least 2-D (plan_nd_matmul handles batch broadcasting)
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
    // Each output element requires K multiplications and K-1 additions ≈ 2*K MACs.
    scope.set_flops(static_cast<std::int64_t>(2) * info.batch * M * N * K);

    Storage a_use =
        broadcast_for_matmul(a->storage(), a->shape(), info.a_bcast_shape, a->dtype(), a->device());
    Storage b_use =
        broadcast_for_matmul(b->storage(), b->shape(), info.b_bcast_shape, a->dtype(), a->device());
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .matmul(a_use, b_use, matmul_opts(info, false, false), a->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), info.out_shape, a->dtype(),
                                            a->device(), false);

    kernel::NaryKernel<MatmulBackward, 2>::wire_autograd({a, b}, out);
    return out;
}

// Graph-mode matmul backward: dA = grad_out @ B^T,  dB = A^T @ grad_out.
std::vector<TensorImplPtr> MatmulBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    auto& a = saved_impl_inputs_[0];
    auto& b = saved_impl_inputs_[1];
    if (!a || !b) {
        throw std::runtime_error(
            "apply_for_graph: saved_impl_inputs_ not set for matmul.");
    }

    extern TensorImplPtr mT_op(const TensorImplPtr&);
    extern TensorImplPtr sum_op(const TensorImplPtr&, const std::vector<int>&, bool);
    extern TensorImplPtr reshape_op(const TensorImplPtr&, const Shape&);

    auto da = matmul_op(grad_out, mT_op(b));
    auto db = matmul_op(mT_op(a), grad_out);

    auto reduce = [&](TensorImplPtr g, const Shape& target) -> TensorImplPtr {
        if (g->shape() == target) return g;
        std::vector<int> axes;
        const int ng = static_cast<int>(g->shape().size());
        const int nt = static_cast<int>(target.size());
        for (int i = 0; i < ng - nt; ++i) axes.push_back(i);
        for (int i = 0; i < nt; ++i) {
            if (target[static_cast<std::size_t>(i)] == 1 &&
                g->shape()[static_cast<std::size_t>(i + ng - nt)] != 1)
                axes.push_back(i + ng - nt);
        }
        if (!axes.empty()) g = sum_op(g, axes, false);
        if (g->shape() != target) g = reshape_op(g, target);
        return g;
    };

    return {reduce(da, input_shapes_[0]), reduce(db, input_shapes_[1])};
}

TensorImplPtr matmul_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MatmulBackward::forward(a, b);
}

LUCID_REGISTER_OP(MatmulBackward)

}  // namespace lucid
