#include "Matmul.h"

#include <utility>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/gpu/MlxBridge.h"
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

using lucid::kernel::primitives::cpu_matmul_nd;
using lucid::kernel::primitives::NdMatmulInfo;
using lucid::kernel::primitives::plan_nd_matmul;

namespace lucid {

const OpSchema MatmulBackward::schema_v1{"matmul", /*version=*/1, AmpPolicy::Promote,
                                         /*deterministic=*/true};

namespace {}  // namespace

std::vector<Storage> MatmulBackward::apply(Storage grad_out) {
    // For batched / broadcast matmul: dA = grad @ B^T, dB = A^T @ grad,
    // each computed at the broadcast shape (info.a_bcast_shape /
    // .b_bcast_shape) and then reduced back to the original input shapes
    // via reduce_grad_to_shape (handles broadcast-undo).
    const auto info = plan_nd_matmul(input_shapes_[0], input_shapes_[1]);

    if (device_ == Device::GPU) {
        const auto& gA = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gB = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gA.arr || !gB.arr || !gG.arr)
            ErrorBuilder("matmul backward").fail("null GPU array");
        // dA = grad @ B^T  (B^T = swapaxes(B, -2, -1))
        // dB = A^T @ grad
        auto bT = ::mlx::core::swapaxes(*gB.arr, -2, -1);
        auto aT = ::mlx::core::swapaxes(*gA.arr, -2, -1);
        auto dA = ::mlx::core::matmul(*gG.arr, bT);
        auto dB = ::mlx::core::matmul(aT, *gG.arr);
        // reduce_grad_to_shape collapses any broadcast batch dims back to
        // the original input shapes.
        Storage dA_s{gpu::wrap_mlx_array(std::move(dA), dtype_)};
        Storage dB_s{gpu::wrap_mlx_array(std::move(dB), dtype_)};
        dA_s = reduce_grad_to_shape(dA_s, info.a_bcast_shape, input_shapes_[0], dtype_, device_);
        dB_s = reduce_grad_to_shape(dB_s, info.b_bcast_shape, input_shapes_[1], dtype_, device_);
        return {std::move(dA_s), std::move(dB_s)};
    }

    // CPU: per-slice GEMM with transA / transB at the broadcast shape.
    const CpuStorage& aRaw = std::get<CpuStorage>(saved_inputs_[0]);
    const CpuStorage& bRaw = std::get<CpuStorage>(saved_inputs_[1]);
    const CpuStorage& gRaw = std::get<CpuStorage>(grad_out);

    CpuStorage aBuf, bBuf;
    const CpuStorage* aUse = &aRaw;
    const CpuStorage* bUse = &bRaw;
    if (input_shapes_[0] != info.a_bcast_shape) {
        aBuf = detail::broadcast_cpu(aRaw, input_shapes_[0], info.a_bcast_shape, dtype_);
        aUse = &aBuf;
    }
    if (input_shapes_[1] != info.b_bcast_shape) {
        bBuf = detail::broadcast_cpu(bRaw, input_shapes_[1], info.b_bcast_shape, dtype_);
        bUse = &bBuf;
    }

    // Build the "a-shaped" backward result: grad @ B^T (per slice).
    NdMatmulInfo dA_info = info;
    dA_info.M = info.M;
    dA_info.N = info.K;
    NdMatmulInfo dB_info = info;
    dB_info.M = info.K;
    dB_info.N = info.N;

    CpuStorage dA_cpu = cpu_matmul_nd(
        gRaw, *bUse, NdMatmulInfo{info.a_bcast_shape, {}, {}, info.M, info.N, info.K, info.batch},
        /*transA=*/false, /*transB=*/true, dtype_);
    CpuStorage dB_cpu = cpu_matmul_nd(
        *aUse, gRaw, NdMatmulInfo{info.b_bcast_shape, {}, {}, info.K, info.M, info.N, info.batch},
        /*transA=*/true, /*transB=*/false, dtype_);

    Storage dA_s{std::move(dA_cpu)};
    Storage dB_s{std::move(dB_cpu)};
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

    Storage out_storage;
    if (a->device() == Device::GPU) {
        const auto& gA = std::get<GpuStorage>(a->storage());
        const auto& gB = std::get<GpuStorage>(b->storage());
        // MLX matmul handles batched N-D inputs natively (with broadcasting
        // over leading axes). No manual reshape needed.
        auto out = ::mlx::core::matmul(*gA.arr, *gB.arr);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype())};
    } else {
        // CPU: materialize broadcast for batch dims, then per-slice GEMM.
        const CpuStorage& aRaw = std::get<CpuStorage>(a->storage());
        const CpuStorage& bRaw = std::get<CpuStorage>(b->storage());
        CpuStorage aBuf, bBuf;
        const CpuStorage* aUse = &aRaw;
        const CpuStorage* bUse = &bRaw;
        if (a->shape() != info.a_bcast_shape) {
            aBuf = detail::broadcast_cpu(aRaw, a->shape(), info.a_bcast_shape, a->dtype());
            aUse = &aBuf;
        }
        if (b->shape() != info.b_bcast_shape) {
            bBuf = detail::broadcast_cpu(bRaw, b->shape(), info.b_bcast_shape, a->dtype());
            bUse = &bBuf;
        }
        out_storage = Storage{cpu_matmul_nd(*aUse, *bUse, info,
                                            /*transA=*/false,
                                            /*transB=*/false, a->dtype())};
    }

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
