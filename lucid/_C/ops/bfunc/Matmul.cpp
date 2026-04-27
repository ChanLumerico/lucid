#include "Matmul.h"

#include <utility>
#include <variant>

#include <mlx/ops.h>

#include "../../backend/cpu/Blas.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

const OpSchema MatmulBackward::schema_v1{
    "matmul", /*version=*/1, AmpPolicy::Promote, /*deterministic=*/true};

namespace {

CpuStorage allocate_2d(int M, int N, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

CpuStorage cpu_matmul(const CpuStorage& a, const CpuStorage& b,
                      int M, int N, int K, bool transA, bool transB, Dtype dt) {
    auto out = allocate_2d(M, N, dt);
    const int lda = transA ? M : K;
    const int ldb = transB ? K : N;
    switch (dt) {
        case Dtype::F32:
            backend::cpu::sgemm(transA, transB, M, N, K, 1.0f,
                                reinterpret_cast<const float*>(a.ptr.get()), lda,
                                reinterpret_cast<const float*>(b.ptr.get()), ldb,
                                0.0f,
                                reinterpret_cast<float*>(out.ptr.get()), N);
            break;
        case Dtype::F64:
            backend::cpu::dgemm(transA, transB, M, N, K, 1.0,
                                reinterpret_cast<const double*>(a.ptr.get()), lda,
                                reinterpret_cast<const double*>(b.ptr.get()), ldb,
                                0.0,
                                reinterpret_cast<double*>(out.ptr.get()), N);
            break;
        default:
            throw NotImplementedError("matmul: dtype not supported (Phase 3.1: F32/F64)");
    }
    return out;
}

// CPU N-D matmul: a [..., M, K] @ b [..., K, N] → out [..., M, N]
// Leading dims of a/b are broadcast-aligned; the kernel iterates over the
// flattened batch dim and calls sgemm/dgemm per slice.
struct NdMatmulInfo {
    Shape out_shape;
    Shape a_bcast_shape;
    Shape b_bcast_shape;
    int M, K, N;
    std::size_t batch;
};

NdMatmulInfo plan_nd_matmul(const Shape& a, const Shape& b) {
    if (a.size() < 2 || b.size() < 2)
        throw ShapeMismatch(a, b, "matmul: both operands must be ≥2-D");
    const std::size_t na = a.size(), nb = b.size();
    const std::int64_t M = a[na - 2], Ka = a[na - 1];
    const std::int64_t Kb = b[nb - 2], N = b[nb - 1];
    if (Ka != Kb)
        throw ShapeMismatch(a, b, "matmul: inner dim mismatch");

    Shape ba(a.begin(), a.end() - 2);
    Shape bb(b.begin(), b.end() - 2);
    Shape out_b;
    if (ba.empty()) out_b = bb;
    else if (bb.empty()) out_b = ba;
    else {
        // Broadcast leading batch dims.
        const std::size_t r = std::max(ba.size(), bb.size());
        out_b.assign(r, 1);
        for (std::size_t i = 0; i < r; ++i) {
            const std::size_t ai = (ba.size() >= r - i) ? ba.size() - (r - i) : SIZE_MAX;
            const std::size_t bi = (bb.size() >= r - i) ? bb.size() - (r - i) : SIZE_MAX;
            const std::int64_t da = (ai != SIZE_MAX) ? ba[ai] : 1;
            const std::int64_t db = (bi != SIZE_MAX) ? bb[bi] : 1;
            if (da == db || da == 1 || db == 1) out_b[i] = da == 1 ? db : da;
            else throw ShapeMismatch(a, b, "matmul: incompatible batch dims");
        }
    }
    NdMatmulInfo info;
    info.out_shape = out_b;
    info.out_shape.push_back(M);
    info.out_shape.push_back(N);
    info.a_bcast_shape = out_b;
    info.a_bcast_shape.push_back(M);
    info.a_bcast_shape.push_back(Ka);
    info.b_bcast_shape = out_b;
    info.b_bcast_shape.push_back(Kb);
    info.b_bcast_shape.push_back(N);
    info.M = static_cast<int>(M);
    info.K = static_cast<int>(Ka);
    info.N = static_cast<int>(N);
    std::size_t batch = 1;
    for (auto d : out_b) batch *= static_cast<std::size_t>(d);
    info.batch = batch;
    return info;
}

CpuStorage cpu_matmul_nd(const CpuStorage& a, const CpuStorage& b,
                          const NdMatmulInfo& info, bool transA, bool transB,
                          Dtype dt) {
    const std::size_t batch = info.batch;
    const int M = info.M, K = info.K, N = info.N;
    const std::size_t a_step = static_cast<std::size_t>(M) * K;
    const std::size_t b_step = static_cast<std::size_t>(K) * N;
    const std::size_t o_step = static_cast<std::size_t>(M) * N;
    CpuStorage out;
    out.dtype  = dt;
    out.nbytes = batch * o_step * dtype_size(dt);
    out.ptr    = allocate_aligned_bytes(out.nbytes);
    if (M == 0 || N == 0 || K == 0) {
        if (out.nbytes) std::memset(out.ptr.get(), 0, out.nbytes);
        return out;
    }
    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* ap = reinterpret_cast<const T*>(a.ptr.get());
        const T* bp = reinterpret_cast<const T*>(b.ptr.get());
        T* op = reinterpret_cast<T*>(out.ptr.get());
        const int lda = transA ? M : K;
        const int ldb = transB ? K : N;
        for (std::size_t bi = 0; bi < batch; ++bi) {
            if constexpr (std::is_same_v<T, float>) {
                backend::cpu::sgemm(transA, transB, M, N, K, 1.0f,
                                      ap + bi * a_step, lda,
                                      bp + bi * b_step, ldb, 0.0f,
                                      op + bi * o_step, N);
            } else {
                backend::cpu::dgemm(transA, transB, M, N, K, 1.0,
                                      ap + bi * a_step, lda,
                                      bp + bi * b_step, ldb, 0.0,
                                      op + bi * o_step, N);
            }
        }
    };
    if (dt == Dtype::F32) run(float{});
    else if (dt == Dtype::F64) run(double{});
    else throw NotImplementedError("matmul: dtype not supported (F32/F64)");
    return out;
}

}  // namespace

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
            throw LucidError("matmul backward: null GPU array");
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
        dA_s = reduce_grad_to_shape(std::move(dA_s), info.a_bcast_shape,
                                       input_shapes_[0], dtype_, device_);
        dB_s = reduce_grad_to_shape(std::move(dB_s), info.b_bcast_shape,
                                       input_shapes_[1], dtype_, device_);
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
    dA_info.M = info.M; dA_info.N = info.K;
    NdMatmulInfo dB_info = info;
    dB_info.M = info.K; dB_info.N = info.N;

    CpuStorage dA_cpu = cpu_matmul_nd(gRaw, *bUse, NdMatmulInfo{
        info.a_bcast_shape, {}, {}, info.M, info.N, info.K, info.batch},
        /*transA=*/false, /*transB=*/true, dtype_);
    CpuStorage dB_cpu = cpu_matmul_nd(*aUse, gRaw, NdMatmulInfo{
        info.b_bcast_shape, {}, {}, info.K, info.M, info.N, info.batch},
        /*transA=*/true, /*transB=*/false, dtype_);

    Storage dA_s{std::move(dA_cpu)};
    Storage dB_s{std::move(dB_cpu)};
    dA_s = reduce_grad_to_shape(std::move(dA_s), info.a_bcast_shape,
                                   input_shapes_[0], dtype_, device_);
    dB_s = reduce_grad_to_shape(std::move(dB_s), info.b_bcast_shape,
                                   input_shapes_[1], dtype_, device_);
    return {std::move(dA_s), std::move(dB_s)};
}

TensorImplPtr MatmulBackward::forward(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b) throw LucidError("matmul: null input");
    if (a->dtype_ != b->dtype_)
        throw DtypeMismatch(std::string(dtype_name(a->dtype_)),
                            std::string(dtype_name(b->dtype_)), "matmul");
    if (a->device_ != b->device_)
        throw DeviceMismatch(std::string(device_name(a->device_)),
                             std::string(device_name(b->device_)), "matmul");
    if (a->shape_.size() < 2 || b->shape_.size() < 2) {
        throw ShapeMismatch(a->shape_, b->shape_,
                            "matmul: both operands must be ≥2-D");
    }
    // Item #8 — non-contiguous input guard. CPU only (GPU stride is internal).
    if (a->device_ == Device::CPU &&
        (!a->is_contiguous() || !b->is_contiguous())) {
        throw NotImplementedError(
            "matmul: non-contiguous input not supported "
            "(call .contiguous() first)");
    }

    const auto info = plan_nd_matmul(a->shape_, b->shape_);
    const int M = info.M, N = info.N, K = info.K;

    OpScope scope{MatmulBackward::schema_v1.name, a->device_, a->dtype_,
                  info.out_shape};
    scope.set_flops(static_cast<std::int64_t>(2) * info.batch * M * N * K);

    Storage out_storage;
    if (a->device_ == Device::GPU) {
        const auto& gA = std::get<GpuStorage>(a->storage_);
        const auto& gB = std::get<GpuStorage>(b->storage_);
        // MLX matmul handles batched N-D inputs natively (with broadcasting
        // over leading axes). No manual reshape needed.
        auto out = ::mlx::core::matmul(*gA.arr, *gB.arr);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)};
    } else {
        // CPU: materialize broadcast for batch dims, then per-slice GEMM.
        const CpuStorage& aRaw = std::get<CpuStorage>(a->storage_);
        const CpuStorage& bRaw = std::get<CpuStorage>(b->storage_);
        CpuStorage aBuf, bBuf;
        const CpuStorage* aUse = &aRaw;
        const CpuStorage* bUse = &bRaw;
        if (a->shape_ != info.a_bcast_shape) {
            aBuf = detail::broadcast_cpu(aRaw, a->shape_, info.a_bcast_shape, a->dtype_);
            aUse = &aBuf;
        }
        if (b->shape_ != info.b_bcast_shape) {
            bBuf = detail::broadcast_cpu(bRaw, b->shape_, info.b_bcast_shape, a->dtype_);
            bUse = &bBuf;
        }
        out_storage = Storage{cpu_matmul_nd(*aUse, *bUse, info,
                                              /*transA=*/false,
                                              /*transB=*/false, a->dtype_)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            info.out_shape,
                                            a->dtype_, a->device_,
                                            /*requires_grad=*/false);

    const bool needs_grad = GradMode::is_enabled() &&
                            (a->requires_grad_ || b->requires_grad_);
    if (!needs_grad) return out;

    auto a_edge = detail::ensure_grad_fn(a);
    auto b_edge = detail::ensure_grad_fn(b);

    auto bwd = std::make_shared<MatmulBackward>();
    bwd->input_shapes_ = {a->shape_, b->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a, b};  // Item #9 — for version check
    bwd->saved_inputs_ = {a->storage_, b->storage_};

    std::vector<Edge> edges;
    edges.emplace_back(a_edge, /*input_nr=*/0);
    edges.emplace_back(b_edge, /*input_nr=*/0);
    bwd->set_next_edges(std::move(edges));
    bwd->set_saved_versions({a->version_, b->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

TensorImplPtr matmul_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MatmulBackward::forward(a, b);
}

LUCID_REGISTER_OP(MatmulBackward)

}  // namespace lucid
