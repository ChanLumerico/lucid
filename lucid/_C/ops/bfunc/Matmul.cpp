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

}  // namespace

std::vector<Storage> MatmulBackward::apply(Storage grad_out) {
    // Saved: input_shapes_[0] = [M, K], input_shapes_[1] = [K, N]
    // grad_out: [M, N], saved_inputs_[0] = A, saved_inputs_[1] = B.
    const int M = static_cast<int>(input_shapes_[0][0]);
    const int K = static_cast<int>(input_shapes_[0][1]);
    const int N = static_cast<int>(input_shapes_[1][1]);

    if (device_ == Device::GPU) {
        const auto& gA = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gB = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gG = std::get<GpuStorage>(grad_out);
        if (!gA.arr || !gB.arr || !gG.arr) {
            throw LucidError("matmul backward: null GPU array");
        }
        // dA = grad @ B^T ; dB = A^T @ grad. MLX has no separate transpose
        // flag on matmul; use ::mlx::core::transpose explicitly. Empty cases
        // (M/N/K == 0) fall through naturally — MLX returns empty arrays.
        if (M == 0 || N == 0 || K == 0) {
            auto za = ::mlx::core::zeros(gpu::to_mlx_shape(input_shapes_[0]),
                                         gpu::to_mlx_dtype(dtype_));
            auto zb = ::mlx::core::zeros(gpu::to_mlx_shape(input_shapes_[1]),
                                         gpu::to_mlx_dtype(dtype_));
            return {Storage{gpu::wrap_mlx_array(std::move(za), dtype_)},
                    Storage{gpu::wrap_mlx_array(std::move(zb), dtype_)}};
        }
        auto bT = ::mlx::core::transpose(*gB.arr);
        auto aT = ::mlx::core::transpose(*gA.arr);
        auto dA = ::mlx::core::matmul(*gG.arr, bT);
        auto dB = ::mlx::core::matmul(aT, *gG.arr);
        return {Storage{gpu::wrap_mlx_array(std::move(dA), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dB), dtype_)}};
    }

    const auto& g_cpu = std::get<CpuStorage>(grad_out);
    const auto& a_cpu = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& b_cpu = std::get<CpuStorage>(saved_inputs_[1]);

    // Item #7 — empty case: cblas rejects 0-sized BLAS calls. Allocate
    // zero-filled grads of the right shape instead.
    auto empty_grad = [this](int rows, int cols) -> Storage {
        auto cpu = allocate_2d(rows, cols, this->dtype_);
        if (cpu.nbytes > 0) std::memset(cpu.ptr.get(), 0, cpu.nbytes);
        return Storage{std::move(cpu)};
    };

    Storage dA = (M == 0 || K == 0 || N == 0)
        ? empty_grad(M, K)
        : Storage{cpu_matmul(g_cpu, b_cpu, M, K, N,
                             /*transA=*/false, /*transB=*/true, dtype_)};
    Storage dB = (K == 0 || N == 0 || M == 0)
        ? empty_grad(K, N)
        : Storage{cpu_matmul(a_cpu, g_cpu, K, N, M,
                             /*transA=*/true, /*transB=*/false, dtype_)};
    return {std::move(dA), std::move(dB)};
}

TensorImplPtr MatmulBackward::forward(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b) throw LucidError("matmul: null input");
    if (a->dtype_ != b->dtype_)
        throw DtypeMismatch(std::string(dtype_name(a->dtype_)),
                            std::string(dtype_name(b->dtype_)), "matmul");
    if (a->device_ != b->device_)
        throw DeviceMismatch(std::string(device_name(a->device_)),
                             std::string(device_name(b->device_)), "matmul");
    if (a->shape_.size() != 2 || b->shape_.size() != 2) {
        throw ShapeMismatch(a->shape_, b->shape_,
                            "matmul (Phase 3.1: 2-D only)");
    }
    if (a->shape_[1] != b->shape_[0]) {
        throw ShapeMismatch(a->shape_, b->shape_, "matmul (inner dim mismatch)");
    }
    // Item #8 — non-contiguous input guard. CPU only (GPU stride is internal).
    if (a->device_ == Device::CPU &&
        (!a->is_contiguous() || !b->is_contiguous())) {
        throw NotImplementedError(
            "matmul: non-contiguous input not supported "
            "(call .contiguous() first)");
    }

    const int M = static_cast<int>(a->shape_[0]);
    const int K = static_cast<int>(a->shape_[1]);
    const int N = static_cast<int>(b->shape_[1]);

    OpScope scope{MatmulBackward::schema_v1.name, a->device_, a->dtype_,
                  Shape{a->shape_[0], b->shape_[1]}};
    scope.set_flops(static_cast<std::int64_t>(2) * M * N * K);  // 2*M*N*K standard

    Storage out_storage;
    if (a->device_ == Device::GPU) {
        if (M == 0 || N == 0 || K == 0) {
            auto z = ::mlx::core::zeros(
                gpu::to_mlx_shape(Shape{a->shape_[0], b->shape_[1]}),
                gpu::to_mlx_dtype(a->dtype_));
            out_storage = Storage{gpu::wrap_mlx_array(std::move(z), a->dtype_)};
        } else {
            const auto& gA = std::get<GpuStorage>(a->storage_);
            const auto& gB = std::get<GpuStorage>(b->storage_);
            auto out = ::mlx::core::matmul(*gA.arr, *gB.arr);
            out_storage =
                Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)};
        }
    } else {
        // Item #7 — empty inner dim: cblas_sgemm rejects K=0 with
        // "lda >= max(K,1)". Empty matmul over an empty inner dim produces an
        // all-zeros output (empty sum convention).
        CpuStorage out_storage_cpu;
        if (M == 0 || N == 0 || K == 0) {
            out_storage_cpu = allocate_2d(M, N, a->dtype_);
            if (out_storage_cpu.nbytes > 0) {
                std::memset(out_storage_cpu.ptr.get(), 0, out_storage_cpu.nbytes);
            }
        } else {
            out_storage_cpu = cpu_matmul(std::get<CpuStorage>(a->storage_),
                                         std::get<CpuStorage>(b->storage_),
                                         M, N, K, /*transA=*/false,
                                         /*transB=*/false, a->dtype_);
        }
        out_storage = Storage{std::move(out_storage_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            Shape{a->shape_[0], b->shape_[1]},
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
