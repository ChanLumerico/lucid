// lucid/_C/kernel/primitives/BatchedMatmul.h
//
// CPU batched GEMM helper for N-dimensional matrix multiplication.
// plan_nd_matmul() resolves the broadcast batch dimensions and the M/K/N
// inner dimensions from two ≥2-D shapes. cpu_matmul_nd() then executes
// the batched GEMM by calling sgemm/dgemm once per batch slice via the
// Apple Accelerate BLAS wrappers in backend/cpu/Blas.h.
//
// Both functions are used by the MatMul op's forward and backward passes.
// The backward pass calls cpu_matmul_nd with transA or transB = true to
// compute dA = dOut @ B^T and dB = A^T @ dOut.

#pragma once

#include <cstddef>
#include <vector>

#include "../../backend/cpu/Blas.h"
#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../BinaryKernel.h"

namespace lucid {
namespace kernel {
namespace primitives {

// Precomputed plan for an N-dimensional batched matrix multiplication.
//
// Resolves both operand shapes into a single triplet of broadcast batch
// dimensions plus the inner $(M, K, N)$ contraction tuple, so that the
// downstream GEMM loop can iterate a flat batch counter without
// re-inspecting per-axis shape information.
//
// Attributes
// ----------
// out_shape : Shape
//     Shape of the matmul output: broadcast batch dims concatenated with
//     ``[M, N]``.
// a_bcast_shape : Shape
//     Shape that operand $A$ is conceptually broadcast to before the
//     GEMM loop — batch dims concatenated with ``[M, K]``.
// b_bcast_shape : Shape
//     Shape that operand $B$ is conceptually broadcast to — batch dims
//     concatenated with ``[K, N]``.
// M : int
//     Row count of each per-batch left matrix.
// K : int
//     Shared inner (contraction) dimension.
// N : int
//     Column count of each per-batch right matrix.
// batch : std::size_t
//     Product of all broadcast batch dimensions; the number of 2-D GEMM
//     calls dispatched in the inner loop.
//
// See Also
// --------
// plan_nd_matmul : Builds an ``NdMatmulInfo`` from two operand shapes.
// cpu_matmul_nd  : Consumes an ``NdMatmulInfo`` and runs the GEMM loop.
struct NdMatmulInfo {
    Shape out_shape;          // Shape of the output tensor.
    Shape a_bcast_shape;      // Shape of operand a after batch broadcast.
    Shape b_bcast_shape;      // Shape of operand b after batch broadcast.
    int M = 0, K = 0, N = 0;  // Inner matrix dimensions.
    std::size_t batch = 1;    // Product of all broadcast batch dimensions.
};

// Plan an N-D matmul from two operand shapes.
//
// Validates that both operands are at least 2-D, that the inner
// contraction dimensions agree, and that the leading batch dimensions
// broadcast against each other under NumPy rules.  Returns a fully
// populated :struct:`NdMatmulInfo` describing the GEMM loop bounds.
//
// Math
// ----
// Splits each operand into a leading batch shape and a trailing matrix
// shape, then broadcasts the batch shapes:
// $$
//   A \in \mathbb{R}^{B_a \times M \times K}, \quad
//   B \in \mathbb{R}^{B_b \times K \times N}, \quad
//   B_\text{out} = \mathrm{broadcast}(B_a, B_b)
// $$
// so that the output has shape ``out_shape = B_out + [M, N]``.
//
// Parameters
// ----------
// a : const Shape&
//     Shape of the left operand; must satisfy ``a.size() >= 2`` with
//     trailing dims ``[..., M, K]``.
// b : const Shape&
//     Shape of the right operand; must satisfy ``b.size() >= 2`` with
//     trailing dims ``[..., K, N]``.
//
// Returns
// -------
// NdMatmulInfo
//     Plan describing the output shape, per-operand broadcast shapes,
//     contraction dims, and total batch count.
//
// Raises
// ------
// ShapeMismatch
//     If either operand has rank below 2, the inner contraction
//     dimensions disagree, or the leading batch dimensions cannot be
//     broadcast under NumPy rules (a leading dim of 1 expands to match
//     the other operand's value).
//
// Notes
// -----
// Either operand may have an empty batch shape — in that case the other
// operand's batch dims become the output's batch dims unchanged.
inline NdMatmulInfo plan_nd_matmul(const Shape& a, const Shape& b) {
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
    if (ba.empty()) {
        out_b = bb;
    } else if (bb.empty()) {
        out_b = ba;
    } else {
        auto r = detail::try_broadcast_shapes(ba, bb);
        if (r.is_err())
            throw ShapeMismatch(a, b, "matmul: incompatible batch dims");
        out_b = std::move(r).value();
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
    for (auto d : out_b)
        batch *= static_cast<std::size_t>(d);
    info.batch = batch;
    return info;
}

// Execute a batched N-D matrix multiplication on the CPU.
//
// Iterates over the flattened batch dimension described by ``info`` and
// dispatches one ``sgemm`` (or ``dgemm``) per batch slice via the
// Accelerate BLAS wrappers in ``backend/cpu/Blas.h``.  Optional transpose
// flags reuse the same kernel for the matmul backward pass:
// $dA = dOut \cdot B^T$ uses ``transB = true`` and
// $dB = A^T \cdot dOut$ uses ``transA = true``.
//
// Math
// ----
// For each batch index $\mathbf{b}$ in ``info.out_shape[:-2]``:
// $$
//   C[\mathbf{b}] = \mathrm{op}_A(A[\mathbf{b}]) \cdot \mathrm{op}_B(B[\mathbf{b}])
// $$
// where $\mathrm{op}_X(X) = X^T$ if the corresponding transpose flag is
// set, else $X$.
//
// Parameters
// ----------
// a : const CpuStorage&
//     Left operand; logical shape ``info.a_bcast_shape``.  Assumed to be
//     pre-broadcast and contiguous in row-major layout.
// b : const CpuStorage&
//     Right operand; logical shape ``info.b_bcast_shape``.  Assumed to
//     be pre-broadcast and contiguous in row-major layout.
// info : const NdMatmulInfo&
//     Plan produced by :func:`plan_nd_matmul`.
// transA : bool
//     If true, treat each $A$ slice as $K \times M$ (transposed) before
//     the GEMM call — adjusts the leading dimension accordingly.
// transB : bool
//     If true, treat each $B$ slice as $N \times K$ (transposed) before
//     the GEMM call.
// dt : Dtype
//     Element dtype; only :data:`Dtype::F32` and :data:`Dtype::F64` are
//     supported.
//
// Returns
// -------
// CpuStorage
//     Newly allocated, aligned output buffer of shape
//     ``info.out_shape`` (flattened, row-major).  All-zero contraction
//     dims ($M$, $N$, or $K$ equal to 0) yield a zero-filled output.
//
// Raises
// ------
// ErrorBuilder("matmul")::not_implemented
//     If ``dt`` is not one of :data:`Dtype::F32` or :data:`Dtype::F64`.
//
// Notes
// -----
// Leading dimensions are set based on the transpose flags: the
// untransposed $M \times K$ slice has ``lda = K``, while a transposed
// $K \times M$ view has ``lda = M``.  The output leading dimension is
// always $N$ (untransposed output).
//
// See Also
// --------
// backend::cpu::sgemm : Single-precision Accelerate GEMM wrapper.
// backend::cpu::dgemm : Double-precision Accelerate GEMM wrapper.
inline CpuStorage cpu_matmul_nd(const CpuStorage& a,
                                const CpuStorage& b,
                                const NdMatmulInfo& info,
                                bool transA,
                                bool transB,
                                Dtype dt) {
    const std::size_t batch = info.batch;
    const int M = info.M, K = info.K, N = info.N;
    const std::size_t a_step = static_cast<std::size_t>(M) * K;
    const std::size_t b_step = static_cast<std::size_t>(K) * N;
    const std::size_t o_step = static_cast<std::size_t>(M) * N;
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = batch * o_step * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    if (M == 0 || N == 0 || K == 0) {
        if (out.nbytes)
            std::memset(out.ptr.get(), 0, out.nbytes);
        return out;
    }
    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* ap = reinterpret_cast<const T*>(a.ptr.get());
        const T* bp = reinterpret_cast<const T*>(b.ptr.get());
        T* op = reinterpret_cast<T*>(out.ptr.get());
        // Leading dimensions depend on whether the matrix is transposed:
        // for a normal M×K matrix, lda = K; for a transposed K×M view, lda = M.
        const int lda = transA ? M : K;
        const int ldb = transB ? K : N;
        for (std::size_t bi = 0; bi < batch; ++bi) {
            if constexpr (std::is_same_v<T, float>) {
                backend::cpu::sgemm(transA, transB, M, N, K, 1.0f, ap + bi * a_step, lda,
                                    bp + bi * b_step, ldb, 0.0f, op + bi * o_step, N);
            } else {
                backend::cpu::dgemm(transA, transB, M, N, K, 1.0, ap + bi * a_step, lda,
                                    bp + bi * b_step, ldb, 0.0, op + bi * o_step, N);
            }
        }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("matmul").not_implemented("dtype not supported (F32/F64)");
    return out;
}

}  // namespace primitives
}  // namespace kernel
}  // namespace lucid
