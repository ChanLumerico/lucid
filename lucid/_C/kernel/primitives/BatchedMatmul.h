#pragma once

// =====================================================================
// Lucid C++ engine — kernel/primitives/BatchedMatmul.h
// =====================================================================
//
// Primitive: N-D batched matrix multiply.
//
//   a [..., M, K] @ b [..., K, N] → out [..., M, N]
//
// Leading (batch) dimensions are broadcast-aligned via NumPy semantics.
// Each batch slice is dispatched to backend::cpu::sgemm/dgemm (CPU) or
// mlx::core::matmul (GPU).
//
// Used by: Matmul, Linear backward, Attention, BilinearLayer, MatrixPower.
//
// Layer: kernel/primitives/. Depends on backend/cpu/ (CPU path), core/.

#include <cstddef>
#include <vector>

#include "../../backend/cpu/Blas.h"
#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../BinaryKernel.h"  // detail::try_broadcast_shapes

namespace lucid {
namespace kernel {
namespace primitives {

/// Planning result for an N-D batched matmul.
struct NdMatmulInfo {
    Shape out_shape;      ///< full output shape [..., M, N]
    Shape a_bcast_shape;  ///< a broadcast shape [..., M, K]
    Shape b_bcast_shape;  ///< b broadcast shape [..., K, N]
    int M = 0, K = 0, N = 0;
    std::size_t batch = 1;  ///< product of all leading dims
};

/// Compute the output/broadcast shapes for a @ b.
/// Throws ShapeMismatch on incompatible shapes.
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

/// CPU N-D batched matmul.
/// `a` and `b` must already be broadcast-materialized to info.a_bcast_shape /
/// info.b_bcast_shape respectively (use detail::broadcast_cpu if needed).
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
