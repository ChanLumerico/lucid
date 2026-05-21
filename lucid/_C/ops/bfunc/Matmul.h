// lucid/_C/ops/bfunc/Matmul.h
//
// Declares MatmulBackward, the autograd node for batched matrix multiplication,
// and the public free function matmul_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the generalised matrix multiplication (matmul) with
// NumPy broadcasting on leading batch dimensions.
//
// Saves both inputs.  Forward dispatches to ``cblas_*gemm`` (Accelerate)
// for plain 2-D inputs, the batched-GEMM kernel for any leading-batch
// shape, and MLX's ``matmul`` primitive on the GPU stream.  Backward
// computes the two factor gradients via two further matmuls, then
// ``reduce_grad_to_shape``s each back to the original (pre-broadcast)
// batch shape.
//
// Math
// ----
// 2-D case ($A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$):
// $$
//   C = A B, \qquad
//   \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\, B^\top,
//   \qquad
//   \frac{\partial L}{\partial B} = A^\top\, \frac{\partial L}{\partial C}
// $$
// Batched case ($A,B$ rank $\ge 2$ with broadcastable leading dims):
// $$
//   C[\ldots, i, j] = \sum_k A[\ldots, i, k]\, B[\ldots, k, j]
// $$
//
// Shape
// -----
// - Both inputs must be at least 2-D; ``plan_nd_matmul`` rejects rank < 2.
// - Trailing two dims drive the GEMM: ``a.shape[-1] == b.shape[-2] == K``.
// - Leading dims broadcast NumPy-style; output shape is
//   ``broadcast(a.shape[:-2], b.shape[:-2]) + (M, N)``.
//
// Notes
// -----
// MatmulBackward is a ``FuncOp<..., 2>`` (alias for
// ``AutogradNode<..., 2>``) rather than a ``BinaryOp`` because matmul
// requires a custom :func:`forward` that cannot fit into the generic
// ``BinaryKernel`` dispatch path: it calls ``plan_nd_matmul``, handles
// batch broadcasting, and wires the autograd graph manually.
//
// Under ``AutocastGuard(F16)`` the schema's ``AmpPolicy::Promote`` resolves
// to the autocast target (F32 on CPU per the Accelerate-no-F16 carve-out;
// F16 on GPU where MLX runs matmul natively at half throughput).  The
// AMP cast is wired through autograd-aware ``astype_op`` so the backward
// chain stays intact.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``("matmul", v1, AmpPolicy::Promote, deterministic=true)``.
//
// See Also
// --------
// dot_op : 1-D × 1-D / 2-D × 2-D specialised dispatch without broadcasting.
// inner_op : Last-axis contraction (NumPy ``inner``).
// outer_op : 1-D × 1-D outer product.
//
// References
// ----------
// NumPy ``numpy.matmul`` semantics (which Lucid mirrors).
class LUCID_API MatmulBackward : public FuncOp<MatmulBackward, 2> {
public:
    // Op registration metadata for ``MatmulBackward``.
    //
    // Attributes
    // ----------
    // schema_v1 : OpSchema
    //     ``"matmul"``, version 1, ``AmpPolicy::Promote``, deterministic.
    static const OpSchema schema_v1;

    // Validate inputs, plan the batched matrix multiply, execute the
    // forward kernel, and wire the autograd graph if gradient tracking
    // is active.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Left operand, rank $\ge 2$.  Trailing two dims are $(M, K)$.
    // b : TensorImplPtr
    //     Right operand, rank $\ge 2$.  Trailing two dims are $(K, N)$.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor with shape
    //     ``broadcast(a.shape[:-2], b.shape[:-2]) + (M, N)``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If either input has rank < 2 or the leading batch dims fail
    //     NumPy broadcasting.
    // DeviceMismatch
    //     If ``a`` and ``b`` live on different devices.
    // DtypeMismatch
    //     If the AMP-promoted dtypes of ``a`` and ``b`` still disagree.
    static TensorImplPtr forward(const TensorImplPtr& a, const TensorImplPtr& b);

    // Eager-mode backward: compute $\partial L/\partial A$ and
    // $\partial L/\partial B$ from the upstream gradient and reduce any
    // batch dims that were broadcast-expanded during forward.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient $\partial L/\partial C$ with the output's
    //     broadcasted batch shape and trailing $(M, N)$.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     ``{dA, dB}`` reduced to the original input shapes.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward: build a symbolic backward subgraph
    // $\partial L/\partial A = \partial L/\partial C \cdot B^\top$ and
    // $\partial L/\partial B = A^\top \cdot \partial L/\partial C$ using
    // ``matmul_op``, ``mT_op``, and ``sum_op``.
    //
    // Parameters
    // ----------
    // grad_out : TensorImplPtr
    //     Symbolic upstream gradient.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     ``{dA, dB}`` as graph tensors reduced to the original input
    //     shapes.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

    // Human-readable node label used by the autograd profiler.
    std::string node_name() const override { return "matmul"; }
};

// Compute $A B$ with full NumPy-style batch broadcasting and autograd.
//
// Thin public wrapper that delegates to :func:`MatmulBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand, rank $\ge 2$.
// b : TensorImplPtr
//     Right operand, rank $\ge 2$.
//
// Returns
// -------
// TensorImplPtr
//     Result tensor $C = A B$ with broadcast batch shape and trailing
//     $(M, N)$.
//
// Math
// ----
// $$
//   C[\ldots, i, j] = \sum_{k=0}^{K-1} A[\ldots, i, k]\, B[\ldots, k, j]
// $$
//
// Notes
// -----
// FLOPs per output element ≈ $2K$; total $\approx 2 \cdot B \cdot M \cdot
// N \cdot K$ where $B$ is the broadcast batch size.
//
// Examples
// --------
// ::
//
//     // C++: 2-D × 2-D plain GEMM.
//     auto C = matmul_op(A, B);  // A: [M,K], B: [K,N]  → C: [M,N]
//
// See Also
// --------
// dot_op : Rank-1 / rank-2 specialised dispatch.
// MatmulBackward : Autograd node implementing the backward pass.
LUCID_API TensorImplPtr matmul_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
