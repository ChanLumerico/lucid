// lucid/_C/ops/bfunc/Outer.h
//
// Declares outer_op, the entry point for the outer (tensor) product of two
// 1-D vectors.  The outer product produces a 2-D matrix:
//   C[i, j] = a[i] * b[j]

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the outer (tensor) product of two 1-D vectors, yielding a 2-D
// matrix.
//
// The implementation evaluates the outer product as a degenerate GEMM
// between reshaped operands: ``a[:, None] @ b[None, :]`` — i.e. an
// $(M, 1) \times (1, N)$ matmul producing an $(M, N)$ result.  On
// Accelerate this maps onto ``cblas_sger`` (rank-1 update) or
// ``cblas_*gemm``; on the GPU stream it routes through MLX's matmul
// primitive.  Autograd is wired through the package-private
// ``OuterBackward`` node, which saves both operands.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand.  Must be 1-D with length $M$.
// b : TensorImplPtr
//     Right operand.  Must be 1-D with length $N$.
//
// Returns
// -------
// TensorImplPtr
//     Output matrix of shape $(M, N)$ with
//     $\mathrm{out}[i, j] = a_i\, b_j$.
//
// Math
// ----
// Forward:
// $$
//   C_{ij} = a_i\, b_j
// $$
// Backward (with $G = \partial L/\partial C$):
// $$
//   \frac{\partial L}{\partial a_i} = \sum_j G_{ij}\, b_j
//     = (G\, b)_i, \qquad
//   \frac{\partial L}{\partial b_j} = \sum_i G_{ij}\, a_i
//     = (a^\top G)_j
// $$
//
// Shape
// -----
// - ``a.shape == {M}`` and ``b.shape == {N}``.
// - Output shape ``{M, N}``.
//
// Raises
// ------
// ShapeMismatch
//     If either input has rank other than 1.
//
// Notes
// -----
// Schema uses ``AmpPolicy::KeepInput`` — outer products on integer
// vectors are valid and must not be promoted.
//
// Examples
// --------
// ::
//
//     // a: [3],  b: [4]  →  out: [3, 4]
//     // out[i, j] = a[i] * b[j]
//
// See Also
// --------
// dot_op : 1-D × 1-D inner product (the "dual" of outer).
// inner_op : Last-axis contraction.
// matmul_op : Generalised matmul with batch broadcasting.
//
// References
// ----------
// NumPy ``numpy.outer`` (which Lucid mirrors for the 1-D case).
LUCID_API TensorImplPtr outer_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
