// lucid/_C/ops/bfunc/Dot.h
//
// Declares dot_op, the entry point for the 1-D and 2-D dot-product operation.
// The implementation handles two strictly separate cases:
//   - 1-D × 1-D: inner product yielding a scalar.
//   - 2-D × 2-D: standard matrix multiply equivalent to matmul for rank-2
//     tensors.
// Higher-rank inputs are rejected at runtime.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the dot product of two vectors or the matrix product of two
// matrices, with autograd support for both cases.
//
// Dispatches on rank:
//
//   - **1-D × 1-D** — inner product
//     $c = \sum_i a_i b_i$ producing a scalar (shape ``{}``).
//     Backed by Accelerate's ``cblas_*dot`` via the element-wise multiply
//     and ``reduce_sum`` backend primitives.
//     Backward node: ``Dot1DBackward`` (saves both operands).
//   - **2-D × 2-D** — matrix product $C = A B$.
//     Backed by ``cblas_*gemm`` (Accelerate) on CPU, ``mlx::matmul`` on GPU.
//     Backward node: ``Dot2DBackward`` (saves both operands).
//
// Parameters
// ----------
// a : TensorImplPtr
//     First operand.  Must be 1-D or 2-D.
// b : TensorImplPtr
//     Second operand.  Must have the same rank as ``a``.
//
// Returns
// -------
// TensorImplPtr
//     - 1-D inputs → scalar tensor with shape ``{}``.
//     - 2-D inputs of shape $(M, K)$ and $(K, N)$ → matrix of shape $(M, N)$.
//
// Math
// ----
// 1-D case:
// $$
//   c = \sum_{i=0}^{N-1} a_i b_i, \qquad
//   \frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial c}\, b_i,
//   \qquad
//   \frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial c}\, a_i
// $$
// 2-D case:
// $$
//   C = A B, \qquad
//   \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\, B^\top,
//   \qquad
//   \frac{\partial L}{\partial B} = A^\top\, \frac{\partial L}{\partial C}
// $$
//
// Shape
// -----
// - 1-D: ``a.shape[0] == b.shape[0]``; output is rank-0.
// - 2-D: ``a.shape[1] == b.shape[0]``; output is $(M, N)$.
//
// Raises
// ------
// ShapeMismatch
//     If the contracted dimensions disagree (1-D length mismatch, or 2-D
//     ``a.shape[1] != b.shape[0]``).
// NotImplemented
//     If either input has rank other than 1 or 2.
//
// Notes
// -----
// Unlike :func:`matmul_op` this entry point uses ``AmpPolicy::KeepInput``
// — integer dot products are valid and must not be promoted silently.
//
// See Also
// --------
// matmul_op : N-D matmul with NumPy broadcasting on batch dims.
// inner_op : Last-axis contraction (NumPy ``inner``).
// outer_op : 1-D × 1-D outer product.
//
// References
// ----------
// NumPy ``numpy.dot`` semantics for the 1-D and 2-D cases.
LUCID_API TensorImplPtr dot_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
