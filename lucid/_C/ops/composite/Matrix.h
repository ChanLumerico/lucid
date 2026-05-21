// lucid/_C/ops/composite/Matrix.h
//
// Matrix-product compositions: rank-checked aliases of ``matmul`` plus the
// Kronecker product.
//
//   mm(a, b)   — strict 2-D matmul
//   bmm(a, b)  — strict 3-D batched matmul (batch dims must agree)
//   kron(a, b) — block tensor product over matching ranks
//
// All entry points compose existing differentiable ops; backward flows
// through ``MatmulBackward``, ``MulBackward``, and ``ReshapeBackward``.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Strict 2-D matrix multiply: $C = A \cdot B$.
//
// Composite — rank-checked alias of :func:`matmul_op` that rejects inputs
// of rank $\ne 2$ up front so the error message names ``mm`` rather than
// the planner.  Gradient flows through ``MatmulBackward``.
//
// Math
// ----
// $$
//   C_{ij} = \sum_{k} A_{ik} \, B_{kj}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand of shape ``(M, K)``.
// b : TensorImplPtr
//     Right operand of shape ``(K, N)``.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``(M, N)``.
//
// Raises
// ------
// Failure
//     If either input is null or not exactly 2-D.
//
// See Also
// --------
// :func:`matmul_op` — generalised version with broadcasting and 1-D
// vector promotion; :func:`bmm_op` — strict batched variant.
LUCID_API TensorImplPtr mm_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Strict 3-D batched matrix multiply: $C_b = A_b \cdot B_b$.
//
// Composite — rank-checked alias of :func:`matmul_op`.  Rejects inputs of
// rank $\ne 3$ and demands matching leading batch dimensions so the error
// path is crisp.  Gradient flows through ``MatmulBackward``.
//
// Math
// ----
// For every batch index $b \in [0, B)$:
// $$
//   C_{b, i, j} = \sum_{k} A_{b, i, k} \, B_{b, k, j}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand of shape ``(B, M, K)``.
// b : TensorImplPtr
//     Right operand of shape ``(B, K, N)``.  Leading batch ``B`` must
//     match ``a``.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``(B, M, N)``.
//
// Raises
// ------
// Failure
//     If either input is null, not exactly 3-D, or batch dimensions
//     disagree.
//
// See Also
// --------
// :func:`mm_op`, :func:`matmul_op`.
LUCID_API TensorImplPtr bmm_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Kronecker (block tensor) product over matching ranks.
//
// Composite over :func:`reshape_op` + :func:`mul_op` + :func:`reshape_op`.
// Interleaves size-1 placeholders so the multiply broadcasts to a
// $2 \cdot \mathrm{ndim}$-rank block tensor; the trailing reshape
// collapses each ``(sa[i], sb[i])`` pair into a single ``sa[i] * sb[i]``
// axis.  Gradient flows through ``ReshapeBackward`` and ``MulBackward``.
//
// Math
// ----
// For rank-2 operands $A \in \mathbb{R}^{m \times n}$ and
// $B \in \mathbb{R}^{p \times q}$:
// $$
//   (A \otimes B)_{ip + r,\, jq + s} = A_{ij} \, B_{rs}
// $$
// generalised elementwise to arbitrary equal ranks.
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Operands sharing the same rank ``ndim``.  Shapes need not match
//     dimension-by-dimension.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``(sa[0] * sb[0], ..., sa[-1] * sb[-1])``.
//
// Raises
// ------
// Failure
//     If either input is null or the ranks differ.
//
// Examples
// --------
// ``kron`` of two ``(2, 3)`` matrices yields a ``(4, 9)`` matrix.
//
// See Also
// --------
// :func:`mul_op`, :func:`reshape_op`.
LUCID_API TensorImplPtr kron_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
