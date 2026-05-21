// lucid/_C/ops/bfunc/Inner.h
//
// Declares inner_op, the entry point for the generalised inner product.
// The inner product contracts the last axis of A against the last axis of B:
//   out[i₀,…,iₙ₋₂, j₀,…,jₘ₋₂] = Σ_k A[i₀,…,iₙ₋₂, k] * B[j₀,…,jₘ₋₂, k]
// This matches the semantics of numpy.inner.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the generalised inner product of $A$ and $B$ by contracting
// their last axes (NumPy ``inner`` semantics).
//
// For rank-$n_A$ and rank-$n_B$ inputs whose last dims agree in size, the
// output preserves all leading-of-A dims followed by all leading-of-B
// dims, with the shared last axis summed out.
//
// Parameters
// ----------
// a : TensorImplPtr
//     First operand, rank $\ge 1$.  Last dim is the contraction axis.
// b : TensorImplPtr
//     Second operand, rank $\ge 1$.  Last dim must match ``a.shape[-1]``.
//
// Returns
// -------
// TensorImplPtr
//     Output with shape ``a.shape[:-1] + b.shape[:-1]``.  When both
//     inputs are 1-D the result is a scalar (shape ``{}``).
//
// Math
// ----
// General contraction over the trailing axis ($K = a_{n_A-1} = b_{n_B-1}$):
// $$
//   \mathrm{out}[i_0, \ldots, i_{n_A-2}, j_0, \ldots, j_{n_B-2}]
//     = \sum_{k=0}^{K-1}
//         A[i_0, \ldots, i_{n_A-2}, k]\,
//         B[j_0, \ldots, j_{n_B-2}, k]
// $$
// 2-D × 2-D specialisation:
// $$
//   \mathrm{out}[i, j] = \sum_k A[i, k]\, B[j, k]
// $$
//
// Shape
// -----
// - ``a.shape[-1] == b.shape[-1]`` is required.
// - Output shape concatenates the all-but-last dims of each input.
//
// Notes
// -----
// When gradient tracking is enabled this op is lowered to ``einsum_op``
// with a pattern produced by ``inner_einsum_pattern`` (free axes of $A$
// use letters ``a..o``, free axes of $B$ use ``p..y``, contracted axis
// is ``z``).  The einsum backward pass handles autograd, so this op has
// no dedicated backward node.
//
// On the GPU stream the output shape is read back from the MLX array
// after the kernel returns to absorb any scalar-squeeze MLX may perform.
//
// Raises
// ------
// ShapeMismatch
//     If either input is rank-0 or the last dims disagree.
//
// Examples
// --------
// ::
//
//     // a: [3, 5],  b: [2, 5]  →  out: [3, 2]
//     // out[i, j] = sum_k a[i, k] * b[j, k]
//
// See Also
// --------
// dot_op : 1-D × 1-D / 2-D × 2-D specialised dispatch.
// matmul_op : Standard matmul with NumPy batch broadcasting.
// outer_op : 1-D outer product (the rank-1 "anti-inner").
// einsum_op : The lowering target for the gradient-tracked path.
//
// References
// ----------
// NumPy ``numpy.inner`` (which Lucid mirrors).
LUCID_API TensorImplPtr inner_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
