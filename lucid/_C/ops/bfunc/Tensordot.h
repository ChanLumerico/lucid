// lucid/_C/ops/bfunc/Tensordot.h
//
// Declares tensordot_op, the entry point for tensor contraction over specified
// axis pairs.  Semantics match numpy.tensordot: axes_a[i] and axes_b[i] name
// paired axes that are summed out.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// General multi-axis tensor contraction (NumPy ``tensordot``).
//
// Sums the product of ``a`` and ``b`` over the axes named pairwise by
// ``axes_a`` and ``axes_b``, leaving every other axis as an outer-product
// dimension in the output.  This is the multi-axis generalisation of
// matrix multiplication: ``tensordot(a, b, [k], [0])`` over rank-2 inputs
// is exactly ``a @ b``.
//
// When gradient tracking is active the contraction is lowered to an
// equivalent ``einsum`` string and the einsum backward node handles the
// gradient computation, so this op itself does not register a dedicated
// backward node.  On the CPU inference path the call is fused into a
// permute-reshape-then-scalar-GEMM kernel; on GPU it dispatches to the
// MLX ``tensordot`` primitive.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand.  Any rank, any floating dtype.  ``a->dtype()`` defines
//     the result dtype.
// b : TensorImplPtr
//     Right operand.  Must share dtype and device with ``a``.
// axes_a : std::vector<int>
//     Axis indices of ``a`` to contract over.  Negative indices count
//     from the right and are normalised to ``[0, a.ndim)``.
// axes_b : std::vector<int>
//     Axis indices of ``b`` to contract over, paired element-wise with
//     ``axes_a``.  Must have the same length as ``axes_a``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``a.shape[free_a] ++ b.shape[free_b]`` where
//     ``free_*`` denotes the axes of each operand that are NOT contracted.
//     The output ``ndim`` equals ``a.ndim + b.ndim - 2 * len(axes_a)``.
//
// Math
// ----
// With ``axes = (axes_a, axes_b)`` and ``a``, ``b`` reshaped into
// matrices $A' \in \mathbb{R}^{M \times K}$ and $B' \in \mathbb{R}^{K \times N}$
// (free dims flattened to $M$ and $N$, contracted dims flattened to $K$),
// $$
//   C' = A' B', \qquad C = \mathrm{reshape}(C', \text{free}_a + \text{free}_b).
// $$
// Gradients flow back as two ``matmul``-equivalent contractions:
// $$
//   \frac{\partial L}{\partial A'} = \frac{\partial L}{\partial C'} {B'}^{\top}, \qquad
//   \frac{\partial L}{\partial B'} = {A'}^{\top} \frac{\partial L}{\partial C'}.
// $$
//
// Shape
// -----
// Let ``N = len(axes_a) = len(axes_b)``.  Then
// $$
//   \mathrm{ndim}(\text{out}) = \mathrm{ndim}(a) + \mathrm{ndim}(b) - 2N.
// $$
// The contracted-dim sizes must match pairwise:
// ``a.shape[axes_a[i]] == b.shape[axes_b[i]]`` for all ``i``.
//
// Notes
// -----
// * ``axes_a`` and ``axes_b`` are passed by value so the implementation
//   may normalise negative indices in place.
// * Inputs need not be contiguous; the CPU path materialises a permuted
//   contiguous copy before the GEMM step.
// * Saves both inputs and the axes specification (under the einsum
//   lowering) for backward.
//
// Examples
// --------
// Contract the trailing axis of ``a`` with the leading axis of ``b``
// (a rank-3 by rank-3 contraction equivalent to ``a @ b`` over the last
// two axes when both are 2-D):
//
//     auto a = randn({4, 5, 6});  // shape (4, 5, 6)
//     auto b = randn({6, 7, 8});  // shape (6, 7, 8)
//     auto c = tensordot_op(a, b, {2}, {0});  // shape (4, 5, 7, 8)
//
// Two-axis contraction (full inner product over the trailing pair):
//
//     auto a = randn({3, 4, 5});           // shape (3, 4, 5)
//     auto b = randn({4, 5, 7});           // shape (4, 5, 7)
//     auto c = tensordot_op(a, b, {1, 2}, {0, 1});  // shape (3, 7)
//
// Attributes
// ----------
// schema : OpScopeFull
//     ``"tensordot"``, ``AmpPolicy::Promote``.  Lowers to ``einsum`` when
//     autograd is active.
//
// Raises
// ------
// LucidError
//     If ``axes_a.size() != axes_b.size()``, if any axis is out of range
//     after normalisation, if a paired pair of axes has mismatched sizes,
//     or if the CPU dtype is neither ``F32`` nor ``F64``.
//
// References
// ----------
// NumPy ``numpy.tensordot`` semantics.
//
// See Also
// --------
// einsum_op, matmul_op
LUCID_API TensorImplPtr tensordot_op(const TensorImplPtr& a,
                                     const TensorImplPtr& b,
                                     std::vector<int> axes_a,
                                     std::vector<int> axes_b);

}  // namespace lucid
