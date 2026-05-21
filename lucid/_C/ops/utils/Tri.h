// lucid/_C/ops/utils/Tri.h
//
// Declares the lower-triangular (tril) and upper-triangular (triu) masking ops.
// Both ops zero out elements on the wrong side of a diagonal offset and are
// differentiable: the backward pass applies the same mask to the incoming
// gradient, preserving only the gradient contributions that correspond to
// elements that were retained in the forward pass.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Lower-triangular part of a matrix (or batch of matrices).
//
// Zeros every element strictly above the ``k``-th diagonal of the last
// two axes of ``a``.  The main diagonal is ``k = 0``; positive ``k``
// shifts the diagonal toward the upper-right (keeping more elements);
// negative ``k`` toward the lower-left (keeping fewer).
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of rank $\ge 2$.  The mask is applied to the trailing
//     two axes; leading axes are treated as batch dimensions.
// k : int
//     Diagonal offset.  Default 0 (main diagonal).
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a`` with the upper triangle
//     (relative to ``k``) zeroed.
//
// Math
// ----
// $$\mathrm{out}_{\ldots,i,j} = \begin{cases}
//     a_{\ldots,i,j} & \text{if } j - i \le k \\
//     0 & \text{otherwise}
// \end{cases}$$
//
// Notes
// -----
// Backward applies ``tril`` with the same ``k`` to the incoming gradient,
// preserving only the gradient at positions retained in the forward pass.
//
// See Also
// --------
// triu_op : Upper-triangular counterpart.
LUCID_API TensorImplPtr tril_op(const TensorImplPtr& a, int k);

// Upper-triangular part of a matrix (or batch of matrices).
//
// Zeros every element strictly below the ``k``-th diagonal of the last
// two axes of ``a``.  Diagonal offset convention matches :func:`tril_op`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of rank $\ge 2$.
// k : int
//     Diagonal offset.  Default 0.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a`` with the lower triangle
//     (relative to ``k``) zeroed.
//
// Math
// ----
// $$\mathrm{out}_{\ldots,i,j} = \begin{cases}
//     a_{\ldots,i,j} & \text{if } j - i \ge k \\
//     0 & \text{otherwise}
// \end{cases}$$
//
// Notes
// -----
// Backward applies ``triu`` with the same ``k`` to the incoming gradient.
//
// See Also
// --------
// tril_op : Lower-triangular counterpart.
LUCID_API TensorImplPtr triu_op(const TensorImplPtr& a, int k);

}  // namespace lucid
