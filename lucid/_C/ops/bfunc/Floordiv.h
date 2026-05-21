// lucid/_C/ops/bfunc/Floordiv.h
//
// Declares floordiv_op, the entry point for element-wise integer floor
// division.  The operation requires equal-shape operands and always returns an
// I64 tensor.  No gradient is defined because floor division is not
// differentiable.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the elementwise floor division $y = \lfloor a / b \rfloor$.
//
// Floor division is **non-differentiable** (piecewise constant with jumps at
// every integer crossing), so no autograd node is registered: the returned
// tensor never carries a ``grad_fn`` and the result effectively detaches its
// inputs from the graph.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Numerator tensor.
// b : TensorImplPtr
//     Denominator tensor.  Must have the **same shape and dtype** as ``a`` —
//     no broadcasting and no implicit dtype promotion is performed.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``a.shape`` and dtype :enum:`Dtype::I64` holding
//     $\lfloor a / b \rfloor$.  The cast to I64 is unconditional regardless
//     of the input dtype, matching Python's ``//`` operator semantics.
//
// Math
// ----
// $$
//   y_i = \left\lfloor \frac{a_i}{b_i} \right\rfloor
// $$
//
// Shape
// -----
// ``a.shape == b.shape``.  The result is the same shape; the dtype is always
// I64.
//
// Raises
// ------
// LucidError
//     If ``a.shape != b.shape`` or ``a.dtype != b.dtype``.
//
// Notes
// -----
// Behaviour on division by zero is delegated to the backend primitive
// (``backend.floordiv``).  Because the output is forced to I64, any
// floating-point ``NaN`` / ``Inf`` produced by the backend is reinterpreted
// to a platform-defined integer value.
//
// See Also
// --------
// div_op : True (real-valued) elementwise division, fully differentiable.
LUCID_API TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
