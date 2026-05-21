// lucid/_C/ops/complex/Conj.h
//
// Element-wise complex conjugate $\bar{z} = a - b\,i$ for $z = a + b\,i$.
//
// For complex (C64) input only the imaginary half of the interleaved
// ``[re, im, re, im, ...]`` storage is negated; for real dtypes the input is
// returned unchanged (the conjugate of a real number is itself), so the
// real-dtype path is a true identity that allocates and copies nothing.
//
// CPU uses ``vDSP_vneg`` over the imag-stride-2 view for the C64 path; GPU
// dispatches to ``mlx::core::conjugate``.
//
// Forward only — the Wirtinger-style backward composed at the Python layer is
// again ``conj`` (the conjugate operator is its own adjoint up to sign on the
// holomorphic component, which under Lucid's convention means ``grad_in =
// conj(grad_out)``).
//
// Math
// ----
// $$
//   y = \overline{z} = a - b\,i, \qquad
//   \frac{\partial L}{\partial z} = \overline{\frac{\partial L}{\partial y}}
// $$
//
// References
// ----------
// Hirose, "Complex-Valued Neural Networks" (2003), §3.4 — Wirtinger
// derivatives for conjugation.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Return the element-wise complex conjugate of a tensor.
//
// For ``C64`` input, produces a fresh C64 tensor whose imaginary part is
// negated.  For any real dtype the operator is a no-op (the conjugate of a
// real number equals itself) and the backend short-circuits the dispatch.
//
// Math
// ----
// $$
//   y_k = \overline{z_k}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  May be any dtype; only ``C64`` produces a non-trivial
//     transformation.
//
// Returns
// -------
// TensorImplPtr
//     Tensor with the same shape, dtype, and device as ``a``.
//
// Notes
// -----
// Backward is performed at the Python autograd layer as another ``conj`` on
// the incoming gradient, consistent with the Wirtinger calculus convention
// adopted by the reference framework.
//
// See Also
// --------
// real_op, imag_op, complex_op
LUCID_API TensorImplPtr conj_op(const TensorImplPtr& a);

}  // namespace lucid
