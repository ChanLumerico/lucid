// lucid/_C/ops/complex/Complex.h
//
// Forward op that combines two real-valued tensors into one complex tensor.
//
// Given real-floating inputs $a$ and $b$ of identical shape and device, build
// the complex (C64) tensor $z = a + b\,i$.  CPU uses ``vDSP_ztoc`` to
// interleave the two arrays into the canonical ``[re, im, re, im, ...]``
// storage layout used by Lucid for complex dtypes; GPU constructs the result
// as ``re + 1j * im`` via ``mlx::core::astype`` + ``multiply`` + ``add``.
//
// This entry point is forward-only — the Wirtinger-style backward is composed
// at the Python autograd layer from ``real`` / ``imag`` of the incoming
// gradient (``d complex(re, im) / d re = real(grad)``,
// ``d complex(re, im) / d im = imag(grad)``).
//
// Math
// ----
// $$
//   z = a + b\,i, \qquad
//   \frac{\partial L}{\partial a} = \Re\!\left(\frac{\partial L}{\partial z}\right), \quad
//   \frac{\partial L}{\partial b} = \Im\!\left(\frac{\partial L}{\partial z}\right)
// $$

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Combine two real tensors into a complex tensor.
//
// The result has dtype ``C64`` and the same shape and device as the inputs.
// Both inputs must be real-floating (``F16`` / ``F32`` / ``F64``); the
// resulting interleaved storage holds $a + b\,i$ element-wise.
//
// Math
// ----
// $$
//   z_k = a_k + b_k\,i
// $$
//
// Parameters
// ----------
// re : TensorImplPtr
//     Real part.  Must be a real-floating dtype.
// im : TensorImplPtr
//     Imaginary part.  Same shape, device, and dtype family as ``re``.
//
// Returns
// -------
// TensorImplPtr
//     Complex (C64) tensor with the same shape and device as the inputs.
//
// Raises
// ------
// DtypeMismatch
//     If either input is not a real-floating dtype.
// ShapeMismatch
//     If ``re`` and ``im`` have different shapes.
// DeviceMismatch
//     If ``re`` and ``im`` live on different devices.
//
// Notes
// -----
// Backward is supplied by the Python autograd layer via ``real`` / ``imag``
// extraction of the incoming complex gradient; this entry point only
// implements the forward.
//
// See Also
// --------
// real_op, imag_op, conj_op
LUCID_API TensorImplPtr complex_op(const TensorImplPtr& re, const TensorImplPtr& im);

}  // namespace lucid
