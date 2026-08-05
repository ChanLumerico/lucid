// lucid/_C/ops/complex/Imag.h
//
// Imaginary-part extraction $\Im(z) = b$ for $z = a + b\,i$.
//
// Complex (C64) input yields an F32 output of the same shape: CPU walks the
// interleaved ``[re, im, re, im, ...]`` storage at stride-2 starting from the
// imaginary offset; GPU dispatches to ``mlx::core::imag``.  Real-dtype inputs
// are rejected by ``complex_detail::require_complex``.
//
// Forward only — the Python autograd layer embeds the incoming real-valued
// gradient as the imaginary part of a fresh complex gradient with zero real
// part: ``d imag(z) / d z = complex(0, grad)``.  Under the Wirtinger
// convention this gives the correct adjoint for the non-holomorphic operator
// $z \mapsto \Im(z)$.
//
// Math
// ----
// $$
//   y = \Im(z), \qquad
//   \frac{\partial L}{\partial z} = 0 + \frac{\partial L}{\partial y}\,i
// $$
//
// References
// ----------
// Hirose, "Complex-Valued Neural Networks: Theories and Applications"
// (2003), §3.4 (Wirtinger derivatives).

#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Extract the imaginary part of a complex tensor as a real tensor.
//
// The result dtype is the corresponding real dtype (``C64`` → ``F32``); the
// shape and device are unchanged.
//
// Math
// ----
// $$
//   y_k = \Im(z_k)
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Complex-dtype input tensor (currently ``C64``).
//
// Returns
// -------
// TensorImplPtr
//     Real-dtype output (``F32``) of the same shape and device as ``a``.
//
// Raises
// ------
// DtypeMismatch
//     If ``a`` is not a complex dtype.
//
// See Also
// --------
// real_op, complex_op, conj_op
// Backward for ``imag`` — see :class:`RealBackward` for why these were
// missing and what it cost.  The same statement about the other lane.
class LUCID_API ImagBackward : public FuncOp<ImagBackward, 1> {
public:
    static const OpSchema schema_v1;
    Shape shape_;
    Dtype lane_ = Dtype::F32;
    Device device_ = Device::CPU;

    // ``g`` lands in the imaginary lane: ``0 + g i``.
    std::vector<Storage> apply(Storage grad_out);
};

LUCID_API TensorImplPtr imag_op(const TensorImplPtr& a);

}  // namespace lucid
