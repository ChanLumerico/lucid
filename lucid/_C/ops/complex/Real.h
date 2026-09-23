// lucid/_C/ops/complex/Real.h
//
// Real-part extraction $\Re(z) = a$ for $z = a + b\,i$.
//
// Complex (C64) input yields an F32 output of the same shape.  Each backend
// implements the projection natively: CPU walks the interleaved
// ``[re, im, re, im, ...]`` storage with stride-2 reads from the real offset,
// GPU dispatches to ``mlx::core::real``.  Real-dtype inputs are rejected by
// ``complex_detail::require_complex``.
//
// Forward only — the Python autograd layer embeds the incoming real-valued
// gradient as the real part of a fresh complex gradient with zero imaginary
// part: ``d real(z) / d z = complex(grad, 0)``.  This matches the Wirtinger
// calculus convention used by complex autograd in the reference framework.
//
// Math
// ----
// $$
//   y = \Re(z), \qquad
//   \frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} + 0\,i
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

// Extract the real part of a complex tensor as a real tensor.
//
// The result dtype is the corresponding real dtype (``C64`` → ``F32``); the
// shape and device are unchanged.
//
// Math
// ----
// $$
//   y_k = \Re(z_k)
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
// imag_op, complex_op, conj_op
// Backward for the complex projections.
//
// None of them had one.  ``real``, ``imag`` and ``conj`` built their
// output and returned it, so a complex tensor was where every gradient
// stopped:
//
//     real(fft(x)).sum().backward()   ->  x.grad is None
//
// with no error raised.  ``fft`` is wired correctly; the chain broke one
// step later, which meant any loss written through a frequency-domain
// transform trained on nothing at all.  ``abs`` on a complex input is a
// composite of ``real`` and ``imag``, so it was unreachable for the same
// reason and becomes differentiable with them.
//
// The header above used to say this belonged to "the Python autograd
// layer".  There was no such layer.
//
// The conventions are the reference's, measured rather than derived:
//
//     d/dz real(z) = g + 0i        d/dz imag(z) = 0 + g i
//     d/dz conj(z) = conj(g)       d/dz |z|     = g z / |z|
//
// Each is expressible with ops that already exist, so the backward is a
// storage-level call rather than a new kernel.
class LUCID_API RealBackward : public FuncOp<RealBackward, 1> {
public:
    static const OpSchema schema_v1;
    Shape shape_;
    Dtype lane_ = Dtype::F32;  // the real dtype one lane holds
    Device device_ = Device::CPU;

    // ``g`` lands in the real lane; the imaginary lane receives nothing.
    std::vector<Storage> apply(Storage grad_out);
};

LUCID_API TensorImplPtr real_op(const TensorImplPtr& a);

}  // namespace lucid
