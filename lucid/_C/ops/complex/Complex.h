// lucid/_C/ops/complex/Complex.h
//
// Forward op that combines two real-valued tensors into one complex tensor.
//
// Given broadcastable real-floating inputs $a$ and $b$ on one device, build
// the complex (C64, or CPU C128) tensor $z = a + b\,i$. CPU interleaves
// the two arrays into the canonical ``[re, im, re, im, ...]``
// storage layout used by Lucid for complex dtypes; GPU constructs the result
// as ``re + 1j * im`` via ``mlx::core::astype`` + ``multiply`` + ``add``.
//
// Native storage and graph backward are composed
// from ``real`` / ``imag`` of the incoming
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

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Combine two real tensors into a complex tensor.
//
// The result is C64 for F32 parts or C128 for CPU F64 parts, on the input device.
// Both inputs must be real-floating; half formats widen to F32. The
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
//     Imaginary part. Broadcastable shape, same device and widened dtype as re.
//
// Returns
// -------
// TensorImplPtr
//     Complex tensor with the broadcast shape and input device.
//
// Raises
// ------
// DtypeMismatch
//     If either input is not a real-floating dtype.
// ShapeMismatch
//     If ``re`` and ``im`` cannot broadcast.
// DeviceMismatch
//     If ``re`` and ``im`` live on different devices.
//
// Notes
// -----
// Native backward extracts both lanes of the incoming complex gradient.
// Graph mode uses the same projections to preserve higher derivatives.
//
// See Also
// --------
// real_op, imag_op, conj_op
// Backward for ``complex`` — see :class:`RealBackward`.  The inverse of
// the two projections: each operand takes the lane it supplied.
class LUCID_API ComplexBackward : public FuncOp<ComplexBackward, 2> {
public:
    static const OpSchema schema_v1;
    Shape shape_;
    Device device_ = Device::CPU;

    // Returns ``(real(g), imag(g))``, one per operand.
    std::vector<Storage> apply(Storage grad_out) override;
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

LUCID_API TensorImplPtr complex_op(const TensorImplPtr& re, const TensorImplPtr& im);

}  // namespace lucid
