// lucid/_C/ops/complex/Complex.h
//
// Build a complex (C64) tensor from two real (F32) tensors of identical
// shape.  CPU uses ``vDSP_ztoc`` to interleave the two arrays into the
// ``[re, im, re, im, ...]`` layout; GPU constructs ``re + 1j * im`` via
// ``mlx::core::astype`` + ``multiply`` + ``add``.
//
// Forward only — autograd handled at the Python layer
// (``d complex(re, im) / d re = real(grad)``,
//  ``d complex(re, im) / d im = imag(grad)``).

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr complex_op(const TensorImplPtr& re, const TensorImplPtr& im);

}  // namespace lucid
