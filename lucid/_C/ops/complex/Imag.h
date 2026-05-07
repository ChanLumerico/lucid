// lucid/_C/ops/complex/Imag.h
//
// Imaginary-part extraction: complex (C64) input → F32 output of the same
// shape.  CPU walks interleaved storage at stride-2 starting from the
// imaginary offset; GPU dispatches to ``mlx::core::imag``.
//
// Forward only — autograd handled at the Python layer (the gradient
// w.r.t. complex input is ``complex(0, grad)``).

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr imag_op(const TensorImplPtr& a);

}  // namespace lucid
