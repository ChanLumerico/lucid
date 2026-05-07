// lucid/_C/ops/complex/Conj.h
//
// Element-wise complex conjugate.  For complex (C64) input the imaginary
// part is negated; for real dtypes the input is returned unchanged (the
// conjugate of a real number is itself).
//
// CPU uses ``vDSP_vneg`` over the imag-stride-2 view for the C64 path;
// GPU dispatches to ``mlx::core::conjugate``.  The real-dtype shortcut
// avoids any allocation or copy.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr conj_op(const TensorImplPtr& a);

}  // namespace lucid
