// lucid/_C/ops/complex/Real.h
//
// Real-part extraction: complex (C64) input → F32 output of the same shape.
// Each backend implements the extraction natively — CPU walks the
// interleaved ``[re, im, re, im, ...]`` storage with stride-2 reads, GPU
// dispatches to ``mlx::core::real``.
//
// Forward only (autograd handled at the Python layer via
// ``lucid.autograd.Function``: ``d real(x) / d x = complex(grad, 0)``).

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr real_op(const TensorImplPtr& a);

}  // namespace lucid
