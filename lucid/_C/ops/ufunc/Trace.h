// lucid/_C/ops/ufunc/Trace.h
//
// Public entry point for the matrix trace operation.  The backward node
// (TraceBackward) is defined entirely inside Trace.cpp to keep it private.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Sum the main-diagonal elements of a matrix (or a batch of matrices).
// For a 2-D input of shape [m, n] the result is a scalar.  For an input of
// shape [b..., m, n] the result has shape [b...].
// Autograd wiring is performed only for strictly 2-D inputs; batch trace does
// not yet support autograd and returns a plain tensor.
LUCID_API TensorImplPtr trace_op(const TensorImplPtr& a);

}  // namespace lucid
