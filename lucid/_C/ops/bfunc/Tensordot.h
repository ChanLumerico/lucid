// lucid/_C/ops/bfunc/Tensordot.h
//
// Declares tensordot_op, the entry point for tensor contraction over specified
// axis pairs.  Semantics match numpy.tensordot: axes_a[i] and axes_b[i] name
// paired axes that are summed out.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Contract tensor A over axes_a against tensor B over axes_b.
//
// axes_a and axes_b must have equal length.  Negative axis indices are
// normalised to positive before use.  When gradient tracking is active, the
// contraction is lowered to an equivalent einsum string and forwarded to
// einsum_op.  On the CPU inference path a hand-written GEMM loop is used.
LUCID_API TensorImplPtr tensordot_op(const TensorImplPtr& a,
                                     const TensorImplPtr& b,
                                     std::vector<int> axes_a,
                                     std::vector<int> axes_b);

}
