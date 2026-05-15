// lucid/_C/ops/ufunc/Var.h
//
// Public entry point for the variance reduction op.  The backward node
// (VarBackward) is defined entirely inside Var.cpp to keep it private.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the biased variance of `a` along `axes` and return the result.
// keepdims controls whether collapsed axes are retained as size-1 dimensions.
// Autograd is wired when a->requires_grad() and GradMode is enabled.
LUCID_API TensorImplPtr var_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

}  // namespace lucid
