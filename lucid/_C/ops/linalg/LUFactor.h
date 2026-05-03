// lucid/_C/ops/linalg/LUFactor.h
//
// LU factorisation op: given a square matrix A, compute the packed LU
// decomposition using LAPACK dgetrf_ (partial pivoting).
//
// Returns a pair (LU, pivots):
//   LU     — n×n packed matrix; upper triangle is U, lower triangle (excluding
//             the diagonal) is L with an implicit unit diagonal.
//   pivots — int32 vector of 1-based pivot indices (length n per batch element).
//
// This matches the output of torch.linalg.lu_factor exactly.
//
// No autograd is wired: the backward through LU factorisation (needed for
// differentiable linear solvers) is not implemented.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the LU factorisation of A.
//
// Returns {LU_packed, pivots} where LU_packed has the same shape as A and
// pivots has shape (..., n) with dtype I32.
LUCID_API std::vector<TensorImplPtr> lu_factor_op(const TensorImplPtr& a);

}  // namespace lucid
