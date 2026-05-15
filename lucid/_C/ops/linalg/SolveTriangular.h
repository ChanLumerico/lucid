// lucid/_C/ops/linalg/SolveTriangular.h
//
// Triangular solve: given a triangular matrix A and a right-hand-side B,
// compute X such that A X = B.
//
// Parameters mirror solve_triangular:
//   upper          — true if A is upper triangular, false if lower.
//   unitriangular  — if true, the diagonal of A is treated as all-ones
//                    (unit triangular; the actual values are ignored).
//
// Dispatches to LAPACK strtrs_/dtrtrs_ on CPU.
// No autograd node is wired.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Solve A X = B for X where A is triangular.
// Output shape equals the shape of B.
LUCID_API TensorImplPtr solve_triangular_op(const TensorImplPtr& a,
                                            const TensorImplPtr& b,
                                            bool upper = true,
                                            bool unitriangular = false);

}  // namespace lucid
