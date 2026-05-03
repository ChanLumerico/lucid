// lucid/_C/ops/linalg/MatrixPower.h
//
// Integer matrix power op: given a square matrix A and an integer exponent p,
// compute A^p (A to the power p).
//
// The backend uses repeated squaring (binary exponentiation) for O(log|p|)
// matrix multiplications, which is much cheaper than p naive multiplications
// for large |p|.
//
// Special cases handled by the backend:
//   p == 0: returns the identity matrix I (of the same size as A).
//   p > 0:  standard repeated squaring of A.
//   p < 0:  compute (A⁻¹)^|p|; requires A to be invertible.
//
// Forward dispatch goes to IBackend::linalg_matrix_power().
// The output shape is identical to the input shape.
//
// Note: no backward node is registered.  The backward through repeated
// squaring requires propagating through the binary decomposition tree.
// For a chain of k squarings, the gradient is a sum of k terms, each
// involving all the intermediate powers.  This is deferred to a future phase.
//
// Typical use case: computing matrix exponentials or kernel powers in graph
// neural networks where A is an adjacency/Laplacian matrix.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute A^p for a square float matrix a with integer exponent p.
//
// Validates that a is at least 2-D, square, and float-typed before dispatching.
// The parameter is named n in the function signature for NumPy API parity
// (numpy.linalg.matrix_power(a, n)); internally the implementation uses p.
// Autograd is not wired; the result is a leaf in the gradient graph.
LUCID_API TensorImplPtr matrix_power_op(const TensorImplPtr& a, int n);

}  // namespace lucid
