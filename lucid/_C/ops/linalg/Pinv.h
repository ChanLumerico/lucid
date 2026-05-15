// lucid/_C/ops/linalg/Pinv.h
//
// Moore-Penrose pseudoinverse op: given a matrix A of shape (..., m, n),
// compute A⁺ of shape (..., n, m) via the SVD-based formula
//   A⁺ = V S⁺ Uᵀ
// where S⁺ inverts the non-zero singular values (below a rcond threshold,
// singular values are treated as zero).
//
// Forward dispatch goes to IBackend::linalg_pinv(), which uses LAPACK's
// dgesdd (via SVD) on the CPU path and mlx::core::linalg::pinv on the GPU path.
// The output shape is the input shape with the last two dimensions transposed.
//
// Note: no backward node is registered.  A future backward would express
// gradients in terms of A⁺ and the upstream gradient following the standard
// pseudoinverse differential identity.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the Moore-Penrose pseudoinverse of matrix a.
//
// Input shape (..., m, n) → output shape (..., n, m).
// Validates that a is at least 2-D and float-typed before dispatching.
LUCID_API TensorImplPtr pinv_op(const TensorImplPtr& a);

}  // namespace lucid
