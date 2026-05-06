// lucid/_C/ops/composite/Matrix.h
//
// Matrix-product compositions: rank-checked aliases of ``matmul`` plus the
// Kronecker product.
//
//   mm(a, b)   — strict 2-D matmul
//   bmm(a, b)  — strict 3-D batched matmul (batch dims must agree)
//   kron(a, b) — block tensor product over matching ranks
//
// All entry points compose existing differentiable ops; backward flows
// through ``MatmulBackward``, ``MulBackward``, and ``ReshapeBackward``.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr mm_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr bmm_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr kron_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
