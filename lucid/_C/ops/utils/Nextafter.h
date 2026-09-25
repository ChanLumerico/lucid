// lucid/_C/ops/utils/Nextafter.h
//
// IEEE-754 "next representable float" operation.  ``nextafter(a, b)`` returns
// the float adjacent to ``a`` in the direction of ``b``.  Two compute paths:
//
//   * (GPU, F32): pure MLX bit-twiddle pipeline — view to int32, conditional
//     ±1 step, view back.  No CPU round-trip.
//   * (CPU, F32 or F64) and (GPU, F64): per-element ``std::nextafter`` loop
//     on host memory.  F64 has no GPU path because MLX does not support F64.
//
// Both operands must have the same floating-point dtype (F32 or F64).

#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Return the next representable floating-point value after ``a`` toward ``b``.
//
// Computes element-wise IEEE-754 ``nextafter``: for each element pair
// ``(a_k, b_k)`` returns the float bit-adjacent to ``a_k`` in the
// direction of ``b_k``.  If ``a_k == b_k`` the result is ``b_k``.  NaN
// propagates.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source values.  Floating-point dtype (F32 or F64).
// b : TensorImplPtr
//     Direction values.  Must match ``a`` in dtype and broadcast-compatible
//     shape.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same dtype and broadcast shape as the inputs.
//
// Math
// ----
// $$\mathrm{out}_k = \begin{cases}
//     a_k & \text{if } a_k = b_k \\
//     \mathrm{succ}(a_k) & \text{if } a_k < b_k \\
//     \mathrm{pred}(a_k) & \text{if } a_k > b_k
// \end{cases}$$
// where $\mathrm{succ}$ / $\mathrm{pred}$ are the IEEE-754 next- / previous-
// representable functions.
//
// Notes
// -----
// Non-differentiable: the output is piecewise constant in ``a`` (the step
// is always one ULP regardless of magnitude).  No backward is registered.
//
// On the GPU stream, the F32 fast path operates entirely on int32 views of
// the floating bits (sign-magnitude adjusted to two's complement) and avoids
// a CPU round-trip.  F64 inputs always traverse the CPU stream because MLX
// does not provide F64 buffers.
//
// Raises
// ------
// ValueError
//     If ``a`` and ``b`` differ in dtype, or the dtype is not F32 / F64.
// Autograd node for ``nextafter(a, b)``.
//
// The result is ``a`` moved by one unit in the last place, so within a
// step it follows ``a`` exactly and does not depend on ``b`` at all: the
// gradient is the incoming one for ``a`` and zero for ``b`` — the reference
// framework's rule.  The op used to return a detached tensor, dropping the
// gradient to ``a`` without a word.
class LUCID_API NextafterBackward : public FuncOp<NextafterBackward, 2> {
public:
    static const OpSchema schema_v1;
    std::vector<Storage> apply(Storage grad_out) override;
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

LUCID_API TensorImplPtr nextafter_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
