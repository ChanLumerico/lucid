// lucid/_C/ops/linalg/Det.cpp
//
// Implementation of the determinant op and its autograd backward node.
//
// Forward: dispatches to IBackend::linalg_det() via Dispatcher.
//   CPU path: LAPACK dgetrf computes an LU factorisation; the determinant is
//             the product of the diagonal elements of U, adjusted for the sign
//             of the permutation.
//   GPU path: mlx::core::linalg::det() on the CPU stream (see _Detail.h).
//
// Backward: implements ∂L/∂A = broadcast(det(A) · g, input_shape) * (A⁻¹)ᵀ
//   by reusing inv_op (which itself is differentiable) and existing elementwise
//   ops (mul_op, broadcast_to_op).  This means second-order gradients are
//   available for free: the backward sub-graph is itself differentiable through
//   the same ops.
//
// Output shape: the trailing two matrix dimensions are dropped.  A 3-D input
// [B, N, N] produces a 1-D output [B]; a plain [N, N] input produces a scalar
// (shape []).

#include "Det.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/ufunc/Transpose.h"
#include "../../ops/utils/Layout.h"
#include "Inv.h"
#include "_Detail.h"

namespace lucid {

// schema_v1 tags this node as "det" in the op registry.  AmpPolicy::KeepInput
// prevents AMP from downcasting A before the inverse call in the backward.
const OpSchema DetBackward::schema_v1{"det", 1, AmpPolicy::KeepInput};

// Backward pass for det_op.
//
// Reconstructs A (from saved_inputs_) and det(A) (from saved_output_), then
// applies the Jacobi-formula gradient:
//   ∂L/∂A = broadcast(det(A) · g, input_shape) * (A⁻¹)ᵀ
//
// Step-by-step:
//   1. Compute (A⁻¹)ᵀ by calling inv_op (which itself saves A⁻¹ for its own
//      backward) then mT_op.
//   2. Form the scalar factor: scale = det(A) * g  (both are batch-shaped).
//   3. Broadcast scale to the full matrix shape (inserting the last two dims).
//   4. Elementwise multiply with (A⁻¹)ᵀ to get ∂L/∂A.
//
// NoGradGuard wraps the entire backward to suppress tracking of the backward
// sub-graph itself (the engine would otherwise accumulate a second-order graph
// on each backward call, wasting memory).
std::vector<Storage> DetBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;

    // Recover A, the upstream scalar gradient, and the saved det value.
    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto ddet = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    auto det_v = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);

    // (A⁻¹)ᵀ — inv_op handles batched inversion, mT_op transposes last two dims.
    auto inv_A = inv_op(A);
    auto inv_AT = mT_op(inv_A);
    // Scale the batch-shaped gradient by the batch-shaped determinant.
    auto scale = mul_op(det_v, ddet);
    // broadcast_to_op inserts two trailing dimensions to match the matrix shape.
    auto dA = mul_op(broadcast_to_op(scale, input_shapes_[0]), inv_AT);
    return {dA->storage()};
}

// Register DetBackward so the autograd engine can deserialise it by name.
LUCID_REGISTER_OP(DetBackward)

// Compute det(A).
//
// The output shape is the input shape with the last two matrix dimensions
// removed.  save_inputs=true is passed to wire_autograd because DetBackward
// needs A to call inv_op; without it saved_inputs_[0] would be empty.
TensorImplPtr det_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "det.a").float_only().square_2d();
    OpScopeFull scope{"det", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    // Drop the last two dims: [B, N, N] -> [B],  [N, N] -> [] (scalar).
    Shape out_shape(sh.begin(), sh.end() - 2);

    Storage out_storage =
        backend::Dispatcher::for_device(a->device()).linalg_det(a->storage(), sh, a->dtype());
    auto out = fresh(std::move(out_storage), out_shape, a->dtype(), a->device());
    auto bwd = std::make_shared<DetBackward>();
    // Save det(A) so the backward can use it as the multiplicative factor.
    bwd->saved_output_ = out->storage();
    // save_inputs=true: DetBackward needs A to recompute inv(A) in the backward.
    kernel::NaryKernel<DetBackward, 1>::wire_autograd(std::move(bwd), {a}, out, true);
    return out;
}

}  // namespace lucid
