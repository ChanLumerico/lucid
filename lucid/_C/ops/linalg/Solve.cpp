// lucid/_C/ops/linalg/Solve.cpp
//
// Implementation of the linear system solve op and its autograd backward node.
//
// Forward: dispatches to IBackend::linalg_solve() via Dispatcher.
//   CPU path: LAPACK dgesv performs LU factorisation of A (in-place) followed
//             by the forward and backward substitution steps to solve AX = B.
//   GPU path: mlx::core::linalg::solve() on the CPU stream.
//
// Backward:
//   Given upstream gradient G = ∂L/∂X:
//     ∂L/∂B = solve(Aᵀ, G)
//     ∂L/∂A = -(∂L/∂B) Xᵀ
//   Both calls are composed from existing ops so that second-order gradients
//   flow automatically.
//
// Note on the backward solve call: the transpose solve solve(Aᵀ, G) does
// not reuse the LU factors from the forward pass; instead it factors Aᵀ
// independently.  A future optimisation could pass the LU factors through
// as a saved tensor to halve the backward factorisation cost.

#include "Solve.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Matmul.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Transpose.h"
#include "_Detail.h"

namespace lucid {

// schema_v1: the OpSchema tag "solve" with one saved input slot.  The template
// parameter 2 on FuncOp means two gradient outputs (one per operand), but the
// schema input count of 1 refers to the number of saved inputs in the registry
// sense.  AmpPolicy::KeepInput prevents lossy dtype promotion before the solve.
const OpSchema SolveBackward::schema_v1{"solve", 1, AmpPolicy::KeepInput};

// Backward pass for solve_op.
//
// Given upstream gradient G = ∂L/∂X:
//   ∂L/∂B = solve(Aᵀ, G)     — solving the transpose system
//   ∂L/∂A = -(∂L/∂B) Xᵀ     — outer product of the two gradients
//
// Derivation:
//   Differentiate AX = B with respect to each input:
//     d/dB: A dX = dB  =>  dX = A⁻¹ dB  =>  ∂L/∂B = (A⁻¹)ᵀ G = (Aᵀ)⁻¹ G
//     d/dA: (dA) X + A dX = 0  =>  dX = -A⁻¹ (dA) X
//           =>  ∂L/∂A = -G Xᵀ (A⁻ᵀ)ᵀ ... with the result from dB inserted:
//           ∂L/∂A = -(∂L/∂B) Xᵀ
//
// The result is returned as {∂L/∂A, ∂L/∂B} to align with the input ordering
// [A=0, B=1] that NaryKernel uses when distributing gradients.
std::vector<Storage> SolveBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;

    // Recover A (saved input), G (upstream gradient), and X (saved output).
    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto dX = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    auto X = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);

    // Transpose of A; needed to solve the adjoint system Aᵀ dB = G.
    auto AT = mT_op(A);
    // ∂L/∂B = solve(Aᵀ, G)
    auto dB = solve_op(AT, dX);

    // ∂L/∂A = -dB @ Xᵀ
    auto XT = mT_op(X);
    auto dA = neg_op(matmul_op(dB, XT));
    // Return in input order: [dA, dB]
    return {dA->storage(), dB->storage()};
}

// Register SolveBackward for graph serialisation and engine lookup.
LUCID_REGISTER_OP(SolveBackward)

// Solve AX = B for X.
//
// The output has the same shape as B.  save_inputs=true is required because
// SolveBackward needs A to call solve(Aᵀ, G) in the backward pass.
// save_inputs also implicitly saves B (via NaryKernel), but B is not accessed
// in the backward — only A (index 0) and the saved output X are used.
TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    Validator::input(a, "solve.a").float_only().square_2d();
    Validator::pair(a, b, "solve").same_dtype().same_device();
    OpScopeFull scope{"solve", a->device(), a->dtype(), a->shape()};

    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
            .linalg_solve(a->storage(), b->storage(), a->shape(), b->shape(), a->dtype());
    // The output takes the shape of B (= X), not A.
    auto out = linalg_detail::fresh(std::move(out_storage), b->shape(), a->dtype(), a->device());
    auto bwd = std::make_shared<SolveBackward>();
    // Save X so the backward can form -dB @ Xᵀ.
    bwd->saved_output_ = out->storage();
    // save_inputs=true: SolveBackward needs A (saved_inputs_[0]) to form solve(Aᵀ, grad).
    kernel::NaryKernel<SolveBackward, 2>::wire_autograd(std::move(bwd), {a, b}, out, true);
    return out;
}

}  // namespace lucid
