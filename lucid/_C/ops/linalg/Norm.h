// lucid/_C/ops/linalg/Norm.h
//
// Tensor norm op: compute the Frobenius (L2 / Euclidean) or L1 norm over
// specified axes of a float tensor, with optional keepdims.
//
// Forward dispatch goes to IBackend::linalg_norm(), which uses Apple
// Accelerate vDSP routines on the CPU path and MLX reduction ops on the
// GPU path.  The output shape is the input shape with the reduced axes
// either removed (keepdims=false) or collapsed to size 1 (keepdims=true).
//
// Supported norm orders (ord parameter):
//   ord=2.0 — Frobenius / Euclidean / L2 norm: ‖A‖_F = sqrt(sum(aᵢⱼ²))
//   ord=1.0 — L1 norm: ‖A‖_1 = sum(|aᵢⱼ|)
//   Other values will produce an error in the backward pass (gradient not
//   implemented); some may be forwarded to the backend if the backend handles
//   them.
//
// Backward formulas:
//   ord=2:  ∂L/∂A = (A / max(‖A‖, ε)) * expand(∂L/∂N)
//           (clipping avoids division by zero at A=0)
//   ord=1:  ∂L/∂A = sign(A) * expand(∂L/∂N)
//
// The expand() operation broadcasts the reduced gradient back to the full
// input shape by reinserting the collapsed axes.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the norm backward pass.
//
// Invariants:
//   saved_inputs_[0] = Storage of the input tensor A
//   saved_output_    = Storage of norm(A) (the reduced output N)
//   ord_             = the norm order used in the forward pass
//   axis_            = axes that were reduced (empty = all axes)
//   keepdims_        = whether reduced axes were kept as size-1 dimensions
//
// A is saved because the L2 backward needs A/N (element-divide).
// N is saved to avoid recomputing the norm during the backward.
//
// Backward formulas:
//   ord=2:  ∂L/∂A = (A / clip(N, 1e-12, ∞)) * expand(∂L/∂N)
//           The clip threshold 1e-12 is chosen to be below any representable
//           F32 norm of a non-trivial tensor while safely above zero.
//   ord=1:  ∂L/∂A = sign(A) * expand(∂L/∂N)
//   other:  not_implemented error (raised at backward time, not forward time)
//
// expand() reconstructs the broadcast-compatible shape by:
//   - If keepdims_=true or the reduction was over all axes: broadcast directly.
//   - Otherwise: unsqueeze each reduced axis in sorted order, then broadcast.
//   Sorted unsqueeze is important because inserting axis i shifts all axes
//   at positions >= i, so previously computed positions remain valid when
//   inserted in ascending order.
class LUCID_API NormBackward : public FuncOp<NormBackward, 1> {
public:
    static const OpSchema schema_v1;

    // Norm order that was used in the forward pass (typically 1.0 or 2.0).
    double ord_ = 2.0;

    // Axes that were reduced; empty means a full reduction over all elements.
    std::vector<int> axis_;

    // Whether keepdims was true in the forward pass; affects how grad is broadcast.
    bool keepdims_ = false;

    // Compute and return the gradient for the single input A.
    //
    // Returns a one-element vector {∂L/∂A}.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Compute the p-norm of tensor a reduced over the given axes.
//
// ord  — norm order: 1.0 for L1, 2.0 for Frobenius/Euclidean.
// axis — dimensions to reduce (empty = all dims, i.e. global reduction).
// keepdims — if true, reduced dimensions remain as size-1 stubs in the output.
// Wires NormBackward into the autograd graph when grad mode is active.
LUCID_API TensorImplPtr norm_op(const TensorImplPtr& a,
                                double ord,
                                std::vector<int> axis,
                                bool keepdims);

}  // namespace lucid
