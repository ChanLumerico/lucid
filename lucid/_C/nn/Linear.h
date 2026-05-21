// lucid/_C/nn/Linear.h
//
// Autograd-aware fully-connected (linear / affine) layer that implements
// $\mathbf{y} = \mathbf{x}\mathbf{W}^\top + \mathbf{b}$.
//
// This header declares the autograd node ``LinearBackward`` (FuncOp<…, 3>
// with three saved-input slots — ``x``, ``W``, ``b``) and the free
// function entry point ``linear_op`` that the Python binding layer
// forwards to.  All leading batch dimensions of ``x`` are flattened
// internally into a single $M$-row GEMM and the output dimension
// list is then restored.  AMP autocast is honoured: under
// ``AutocastGuard(F16)`` all three operands are cast to the effective
// dtype via ``astype_op`` (which preserves grad wiring) before reaching
// the backend GEMM.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the linear (fully-connected) transformation
// $\mathbf{y} = \mathbf{x}\mathbf{W}^\top + \mathbf{b}$.
//
// Inherits ``FuncOp<LinearBackward, 3>`` which wires three saved-input
// slots (``x``, ``W``, ``b``) and records each operand's shape so that
// :func:`apply` can reconstruct the three gradients during backward.
// The forward dispatches to ``IBackend::linear`` (BNNS / Accelerate on
// CPU, MLX on Metal); the backward delegates to
// ``IBackend::linear_backward`` which returns ``{dx, dW, db}`` in one
// fused call.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (name ``"linear"``, version 1,
//     :member:`AmpPolicy::Promote`).  ``Promote`` means the op
//     participates in autocast and its operands are cast to the
//     current autocast target dtype — F16 on GPU and F32 on CPU per
//     the Accelerate-no-F16 carve-out.
//
// Notes
// -----
// The bias slot is mandatory at the C++ level.  When the Python
// ``Linear(bias=False)`` form is requested, the binding layer passes
// a zero-length tensor (shape ``(0,)``) and the GEMM contribution from
// ``b`` evaluates to zero.  Thread safety: instances are created once
// during forward and consumed once during backward — no concurrent
// access is expected.
class LUCID_API LinearBackward : public FuncOp<LinearBackward, 3> {
public:
    // Schema for this op.  ``name="linear"``, ``version=1``,
    // ``amp_policy=Promote``, ``produces_grad=true``.
    static const OpSchema schema_v1;

    // Run the forward pass and, when grad mode is on, register ``this``
    // as the ``grad_fn`` of the returned output.
    //
    // The forward flattens all leading dimensions of ``x`` into an
    // ``M``-row batch ($M = \prod_{i<\text{ndim}-1} x.\text{shape}[i]$),
    // dispatches the GEMM through :class:`backend::Dispatcher`, and
    // reports ``2 * M * N * K`` FLOPs to the profiler scope.  Under an
    // active ``AutocastGuard``, the three operands are cast to the
    // effective dtype using ``astype_op`` (which preserves autograd)
    // before the kernel call so the matmul takes the native F16 fast
    // path on Metal.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input tensor of shape $(\ast, K)$ — any number of leading
    //     batch dimensions followed by ``K = in_features``.  Must be
    //     at least 1-D.
    // W : TensorImplPtr
    //     Weight matrix of shape ``(N, K)``.  Treated as
    //     $\mathbf{W}^\top$ at the math level: the kernel multiplies
    //     ``x`` by ``W^T``.
    // b : TensorImplPtr
    //     Bias of shape ``(N,)``.  Pass an empty 1-D tensor when the
    //     Python module is constructed with ``bias=False``.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor of shape $(\ast, N)$, matching ``x``'s leading
    //     dimensions with the last axis replaced by ``N = out_features``.
    //
    // Math
    // ----
    // $$\mathbf{y} = \mathbf{x}\mathbf{W}^\top + \mathbf{b}$$
    //
    // Shape
    // -----
    // - Input ``x`` : $(\ast, K)$
    // - Weight ``W`` : $(N, K)$
    // - Bias ``b`` : $(N,)$
    // - Output : $(\ast, N)$
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``W.shape[1] != x.last_dim`` or ``b.shape[0] != W.shape[0]``.
    // DeviceMismatch
    //     If ``x``, ``W``, and ``b`` do not all live on the same device.
    static TensorImplPtr
    forward(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b);

    // Compute the three gradients from the incoming ``grad_out``.
    //
    // Delegates to ``IBackend::linear_backward`` which produces
    // $dx = \nabla_{\!x}\ell$, $dW = \nabla_{\!W}\ell$, and
    // $db = \nabla_{\!b}\ell$ in a single backend call using the
    // shapes recorded in ``input_shapes_``.  The closed-form is
    //
    // $$dx = \text{grad\_out} \cdot \mathbf{W},\quad
    //   dW = \text{grad\_out}^\top \cdot \mathbf{x}_\text{flat},\quad
    //   db = \sum_{\text{batch}} \text{grad\_out}.$$
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient of shape $(\ast, N)$ matching the forward
    //     output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Three-element vector ``{dx, dW, db}`` matching the saved-input
    //     ordering (``x``, ``W``, ``b``).
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public free-function entry point for the linear op.
//
// Thin wrapper that delegates to :func:`LinearBackward::forward`.  This
// is the symbol the pybind11 binding layer (``bind_nn.cpp``) and other
// C++ call sites use; calling it directly is equivalent to using the
// node's static forward.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of shape $(\ast, K)$.
// W : TensorImplPtr
//     Weight matrix of shape ``(N, K)``.
// b : TensorImplPtr
//     Bias of shape ``(N,)`` (zero-length tensor when bias is disabled).
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of shape $(\ast, N)$.
//
// See Also
// --------
// LinearBackward : Autograd node implementing the actual forward + backward.
LUCID_API TensorImplPtr linear_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b);

}  // namespace lucid
