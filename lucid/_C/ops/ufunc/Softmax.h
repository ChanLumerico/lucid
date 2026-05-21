// lucid/_C/ops/ufunc/Softmax.h
//
// Backward node and entry point for numerically stable softmax along a given
// axis.  SoftmaxBackward inherits from FuncOp (not the standard UnaryOp CRTP)
// because the forward and backward implementations are both non-trivial and
// hand-written rather than following the simple dispatch/grad_formula pattern.
//
// The backward uses the well-known Jacobian-vector product identity:
//   dL/dx = p * (dL/dy - sum_j(p_j * dL/dy_j))
//   where p = softmax(x) and the inner sum contracts along the softmax axis.
// This avoids forming the full Jacobian matrix.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the numerically stable softmax along a single axis
// $p_i = e^{x_i - m} / \sum_j e^{x_j - m}$ with $m = \max_j x_j$.
//
// Subtracting the per-row maximum $m$ before exponentiating is the standard
// log-sum-exp trick — it preserves the analytic result while keeping every
// exponentiated term in $(0, 1]$, which is essential under ``ForceFP32``
// AMP semantics for fp16/bf16 logits.  The softmax probabilities $p$ are
// saved on the node so the backward needs no re-evaluation.
//
// Math
// ----
// $$p_i = \frac{e^{x_i}}{\sum_j e^{x_j}},
//   \qquad \frac{\partial L}{\partial x_i}
//   = p_i\,\Bigl(\frac{\partial L}{\partial y_i}
//                - \sum_j p_j\,\frac{\partial L}{\partial y_j}\Bigr)$$
//
// The Jacobian-vector form sidesteps materialising the dense
// $N \times N$ softmax Jacobian and runs in $O(N)$ per row.
//
// Shape
// -----
// Input and output share the same shape; the normalisation contracts only
// along ``axis_``.  Negative axis values are normalised to non-negative
// indices by ``forward`` before being stored.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"softmax"``, ``AmpPolicy::ForceFP32`` (probability underflow on
//     float16 is unsafe for large logit magnitudes).
// axis_ : int
//     Canonical (non-negative) axis along which the softmax is computed.
// saved_output_ : Storage
//     Softmax probabilities $p$ produced by ``forward``, used by ``apply``.
//
// Raises
// ------
// IndexError
//     If the normalised axis falls outside ``[0, ndim)``.
//
// References
// ----------
// Bishop, "Pattern Recognition and Machine Learning", §4.3.4.
class LUCID_API SoftmaxBackward : public FuncOp<SoftmaxBackward, 1> {
public:
    static const OpSchema schema_v1;
    int axis_ = -1;

    // Validate the axis, dispatch the forward softmax through the backend,
    // save the resulting probabilities on the node, and wire the autograd
    // edges back to ``a``.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input logits.
    // axis : int
    //     Softmax axis; negative values are normalised to ``axis + ndim``.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output probabilities $p$ with the same shape and dtype as ``a``.
    //
    // Raises
    // ------
    // IndexError
    //     If ``axis`` is out of range after normalisation.
    static TensorImplPtr forward(const TensorImplPtr& a, int axis);

    // Eager-mode backward: dispatches to ``IBackend::softmax_backward``
    // which evaluates $p \odot (\partial L/\partial y - \langle p,
    // \partial L/\partial y\rangle)$ in a single fused pass along ``axis_``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Numerically stable softmax $p_i = e^{x_i} / \sum_j e^{x_j}$ along ``axis``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input logits.
// axis : int
//     Axis along which to normalise.  Negative values count from the end.
//
// Returns
// -------
// TensorImplPtr
//     Output probabilities with the same shape and dtype as ``a``.
//
// See Also
// --------
// log_softmax_op : Numerically advantageous log-domain variant for
//     downstream NLL / cross-entropy losses.
// SoftmaxBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr softmax_op(const TensorImplPtr& a, int axis);

// Autograd node for the numerically stable log-softmax along a single axis
// $y = x - \mathrm{logsumexp}(x, \text{axis})$.
//
// Computing $\log\mathrm{softmax}$ directly (rather than $\log$ of
// $\mathrm{softmax}$) is the standard numerical-stability win when the
// downstream loss is NLL / cross-entropy: it avoids the catastrophic
// cancellation that arises when $p_i$ underflows to zero before the log
// step.  Forward saves the log-softmax output on the node, which the
// backward reuses to recover the probabilities as $\exp(y)$.
//
// Math
// ----
// $$y_i = x_i - \log\sum_j e^{x_j},
//   \qquad \frac{\partial L}{\partial x_i}
//   = \frac{\partial L}{\partial y_i}
//     - e^{y_i}\,\sum_j \frac{\partial L}{\partial y_j}$$
//
// Here $e^{y_i} = p_i$ is the underlying softmax probability.
//
// Shape
// -----
// Input and output share the same shape; normalisation contracts along
// ``axis_`` only.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"log_softmax"``, ``AmpPolicy::ForceFP32``.
// axis_ : int
//     Canonical (non-negative) axis stored after normalisation by ``forward``.
// saved_output_ : Storage
//     Log-softmax output $y$ saved for backward.
//
// Raises
// ------
// IndexError
//     If the normalised axis falls outside ``[0, ndim)``.
//
// Notes
// -----
// LogSoftmax is preferred over ``log(softmax(x))`` because it remains
// finite when probabilities underflow: the subtraction is performed in
// the log domain via ``logsumexp`` rather than producing zero and then
// taking the log.
class LUCID_API LogSoftmaxBackward : public FuncOp<LogSoftmaxBackward, 1> {
public:
    static const OpSchema schema_v1;
    int axis_ = -1;

    // Validate the axis, dispatch the forward log-softmax, save its output,
    // and wire autograd back to ``a``.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input logits.
    // axis : int
    //     Axis along which to normalise; negative values are wrapped.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Log-softmax output $y$ with the same shape and dtype as ``a``.
    static TensorImplPtr forward(const TensorImplPtr& a, int axis);

    // Eager-mode backward: dispatches to ``IBackend::log_softmax_backward``
    // which computes $\partial L/\partial x = \partial L/\partial y
    // - e^{y}\,\sum_{\text{axis}}\partial L/\partial y$.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Numerically stable log-softmax $y = x - \mathrm{logsumexp}(x, \text{axis})$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input logits.
// axis : int
//     Axis along which to normalise.  Negative values count from the end.
//
// Returns
// -------
// TensorImplPtr
//     Log-softmax output with the same shape and dtype as ``a``.
//
// See Also
// --------
// softmax_op : Probability-domain variant.
// LogSoftmaxBackward : Autograd node implementing the gradient rule.
//
// Notes
// -----
// Prefer ``log_softmax_op`` followed by ``nll_loss`` over
// ``log(softmax_op(...))`` followed by ``nll_loss`` — the former is
// numerically stable for very confident logits, the latter is not.
LUCID_API TensorImplPtr log_softmax_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
