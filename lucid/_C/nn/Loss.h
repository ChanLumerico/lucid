// lucid/_C/nn/Loss.h
//
// Autograd-aware loss functions: regression (MSE / Huber) and classification
// (BCE, BCE-with-logits, cross-entropy, NLL) primitives plus CTC.
//
// All forward methods produce either a scalar (after Mean / Sum reduction) or
// an element-wise loss tensor (Reduction::None) and attach the corresponding
// backward node whenever any input requires a gradient.  The op schemas are
// pinned to ``AmpPolicy::ForceFP32`` so that the reduction sums accumulate in
// single precision regardless of the surrounding autocast context — this is
// essential for numerical stability of log-sum-exp and product-of-many-terms
// gradients.
//
// Supported losses
// ----------------
// * ``MseLoss``           — mean squared error, $\frac{1}{N} \sum (\hat{y} - y)^2$.
// * ``BCELoss``           — binary cross-entropy on probabilities in $(0, 1)$.
// * ``BCEWithLogits``     — numerically stable BCE applied directly to logits.
// * ``CrossEntropy``      — fused log-softmax + NLL with optional class weights,
//                           label smoothing and an ignore-index mask.
// * ``NLLLoss``           — negative log-likelihood applied to log-probabilities.
// * ``HuberLoss``         — smooth L1 loss parameterised by ``delta``.
// * ``CTCLoss``           — Connectionist Temporal Classification for sequence
//                           transcription with variable-length targets.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Reduction strategy applied after the per-element loss has been computed.
//
// The enum mirrors the integer codes accepted by the public ``*_op`` entry
// points and the Python ``reduction=`` keyword convention used throughout
// ``lucid.nn``.
//
// Attributes
// ----------
// None : int
//     Code ``0``.  No reduction — return the full element-wise loss tensor
//     with the same shape as the input.
// Mean : int
//     Code ``1``.  Average over all (non-ignored) elements; returns a scalar.
// Sum : int
//     Code ``2``.  Sum over all elements; returns a scalar.
enum class Reduction : int { None = 0, Mean = 1, Sum = 2 };

// Autograd node for mean-squared-error loss.
//
// Computes
// $$
//   \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
// $$
// with ``Mean`` reduction; for ``Sum`` the leading $1/N$ factor is dropped, and
// for ``None`` the full per-element squared error $(\hat{y}_i - y_i)^2$ is
// returned without reduction.  Both ``input`` and ``target`` are saved so that
// the backward pass can re-emit
// $\nabla_{\hat y} = \frac{2}{N} (\hat y - y) \cdot g$ (or $2 (\hat y - y) g$
// for ``Sum``, or $2 (\hat y - y) \odot g$ for ``None``).
//
// Math
// ----
// $$
//   \mathcal{L}_{\text{mse}}(\hat{y}, y) = \begin{cases}
//     \frac{1}{N} \lVert \hat{y} - y \rVert_2^2 & \text{Mean} \\
//     \lVert \hat{y} - y \rVert_2^2             & \text{Sum} \\
//     (\hat{y} - y)^2                           & \text{None}
//   \end{cases}
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"mse_loss"``, ``AmpPolicy::ForceFP32``.
// reduction_ : Reduction
//     Reduction mode applied by the forward and unwound by the backward.
// orig_shape_ : Shape
//     Element-wise shape before reduction; needed to broadcast ``grad_out``
//     back to the input shape during ``apply``.
//
// Notes
// -----
// Gradient magnitude scales linearly with the residual, so MSE is sensitive
// to outliers.  Prefer ``HuberLoss`` when robustness matters.
class LUCID_API MseLossBackward : public FuncOp<MseLossBackward, 2> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    Shape orig_shape_;  // Element-wise shape before reduction.

    // Compute the MSE loss with autograd wiring.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     Predicted tensor of arbitrary shape.
    // target : TensorImplPtr
    //     Ground-truth tensor.  Must share ``input``'s shape and dtype.
    // reduction : Reduction
    //     ``Mean``, ``Sum``, or ``None``.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Scalar (Mean / Sum) or per-element (None) loss tensor.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``input.shape() != target.shape()``.
    // DtypeMismatch
    //     If ``input.dtype() != target.dtype()``.
    static TensorImplPtr
    forward(const TensorImplPtr& input, const TensorImplPtr& target, Reduction reduction);

    // Backward pass: emits gradients w.r.t. ``input`` and ``target``.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     ``{grad_input, grad_target}`` of the same shape as the originals.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for binary cross-entropy on probability inputs.
//
// Computes the element-wise loss
// $$
//   \ell(\hat{y}, y) = -w \,\bigl[\, y \log(\hat{y} + \varepsilon)
//                                 + (1 - y) \log(1 - \hat{y} + \varepsilon) \,\bigr]
// $$
// where $\hat{y} \in (0, 1)$ is a probability (e.g. the output of a sigmoid)
// and $y \in \{0, 1\}$ is a binary target.  The clamp ``eps_`` is added inside
// the logarithm to protect against $\log 0$.  Both ``input``, ``target`` and
// the (optional) per-element ``weight`` are saved so that the backward can
// emit $\nabla_{\hat y} = w \cdot \bigl(-y / \hat y + (1 - y) / (1 - \hat y)\bigr)
// \cdot g / N$ (for ``Mean``; analogous for the other modes).
//
// Math
// ----
// $$
//   \ell(\hat{y}, y) = -w \,\bigl[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\bigr]
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"bce_loss"``, ``AmpPolicy::ForceFP32``.
// reduction_ : Reduction
//     Reduction mode.
// eps_ : double
//     Numerical clamp added to ``log`` arguments.  Default ``1e-7``.
// orig_shape_ : Shape
//     Element-wise shape before reduction.
//
// Notes
// -----
// Inputs **must** be valid probabilities; values outside $(0, 1)$ produce
// ``NaN`` / ``inf`` losses.  For raw logits use ``BCEWithLogitsBackward``,
// which is numerically more stable.
class LUCID_API BCELossBackward : public FuncOp<BCELossBackward, 3> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double eps_ = 1e-7;  // Clamp added to log argument for numerical safety.
    Shape orig_shape_;

    // Compute BCE loss with autograd wiring.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     Probabilities in $(0, 1)$; arbitrary shape.
    // target : TensorImplPtr
    //     Binary labels in $\{0, 1\}$; same shape as ``input``.
    // weight : TensorImplPtr
    //     Per-element weighting tensor broadcastable to ``input.shape()``.
    //     Pass a ones tensor if no weighting is required.
    // reduction : Reduction
    //     ``Mean``, ``Sum``, or ``None``.
    // eps : double
    //     Clamp inside the logarithm; values around ``1e-7`` recommended.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Reduced or per-element loss tensor.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 Reduction reduction,
                                 double eps);

    // Backward pass: gradients w.r.t. ``input``, ``target``, and ``weight``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for sigmoid + binary cross-entropy fused for numerical stability.
//
// Equivalent to ``BCELoss(Sigmoid(x), y)`` but evaluated using the
// log-sum-exp identity $\log(1 + e^x) = \max(x, 0) + \log(1 + e^{-|x|})$
// so that no intermediate sigmoid value is required:
// $$
//   \ell(x, y) = \max(x, 0) - x \cdot y + \log\bigl(1 + e^{-|x|}\bigr)
// $$
// When ``pos_weight`` is supplied, positive examples are re-weighted as
// $\ell(x, y) = -p \cdot y \log \sigma(x) - (1 - y) \log(1 - \sigma(x))$
// where $\sigma$ is the sigmoid; the closed form above is rearranged
// internally to remain numerically stable.
//
// Math
// ----
// $$
//   \ell(x, y) = \max(x, 0) - x y + \log\bigl(1 + e^{-|x|}\bigr)
// $$
// with an additional positive-class scale $p$ applied to the $y \log \sigma(x)$
// term when ``pos_weight`` is non-trivial.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"bce_with_logits"``, ``AmpPolicy::ForceFP32``.
// reduction_ : Reduction
//     Reduction mode.
// orig_shape_ : Shape
//     Element-wise shape before reduction.
//
// Notes
// -----
// Always prefer this op over a manual ``Sigmoid`` → ``BCELoss`` chain: the
// fused form preserves gradient magnitude in the saturated regions of the
// sigmoid where the explicit decomposition produces vanishing gradients.
class LUCID_API BCEWithLogitsBackward : public FuncOp<BCEWithLogitsBackward, 4> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    Shape orig_shape_;

    // Compute BCE-with-logits loss with autograd wiring.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     Raw logits (any real value); arbitrary shape.
    // target : TensorImplPtr
    //     Binary labels in $\{0, 1\}$; same shape as ``input``.
    // weight : TensorImplPtr
    //     Per-element weighting; broadcastable to ``input.shape()``.
    // pos_weight : TensorImplPtr
    //     Positive-class re-weighting; shape broadcastable to the channel
    //     dimension of ``input``.  Pass a ones tensor if unused.
    // reduction : Reduction
    //     ``Mean``, ``Sum``, or ``None``.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Reduced or per-element loss tensor.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& pos_weight,
                                 Reduction reduction);

    // Backward pass: gradients w.r.t. ``input``, ``target``, ``weight``, ``pos_weight``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for multi-class cross-entropy loss (fused log-softmax + NLL).
//
// For a batch of $N$ samples with logits $x \in \mathbb{R}^{N \times C}$ and
// hard integer targets $y \in \{0, \ldots, C-1\}^{N}$ the loss is
// $$
//   \mathcal{L} = -\frac{1}{N'} \sum_{n: y_n \neq \text{ignore}}
//     w_{y_n} \log p_{n, y_n},
//   \quad p_{n, c} = \mathrm{softmax}(x_n)_c
// $$
// where $N'$ counts samples that are *not* masked by ``ignore_index`` and
// $w$ is the optional per-class weight vector.  The backend computes the
// softmax probabilities internally using the standard log-sum-exp
// stabilisation $p_{n, c} = e^{x_{n, c} - \max_c x_{n, c}} / Z_n$ and
// returns them in ``saved_softmax_`` for the backward pass, which emits
// $\nabla_x = (p - \mathbf{1}_{c = y}) \cdot w_y / N'$.
//
// Math
// ----
// $$
//   \mathcal{L}_{\text{ce}} = -\frac{1}{N'} \sum_{n: y_n \neq \text{ignore}}
//     w_{y_n} \log \frac{\exp(x_{n, y_n})}{\sum_{c=1}^{C} \exp(x_{n, c})}
// $$
// with the leading $1/N'$ factor present only for ``Reduction::Mean``.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"cross_entropy"``, ``AmpPolicy::ForceFP32``.
// reduction_ : Reduction
//     Reduction mode.
// eps_ : double
//     Numerical clamp passed to the backend (used for label-smoothing paths).
// ignore_index_ : int
//     Class index whose samples are excluded from both the loss numerator
//     and the denominator $N'$.  Default ``-100`` (matches the public API).
// has_weight_ : bool
//     ``true`` when ``saved_weight_`` holds a non-trivial per-class weight.
// orig_input_shape_ : Shape
//     Full logits shape ``(N, C, ...)`` for backward reconstruction.
// saved_softmax_ : Storage
//     Per-sample softmax probabilities returned by the backend.
// saved_target_ : Storage
//     Integer class labels.
// saved_weight_ : Storage
//     Optional per-class weight vector (length $C$).
// saved_valid_count_ : Storage
//     Scalar count of non-ignored samples used for the ``Mean`` denominator.
//
// Notes
// -----
// Only the input logits are wired as a saved autograd edge; the target,
// weight and softmax tensors are stored in the dedicated ``Storage`` fields
// because they are not differentiable inputs.
//
// References
// ----------
// Goodfellow, Bengio, Courville, "Deep Learning" §6.2.2 (2016).
class LUCID_API CrossEntropyBackward : public FuncOp<CrossEntropyBackward, 1> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double eps_ = 1e-7;
    int ignore_index_ = -100;  // Class index whose samples do not contribute.
    bool has_weight_ = false;
    Shape orig_input_shape_;
    Storage saved_softmax_;      // Per-sample softmax probabilities.
    Storage saved_target_;       // Integer class labels.
    Storage saved_weight_;       // Optional per-class weights.
    Storage saved_valid_count_;  // Count of non-ignored samples (for Mean).

    // Compute cross-entropy loss with autograd wiring.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     Logits of shape ``(N, C)`` or ``(N, C, d_1, ...)``.
    // target : TensorImplPtr
    //     Integer class labels of shape ``(N,)`` or ``(N, d_1, ...)``.
    // weight_or_null : TensorImplPtr
    //     Optional per-class weight vector of length $C$.  Pass ``nullptr``
    //     to disable class weighting.
    // reduction : Reduction
    //     ``Mean``, ``Sum``, or ``None``.
    // eps : double
    //     Numerical clamp passed to the backend.
    // ignore_index : int
    //     Class index excluded from the loss / gradient.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Scalar (Mean / Sum) or per-sample (None) loss tensor.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight_or_null,
                                 Reduction reduction,
                                 double eps,
                                 int ignore_index);

    // Backward pass: gradient w.r.t. logits ``input``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for negative log-likelihood loss.
//
// Operates on **log-probabilities** (e.g. the output of ``log_softmax``).
// The per-sample loss is
// $$
//   \ell_n = -w_{y_n} \, x_{n, y_n}
// $$
// and the final scalar is obtained by reducing $\ell$ according to
// ``reduction_``; samples whose target equals ``ignore_index_`` are skipped.
// The backward emits a sparse gradient that is nonzero only at the target
// class:
// $$
//   \nabla_x[n, c, \ldots] = \begin{cases}
//     -w_{y_n} \, g_n / N' & c = y_n \\
//     0                    & \text{otherwise}
//   \end{cases}
// $$
//
// Math
// ----
// $$
//   \mathcal{L}_{\text{nll}} = -\frac{1}{N'} \sum_{n: y_n \neq \text{ignore}}
//     w_{y_n} \, x_{n, y_n}
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"nll_loss"``, ``AmpPolicy::ForceFP32``.
// reduction_ : Reduction
//     Reduction mode.
// ignore_index_ : int
//     Target class to skip; default ``-100``.
// has_weight_ : bool
//     Whether ``saved_weight_`` is populated.
// orig_input_shape_ : Shape
//     Full input shape ``(N, C, ...)`` for backward reconstruction.
// saved_target_ : Storage
//     Integer class labels.
// saved_weight_ : Storage
//     Optional per-class weight vector.
// saved_valid_count_ : Storage
//     Scalar count of non-ignored samples for the ``Mean`` denominator.
//
// Notes
// -----
// Feeding raw logits or softmax probabilities produces incorrect results —
// the input *must* already be in log-probability space.  Prefer
// ``CrossEntropyBackward`` when starting from logits.
class LUCID_API NLLLossBackward : public FuncOp<NLLLossBackward, 1> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    int ignore_index_ = -100;
    bool has_weight_ = false;
    Shape orig_input_shape_;
    Storage saved_target_;
    Storage saved_weight_;
    Storage saved_valid_count_;

    // Compute NLL loss with autograd wiring.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     Log-probabilities of shape ``(N, C)`` or ``(N, C, d_1, ...)``.
    // target : TensorImplPtr
    //     Integer class labels of shape ``(N,)`` or ``(N, d_1, ...)``.
    // weight_or_null : TensorImplPtr
    //     Optional per-class weight vector of length $C$; may be null.
    // reduction : Reduction
    //     ``Mean``, ``Sum``, or ``None``.
    // ignore_index : int
    //     Class index excluded from the loss / gradient.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Reduced or per-sample loss tensor.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight_or_null,
                                 Reduction reduction,
                                 int ignore_index);

    // Backward pass: gradient w.r.t. log-probability ``input``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for Huber (smooth L1) loss.
//
// Combines a quadratic regime for small residuals with a linear regime for
// large residuals, providing robustness to outliers without sacrificing
// differentiability at zero:
// $$
//   \ell_\delta(\hat{y}, y) = \begin{cases}
//     \tfrac{1}{2} (\hat{y} - y)^2                            & |\hat{y} - y| \le \delta \\
//     \delta \bigl(|\hat{y} - y| - \tfrac{1}{2} \delta\bigr)  & |\hat{y} - y| > \delta
//   \end{cases}
// $$
// The backward emits $\nabla_{\hat y} = \operatorname{clamp}(\hat y - y, -\delta, \delta)
// \cdot g$ (scaled by $1/N$ for ``Mean``).
//
// Math
// ----
// $$
//   \ell_\delta(r) = \begin{cases}
//     \tfrac{1}{2} r^2          & |r| \le \delta \\
//     \delta |r| - \tfrac{\delta^2}{2} & \text{otherwise}
//   \end{cases}, \quad r = \hat y - y
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"huber_loss"``, ``AmpPolicy::ForceFP32``.
// reduction_ : Reduction
//     Reduction mode.
// delta_ : double
//     Transition threshold between quadratic and linear regimes;
//     must be strictly positive.  Default ``1.0`` (SmoothL1).
// orig_shape_ : Shape
//     Element-wise shape before reduction.
//
// Notes
// -----
// Smaller $\delta$ behaves more like an absolute-error loss (robust to
// outliers); larger $\delta$ behaves more like MSE (sensitive to outliers).
class LUCID_API HuberLossBackward : public FuncOp<HuberLossBackward, 2> {
public:
    static const OpSchema schema_v1;
    Reduction reduction_ = Reduction::Mean;
    double delta_ = 1.0;
    Shape orig_shape_;

    // Compute the Huber loss with autograd wiring.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     Predicted tensor.
    // target : TensorImplPtr
    //     Ground-truth tensor of the same shape as ``input``.
    // delta : double
    //     Transition threshold; must be ``> 0``.
    // reduction : Reduction
    //     ``Mean``, ``Sum``, or ``None``.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Reduced or per-element loss tensor.
    //
    // Raises
    // ------
    // std::invalid_argument
    //     If ``delta <= 0``.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 double delta,
                                 Reduction reduction);

    // Backward pass: gradients w.r.t. ``input`` and ``target``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public mean-squared-error entry point.
//
// Thin wrapper that casts ``reduction`` to ``Reduction`` and delegates to
// ``MseLossBackward::forward``.
//
// Parameters
// ----------
// input : TensorImplPtr
//     Predicted tensor.
// target : TensorImplPtr
//     Ground-truth tensor of the same shape and dtype.
// reduction : int
//     ``0`` (None), ``1`` (Mean), or ``2`` (Sum).
//
// Returns
// -------
// TensorImplPtr
//     Scalar (Mean / Sum) or per-element (None) loss.
LUCID_API TensorImplPtr mse_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    int reduction);

// Public binary cross-entropy entry point.
//
// Parameters
// ----------
// input : TensorImplPtr
//     Probabilities in $(0, 1)$.
// target : TensorImplPtr
//     Binary targets in $\{0, 1\}$ of the same shape.
// weight : TensorImplPtr
//     Per-element weighting tensor; pass a ones tensor when unused.
// reduction : int
//     ``0`` (None), ``1`` (Mean), or ``2`` (Sum).
// eps : double
//     Clamp inside the logarithm to keep gradients finite.
//
// Returns
// -------
// TensorImplPtr
//     Reduced or per-element loss tensor.
LUCID_API TensorImplPtr bce_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    const TensorImplPtr& weight,
                                    int reduction,
                                    double eps);

// Public BCE-with-logits entry point.
//
// Parameters
// ----------
// input : TensorImplPtr
//     Raw logits (any real value).
// target : TensorImplPtr
//     Binary targets in $\{0, 1\}$ of the same shape.
// weight : TensorImplPtr
//     Per-element weighting tensor.
// pos_weight : TensorImplPtr
//     Positive-class re-weighting; pass a ones tensor when unused.
// reduction : int
//     ``0`` (None), ``1`` (Mean), or ``2`` (Sum).
//
// Returns
// -------
// TensorImplPtr
//     Reduced or per-element loss tensor.
LUCID_API TensorImplPtr bce_with_logits_op(const TensorImplPtr& input,
                                           const TensorImplPtr& target,
                                           const TensorImplPtr& weight,
                                           const TensorImplPtr& pos_weight,
                                           int reduction);

// Public cross-entropy (fused log-softmax + NLL) entry point.
//
// Parameters
// ----------
// input : TensorImplPtr
//     Logits of shape ``(N, C)`` or ``(N, C, d_1, ...)``.
// target : TensorImplPtr
//     Integer class labels of shape ``(N,)`` or ``(N, d_1, ...)``.
// weight_or_null : TensorImplPtr
//     Optional per-class weight vector of length $C$; pass null when unused.
// reduction : int
//     ``0`` (None), ``1`` (Mean), or ``2`` (Sum).
// eps : double
//     Numerical clamp passed through to the backend.
// ignore_index : int
//     Class index excluded from the loss and gradient.
//
// Returns
// -------
// TensorImplPtr
//     Scalar (Mean / Sum) or per-sample (None) loss tensor.
LUCID_API TensorImplPtr cross_entropy_op(const TensorImplPtr& input,
                                         const TensorImplPtr& target,
                                         const TensorImplPtr& weight_or_null,
                                         int reduction,
                                         double eps,
                                         int ignore_index);

// Public negative log-likelihood entry point.
//
// Parameters
// ----------
// input : TensorImplPtr
//     Log-probabilities of shape ``(N, C)`` or ``(N, C, d_1, ...)``.
// target : TensorImplPtr
//     Integer class labels.
// weight_or_null : TensorImplPtr
//     Optional per-class weight vector of length $C$; pass null when unused.
// reduction : int
//     ``0`` (None), ``1`` (Mean), or ``2`` (Sum).
// ignore_index : int
//     Class index excluded from the loss / gradient.
//
// Returns
// -------
// TensorImplPtr
//     Reduced or per-sample loss tensor.
LUCID_API TensorImplPtr nll_loss_op(const TensorImplPtr& input,
                                    const TensorImplPtr& target,
                                    const TensorImplPtr& weight_or_null,
                                    int reduction,
                                    int ignore_index);

// Public Huber (smooth L1) loss entry point.
//
// Parameters
// ----------
// input : TensorImplPtr
//     Predicted tensor.
// target : TensorImplPtr
//     Ground-truth tensor of the same shape.
// delta : double
//     Quadratic / linear transition threshold; must be ``> 0``.
// reduction : int
//     ``0`` (None), ``1`` (Mean), or ``2`` (Sum).
//
// Returns
// -------
// TensorImplPtr
//     Reduced or per-element loss tensor.
LUCID_API TensorImplPtr huber_loss_op(const TensorImplPtr& input,
                                      const TensorImplPtr& target,
                                      double delta,
                                      int reduction);

// Connectionist Temporal Classification (CTC) loss.
//
// Computes the negative log-probability of all valid alignments between an
// input log-probability sequence and a flat target label sequence using the
// forward algorithm (Graves et al., 2006).  The blank symbol allows the model
// to emit "no character" at any time step, and the loss is summed over all
// valid label expansions:
// $$
//   \mathcal{L} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)}
//     \prod_{t=1}^{T} p_{\pi_t}^{(t)}
// $$
// where $\mathcal{B}$ is the standard CTC collapse operation.  No reduction
// is applied here — the caller is expected to apply ``mean`` / ``sum`` in
// Python.
//
// Parameters
// ----------
// log_probs : TensorImplPtr
//     Log-probability tensor of shape ``(T, N, C)``.
// targets : TensorImplPtr
//     Flat ``int32`` label tensor of shape ``(N * S,)`` where $S$ is the
//     maximum target length.
// input_lengths : TensorImplPtr
//     ``int32`` vector of shape ``(N,)`` giving the valid time-step count
//     per sample.
// target_lengths : TensorImplPtr
//     ``int32`` vector of shape ``(N,)`` giving the true label length per
//     sample.
// blank : int
//     Index of the CTC blank class.
// zero_infinity : bool
//     When ``true``, infinite loss values (alignments with zero probability)
//     are clamped to zero — both forward and backward.
//
// Returns
// -------
// TensorImplPtr
//     Per-sample loss tensor of shape ``(N,)``.
//
// References
// ----------
// Graves et al., "Connectionist Temporal Classification: Labelling
// Unsegmented Sequence Data with Recurrent Neural Networks" (ICML 2006).
LUCID_API TensorImplPtr ctc_loss_op(const TensorImplPtr& log_probs,
                                    const TensorImplPtr& targets,
                                    const TensorImplPtr& input_lengths,
                                    const TensorImplPtr& target_lengths,
                                    int blank,
                                    bool zero_infinity);

}  // namespace lucid
