// lucid/_C/nn/Dropout.h
//
// Five stochastic regularization operators that share a common backward
// shape — multiply the incoming gradient by the saved Bernoulli mask —
// but differ in how the mask is sampled and shaped:
//
//   * ``Dropout``      — element-wise inverted dropout.  Independent
//                        Bernoulli per element, surviving values rescaled
//                        by $1 / (1 - p)$ so the expectation of every
//                        element is preserved.
//   * ``DropoutNd``    — channel-wise dropout.  One Bernoulli draw per
//                        ``(batch, channel)`` pair, broadcast over all
//                        spatial axes — used for 1-D / 2-D / 3-D
//                        convolutional feature maps where adjacent
//                        positions are highly correlated.
//   * ``AlphaDropout`` — SELU-preserving dropout.  Dropped elements are
//                        replaced with the SELU saturation value
//                        $\alpha' = -\lambda\alpha \approx -1.7581$ and
//                        an affine rescaling restores zero mean and unit
//                        variance after masking, keeping a deep
//                        self-normalizing network on its fixed point.
//   * ``DropBlock``    — structured spatial dropout (Ghiasi 2018).  A
//                        sparse seed mask is dilated into contiguous
//                        $\text{block\_size}\!\times\!\text{block\_size}$
//                        zero patches on 4-D inputs.
//   * ``DropPath``     — stochastic depth (Huang 2016).  Drops entire
//                        per-sample residual paths; surviving samples
//                        are optionally rescaled by $1 / (1 - p)$.
//
// All five obey the ``set_deterministic(true)`` policy: when active, an
// explicit :class:`Generator` must be passed; otherwise
// ``check_schema_determinism`` raises.  When ``training == false`` or
// ``p == 0`` the forward is a pass-through (clones the input) and the
// backward simply copies ``grad_out``.

#pragma once

#include <utility>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class Generator;

// Autograd node for standard element-wise (inverted) dropout.
//
// During training, each scalar element of the input is independently
// masked with probability ``p`` and surviving elements are rescaled by
// $1 / (1 - p)$.  The *already-scaled* mask is saved in :member:`mask_`
// so the backward needs only one multiply — no extra scaling step.
// When ``p == 0`` or ``training == false`` the node is wired with an
// empty mask and :func:`apply` clones ``grad_out`` unchanged.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (name ``"dropout"``, version 1,
//     ``AmpPolicy::KeepInput``).  Marked non-deterministic because the
//     mask depends on the Generator state.
// p_ : double
//     Drop probability used by the forward; stored for the backward.
//     ``0`` indicates pass-through.
// mask_ : Storage
//     The scaled Bernoulli mask $m \cdot 1/(1 - p)$ saved for backward.
//     Empty when ``p_ == 0``.
class LUCID_API DropoutBackward : public FuncOp<DropoutBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;  // Drop probability (0 = pass-through).
    Storage mask_;    // Scaled Bernoulli mask saved for backward.

    // Sample the Bernoulli mask and apply inverted-dropout scaling.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input tensor of any shape.
    // p : double
    //     Drop probability in $[0, 1)$.  Must be strictly less than 1
    //     (probability 1 would zero everything and produce NaN under
    //     the $1 / (1 - p)$ scaling).
    // training : bool
    //     If ``false``, the forward is a pass-through.
    // gen : Generator*
    //     Optional explicit RNG.  ``nullptr`` selects the process-wide
    //     ``default_generator()``; under deterministic mode the call
    //     raises if ``gen == nullptr`` and ``p > 0``.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor of the same shape and dtype as ``a``.
    //
    // Math
    // ----
    // $$y_i = \begin{cases}
    //     \dfrac{x_i}{1 - p} & \text{with probability } 1 - p\\[4pt]
    //     0 & \text{with probability } p
    //   \end{cases}$$
    //
    // Raises
    // ------
    // LucidError
    //     If ``p`` is outside $[0, 1)$, or under deterministic mode if
    //     no Generator is provided.
    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);

    // Apply the saved mask to ``grad_out``.
    //
    // Returns ``{grad_out * mask_}`` when a mask is stored; otherwise
    // (pass-through case) returns ``{clone(grad_out)}``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for channel-wise Dropout.
//
// A single Bernoulli value is drawn per ``(batch, channel)`` pair on a
// compact $(N, C, 1, 1, \ldots)$ mask, then expanded by the backend
// to the full input shape so that entire feature-map channels are
// zeroed together.  This is the C++ kernel behind the Python
// ``Dropout1d`` / ``Dropout2d`` / ``Dropout3d`` modules — input rank
// must be at least 2 and the layout is assumed to be
// $(N, C, \text{spatial}\ldots)$.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (name ``"dropoutnd"``, version 1,
//     ``AmpPolicy::KeepInput``).
// p_ : double
//     Drop probability used by the forward; stored for the backward.
// mask_ : Storage
//     Full-resolution mask after the broadcast — same shape as the
//     input.  Saved so the backward is a single multiply.
class LUCID_API DropoutNdBackward : public FuncOp<DropoutNdBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;  // Full-resolution mask after broadcast (same shape as input).

    // Sample a channel mask and broadcast-multiply with the input.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input tensor of rank $\geq 2$ in layout
    //     $(N, C, \text{spatial}\ldots)$.
    // p : double
    //     Channel drop probability in $[0, 1)$.
    // training : bool
    //     If ``false``, the forward is a pass-through.
    // gen : Generator*
    //     Optional RNG (see :class:`DropoutBackward`).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor with the same shape and dtype as ``a``.
    //
    // Math
    // ----
    // $$y_{n,c,\ldots} = \begin{cases}
    //     \dfrac{x_{n,c,\ldots}}{1 - p} & \text{with probability } 1 - p\\[4pt]
    //     \mathbf{0} & \text{with probability } p
    //   \end{cases}$$
    //
    // Raises
    // ------
    // LucidError
    //     If ``a`` has fewer than 2 dimensions, ``p`` is outside
    //     $[0, 1)$, or deterministic mode is on without a Generator.
    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);

    // Backward: multiply ``grad_out`` by the saved full-resolution mask.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for alpha-dropout — element-wise dropout that preserves
// the self-normalising statistics required by SELU activations.
//
// Dropped elements are not zeroed but replaced with the SELU
// saturation value $\alpha' = -\lambda\alpha \approx -1.7581$;
// an affine rescaling $y \mapsto a \cdot y + b$ is then applied so
// that the mean and variance of the post-dropout distribution match
// those of the input.  The chain rule therefore collapses to
// $\partial y / \partial x = a \cdot \text{mask}$, which is what
// :func:`apply` evaluates using the saved raw mask.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (name ``"alpha_dropout"``, version 1,
//     ``AmpPolicy::KeepInput``).
// p_ : double
//     Drop probability used by the forward.
// a_coef_ : double
//     Affine scale factor $a = \bigl(\text{keep}\,(1 + p\,\alpha'^2)\bigr)^{-1/2}$
//     saved for the backward.
// mask_ : Storage
//     Raw $\{0, 1\}$ Bernoulli mask before any scaling.
class LUCID_API AlphaDropoutBackward : public FuncOp<AlphaDropoutBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    double a_coef_ = 1.0;  // Affine scale factor saved for backward.
    Storage mask_;         // Raw Bernoulli mask (0/1, before scaling).

    // Apply alpha dropout with SELU-preserving affine correction.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input tensor of any shape.  Expected to follow a SELU
    //     activation; other activations break the statistical
    //     invariant the affine correction restores.
    // p : double
    //     Drop probability in $[0, 1)$.
    // training : bool
    //     If ``false``, the forward is a pass-through.
    // gen : Generator*
    //     Optional RNG (see :class:`DropoutBackward`).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor with the same shape and dtype as ``a``.
    //
    // Math
    // ----
    // $$y_i = a\,\bigl(x_i\,m_i + \alpha'(1 - m_i)\bigr) + b,
    //   \quad m_i \sim \text{Bernoulli}(1 - p),$$
    //
    // with $a = \bigl(\text{keep}\,(1 + p\,\alpha'^2)\bigr)^{-1/2}$
    // and $b = -a\,p\,\alpha'$.
    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);

    // Backward: $\nabla_x \ell = a \cdot m \cdot \text{grad\_out}$.
    //
    // Returns ``{grad_out * (mask_ * a_coef_)}`` when a mask is stored;
    // otherwise (pass-through case) returns ``{clone(grad_out)}``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for DropBlock — structured spatial dropout for 4-D
// convolutional feature maps (Ghiasi, Lin & Le 2018).
//
// A sparse seed mask of Bernoulli samples is generated at the
// per-element rate
// $\gamma = p \cdot \dfrac{H W}{B^2 (H - B + 1)(W - B + 1) + \varepsilon}$
// (with $B$ = ``block_size``) and then dilated by the backend into
// contiguous $B \times B$ keep blocks so that whole spatial patches
// are zeroed together rather than uncorrelated single pixels.  This is
// far more effective than element-wise dropout on convolutional
// outputs where neighbouring pixels are strongly correlated.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (name ``"drop_block"``, version 1,
//     ``AmpPolicy::KeepInput``).
// p_ : double
//     Target block-level drop rate.
// mask_ : Storage
//     The dilated $\{0, 1\}$ keep mask, same shape as the input,
//     saved for the backward multiply.
class LUCID_API DropBlockBackward : public FuncOp<DropBlockBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;  // Keep mask after dilation (0/1 float, same shape as input).

    // Sample the seed mask, dilate into blocks, and apply.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     4-D input tensor of shape ``(N, C, H, W)``.
    // block_size : std::int64_t
    //     Side length of each dropped block.  Must be strictly positive.
    // p : double
    //     Target block-level drop probability in $[0, 1)$.
    // eps : double
    //     Numerical guard added to the denominator when computing
    //     $\gamma$ to avoid division by zero on small spatial extents.
    // gen : Generator*
    //     Optional RNG (see :class:`DropoutBackward`).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor with the same shape as ``a``.
    //
    // Raises
    // ------
    // LucidError
    //     If ``a`` is not 4-D, ``p`` is outside $[0, 1)$,
    //     ``block_size <= 0``, or deterministic mode is on without a
    //     Generator.
    static TensorImplPtr
    forward(const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen);

    // Backward: multiply ``grad_out`` by the saved keep mask.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for DropPath — per-sample stochastic depth (Huang,
// Sun, Liu & Weinberger 2016) used in residual / transformer blocks.
//
// A single Bernoulli value is drawn per sample (mask shape
// $(N, 1, 1, \ldots)$) and broadcast over every other dimension, so
// either the entire residual contribution of a sample survives or it
// is killed entirely.  When ``scale_by_keep`` is true the surviving
// samples are pre-scaled by $1 / (1 - p)$ so that the *expectation*
// of the residual branch matches a deterministic forward.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (name ``"drop_path"``, version 1,
//     ``AmpPolicy::KeepInput``).
// mask_ : Storage
//     Per-sample mask broadcast to the full input shape; saved for
//     the backward.
class LUCID_API DropPathBackward : public FuncOp<DropPathBackward, 1> {
public:
    static const OpSchema schema_v1;
    Storage mask_;  // Per-sample mask broadcast to full input shape.

    // Apply per-sample stochastic depth.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input tensor of any rank $\geq 1$; the first axis is the
    //     batch axis.
    // p : double
    //     Drop probability in $[0, 1)$.
    // scale_by_keep : bool
    //     If ``true``, surviving samples are multiplied by
    //     $1 / (1 - p)$ to preserve the expectation of the residual
    //     branch.
    // gen : Generator*
    //     Optional RNG (see :class:`DropoutBackward`).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor with the same shape as ``a``.
    //
    // Raises
    // ------
    // LucidError
    //     If ``p`` is outside $[0, 1)$ or deterministic mode is on
    //     without a Generator.
    static TensorImplPtr
    forward(const TensorImplPtr& a, double p, bool scale_by_keep, Generator* gen);

    // Backward: multiply ``grad_out`` by the saved per-sample mask.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Element-wise inverted dropout — free-function entry point.
//
// Thin wrapper that delegates to :func:`DropoutBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.
// p : double
//     Drop probability in $[0, 1)$.
// training : bool
//     If ``false``, the forward is a pass-through.
// gen : Generator*
//     Optional RNG; ``nullptr`` selects ``default_generator()``.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape as ``a``.
LUCID_API TensorImplPtr dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen);

// Stateful inverted dropout for the compile path — sibling of
// :func:`dropout_op` that explicitly threads an MPSGraph-Philox state
// tensor as a second input and returns ``(y, state_out)``.
//
// In **eager mode** the function delegates to :func:`dropout_op` for
// the actual masking (so the result distribution matches the standard
// dropout op identically) and returns a clone of ``state_in`` as the
// state-out tensor — Lucid's eager :class:`Generator` already
// advances per call, so the state buffer is purely a placeholder
// that exists for the trace recording.  Suppresses any nested tracer
// recording from the inner ``dropout_op`` call so the trace contains
// exactly one ``"dropout_stateful"`` op node (not a nested ``"dropout"``
// underneath it).
//
// In the **compile path** the matching MPSGraph emitter consumes the
// captured op node by calling
// ``randomTensorWithShape:descriptor:stateTensor:`` — the 2-output
// stateful Philox API — using ``state_in`` to seed the RNG and binding
// the new state to ``state_out``.  Across dispatches the state buffer
// is rotated by the executable (either as an in/out feed pair or
// promoted to an MPSGraph variable via
// :func:`compile_generic_fused_step_with_vars`), giving genuinely
// per-dispatch varying masks where the stateless seed-only path
// produces dispatch-deterministic ones.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input activations of any shape.
// state_in : TensorImplPtr
//     Philox-4x32 state tensor (``int32[7]``).  Initial values are
//     derived from Lucid's :class:`Generator` on the Python side
//     (one fresh state per dropout call site at trace time); the
//     compile path mutates the buffer in-place across dispatches.
// p : double
//     Drop probability in $[0, 1)$.
// training : bool
//     When ``false``, the eager forward is a pass-through identity
//     (the trace still records the op so the compile path can decide
//     whether to emit the masked or identity branch).
// gen : Generator*
//     Optional explicit RNG used by the eager fallback; ``nullptr``
//     selects ``default_generator()``.  Ignored by the compile path.
//
// Returns
// -------
// std::pair<TensorImplPtr, TensorImplPtr>
//     ``(y, state_out)`` — masked output of the same shape and dtype
//     as ``x``; ``state_out`` is the same shape/dtype as ``state_in``
//     and (in eager mode) carries a verbatim clone of its contents.
LUCID_API std::pair<TensorImplPtr, TensorImplPtr> dropout_stateful_op(
    const TensorImplPtr& x, const TensorImplPtr& state_in, double p, bool training, Generator* gen);

// Channel-wise dropout — free-function entry point.
//
// Thin wrapper that delegates to :func:`DropoutNdBackward::forward`.
// Handles 3-D, 4-D, and 5-D inputs uniformly (assumes the layout
// $(N, C, \text{spatial}\ldots)$).
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of rank $\geq 2$ in layout $(N, C, \ldots)$.
// p : double
//     Channel drop probability in $[0, 1)$.
// training : bool
//     If ``false``, the forward is a pass-through.
// gen : Generator*
//     Optional RNG; ``nullptr`` selects ``default_generator()``.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape as ``a``.
LUCID_API TensorImplPtr dropoutnd_op(const TensorImplPtr& a,
                                     double p,
                                     bool training,
                                     Generator* gen);

// SELU-preserving alpha dropout — free-function entry point.
//
// Thin wrapper that delegates to :func:`AlphaDropoutBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor (typically the output of a SELU activation).
// p : double
//     Drop probability in $[0, 1)$.
// training : bool
//     If ``false``, the forward is a pass-through.
// gen : Generator*
//     Optional RNG; ``nullptr`` selects ``default_generator()``.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape as ``a``.
LUCID_API TensorImplPtr alpha_dropout_op(const TensorImplPtr& a,
                                         double p,
                                         bool training,
                                         Generator* gen);

// DropBlock — structured spatial dropout — free-function entry point.
//
// Thin wrapper that delegates to :func:`DropBlockBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     4-D input tensor of shape ``(N, C, H, W)``.
// block_size : std::int64_t
//     Side length of each dropped spatial block.
// p : double
//     Target block-level drop probability in $[0, 1)$.
// eps : double
//     Numerical guard for the $\gamma$ denominator.
// gen : Generator*
//     Optional RNG; ``nullptr`` selects ``default_generator()``.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape as ``a``.
LUCID_API TensorImplPtr drop_block_op(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen);

// DropPath — per-sample stochastic depth — free-function entry point.
//
// Thin wrapper that delegates to :func:`DropPathBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of rank $\geq 1$; first axis is the batch axis.
// p : double
//     Per-sample drop probability in $[0, 1)$.
// scale_by_keep : bool
//     Whether to rescale surviving samples by $1 / (1 - p)$.
// gen : Generator*
//     Optional RNG; ``nullptr`` selects ``default_generator()``.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape as ``a``.
LUCID_API TensorImplPtr drop_path_op(const TensorImplPtr& a,
                                     double p,
                                     bool scale_by_keep,
                                     Generator* gen);

}  // namespace lucid
