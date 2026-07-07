// lucid/_C/nn/Attention.h
//
// Autograd-aware Scaled Dot-Product Attention (SDPA).
//
// Computes:  out = softmax(Q @ K^T * scale + mask) @ V
//
// Backend dispatch:
//   GPU – calls mlx::core::fast::scaled_dot_product_attention, which returns
//         a dummy {1}-shape weights tensor.  The backward detects ndim==1 and
//         recomputes W = softmax(Q @ K^T * scale) before computing gradients.
//   CPU – rolls the full softmax(Q @ K^T / sqrt(d)) @ V computation by hand.
//
// Two public entry points are provided:
//   scaled_dot_product_attention_op       – returns output only.
//   scaled_dot_product_attention_with_weights_op – returns {output, weights}.

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for scaled dot-product attention with optional additive
// mask and post-softmax causal constraint.
//
// Implements the canonical SDPA operation from Vaswani et al. (2017),
// $\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$,
// where $Q, K, V$ are query / key / value projections shaped
// ``(..., L, D)`` and $M$ is an additive mask (``-inf`` at disallowed
// positions, ``0`` elsewhere).  The node saves three edges
// ``{Q, K, V}`` and a fourth Storage slot ``saved_weights_`` holding the
// pre-multiplication attention weight matrix
// $W = \text{softmax}(QK^\top \cdot s)$, where ``s = scale_``.
//
// On the MLX (GPU) path the fused ``mlx::core::fast::scaled_dot_product_attention``
// kernel does not return $W$ — instead it returns a ``{1}``-shape
// placeholder.  ``apply()`` detects ``saved_weights_.shape().size() == 1``
// and re-computes $W = \text{softmax}(QK^\top \cdot s)$ from the saved
// $Q, K$ before back-propagating.  On the Accelerate (CPU) path the
// kernel returns the genuine weight matrix and the recompute branch is
// skipped.
//
// Math
// ----
// $$
//   A_{ij} = s \cdot (Q_i \cdot K_j), \quad
//   P_{ij} = \frac{\exp(A_{ij} + M_{ij})}{\sum_k \exp(A_{ik} + M_{ik})}, \quad
//   y_i = \sum_j P_{ij}\, V_j
// $$
// Gradients flow back through the softmax Jacobian
// $\partial P / \partial A = \mathrm{diag}(P) - P P^\top$ (per row),
// giving
// $$
//   \frac{\partial \mathcal{L}}{\partial Q} = s \cdot (\Delta A)\, K,\quad
//   \frac{\partial \mathcal{L}}{\partial K} = s \cdot (\Delta A)^\top Q,\quad
//   \frac{\partial \mathcal{L}}{\partial V} = P^\top\, \frac{\partial \mathcal{L}}{\partial y}
// $$
// where $\Delta A_{ij} = P_{ij}\bigl(\partial \mathcal{L}/\partial y_i \cdot V_j^\top - \sum_k P_{ik} (\partial \mathcal{L}/\partial y_i \cdot V_k^\top)\bigr)$.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Op schema (name ``"scaled_dot_product_attention"``, version 1,
//     ``AmpPolicy::ForceFP32``, autograd-aware).  AMP forces the kernel
//     to FP32 because the softmax-exp accumulation is numerically
//     unstable at half precision.
// scale_ : double
//     Dot-product scale factor ``s`` (typically $1/\sqrt{d_k}$) multiplied
//     into $QK^\top$ before the softmax.  Default: ``1.0``.
// orig_q_shape_ : Shape
//     Original (unflattened) shape of ``Q`` recorded at forward; used by
//     ``apply()`` to restore leading batch dims on the gradient tensor.
// orig_k_shape_ : Shape
//     Original shape of ``K``.
// orig_v_shape_ : Shape
//     Original shape of ``V``.
// saved_weights_ : Storage
//     Attention weight matrix ``W``.  Genuine on the CPU path; on the GPU
//     path this is a ``{1}``-shape placeholder and the backward recomputes
//     ``W`` from the saved ``Q``, ``K`` and ``scale_``.
//
// Notes
// -----
// The mask is *additive*: callers must already have converted boolean
// "where to mask" patterns into floats containing ``-inf`` where the
// position should be excluded and ``0`` elsewhere (see
// :func:`lucid.nn.modules.attention._to_additive_mask`).  Causal masking
// is implemented engine-side when ``is_causal=true`` by adding an
// upper-triangular ``-inf`` matrix before the softmax, so the caller
// does not need to materialise the mask in Python.
//
// References
// ----------
// Vaswani et al., "Attention Is All You Need" (NeurIPS 2017).
class LUCID_API ScaledDotProductAttentionBackward
    : public FuncOp<ScaledDotProductAttentionBackward, 3> {
public:
    static const OpSchema schema_v1;
    double scale_ = 1.0;  // Dot-product scale factor (usually 1/sqrt(d_k)).
    Shape orig_q_shape_;
    Shape orig_k_shape_;
    Shape orig_v_shape_;

    // Attention weight matrix; may be a {1} placeholder on the GPU path.
    Storage saved_weights_;

    // Causal flag and additive/keep mask recorded at forward so the GPU VJP
    // backward can reproduce the exact masked attention (the fused kernel saves
    // no weights, so the mask cannot be recovered from ``saved_weights_``).
    bool is_causal_ = false;
    bool has_mask_ = false;
    Dtype mask_dtype_ = Dtype::F32;
    Storage saved_mask_;  // valid iff has_mask_; Bool ⇒ keep-mask, else additive.

    // Run the SDPA forward and return the output tensor.
    //
    // Dispatches to ``IBackend::sdpa_forward`` after validating that the
    // leading batch dims of ``q``, ``k`` and ``v`` agree, that
    // ``q.shape[-1] == k.shape[-1]`` (the dot-product key dim ``d_k``),
    // and that ``k.shape[-2] == v.shape[-2]`` (the source-sequence
    // length ``L_k``).  On the autograd-recording path this also
    // attaches a new ``ScaledDotProductAttentionBackward`` node with
    // ``saved_weights_`` populated.
    //
    // Parameters
    // ----------
    // q : TensorImplPtr
    //     Query tensor of shape ``(..., L_q, d_k)`` with at least 2
    //     dimensions.  All leading "batch" dims must broadcast with the
    //     corresponding dims of ``k`` and ``v``.
    // k : TensorImplPtr
    //     Key tensor of shape ``(..., L_k, d_k)``; the last dim must
    //     equal ``q.shape[-1]``.
    // v : TensorImplPtr
    //     Value tensor of shape ``(..., L_k, d_v)``; the penultimate dim
    //     must equal ``k.shape[-2]``.
    // attn_mask_or_null : TensorImplPtr
    //     Optional additive mask broadcastable to ``(..., L_q, L_k)``.
    //     Pass ``nullptr`` to omit (no additive term is materialised).
    // scale : double
    //     Multiplicative factor applied to ``Q @ K^T`` before the
    //     softmax.  Typically $1/\sqrt{d_k}$.
    // is_causal : bool
    //     If ``true``, an upper-triangular ``-inf`` mask is added
    //     engine-side, restricting position ``i`` to attend only to
    //     positions ``j <= i``.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Attention output of shape ``(..., L_q, d_v)``.
    //
    // Raises
    // ------
    // ErrorBuilder
    //     If any input is null, devices or dtypes disagree, rank is less
    //     than 2, or the shape contracts above are violated.
    static TensorImplPtr forward(const TensorImplPtr& q,
                                 const TensorImplPtr& k,
                                 const TensorImplPtr& v,
                                 const TensorImplPtr& attn_mask_or_null,
                                 double scale,
                                 bool is_causal);

    // Compute gradients with respect to ``Q``, ``K`` and ``V`` from the
    // upstream gradient of the output.
    //
    // Reconstructs the saved weight matrix on the GPU path (where
    // ``saved_weights_`` is a ``{1}``-shape placeholder) by recomputing
    // $W = \text{softmax}(QK^\top \cdot s)$, then dispatches to
    // ``IBackend::sdpa_backward`` for the three input gradients.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient $\partial \mathcal{L}/\partial y$ shaped like
    //     the SDPA output ``(..., L_q, d_v)``.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Three Storage objects in edge order ``{dQ, dK, dV}``, each
    //     reshaped back to ``orig_q_shape_`` / ``orig_k_shape_`` /
    //     ``orig_v_shape_``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Run scaled dot-product attention and return only the output tensor.
//
// Public entry point that wires Lucid's autograd system to the SDPA
// kernel.  Internally calls ``ScaledDotProductAttentionBackward::forward``
// and, when ``GradMode::is_enabled()`` and any of ``q``, ``k``, ``v``
// require a gradient, attaches a fresh ``ScaledDotProductAttentionBackward``
// node so the output participates in BPTT.
//
// Math
// ----
// $\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$.
//
// Parameters
// ----------
// q : TensorImplPtr
//     Query tensor of shape ``(..., L_q, d_k)``.
// k : TensorImplPtr
//     Key tensor of shape ``(..., L_k, d_k)``.
// v : TensorImplPtr
//     Value tensor of shape ``(..., L_k, d_v)``.
// attn_mask_or_null : TensorImplPtr
//     Optional additive mask broadcastable to ``(..., L_q, L_k)``;
//     ``nullptr`` skips the additive term entirely.
// scale : double
//     Multiplicative factor applied to $QK^\top$ before the softmax.
// is_causal : bool
//     If ``true``, an engine-side upper-triangular ``-inf`` mask is
//     added so each query position attends only to earlier keys.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of shape ``(..., L_q, d_v)``.
//
// See Also
// --------
// scaled_dot_product_attention_with_weights_op : Variant that also
//     returns the post-softmax attention weight matrix ``W``.
//
// References
// ----------
// Vaswani et al., "Attention Is All You Need" (NeurIPS 2017).
LUCID_API TensorImplPtr scaled_dot_product_attention_op(const TensorImplPtr& q,
                                                        const TensorImplPtr& k,
                                                        const TensorImplPtr& v,
                                                        const TensorImplPtr& attn_mask_or_null,
                                                        double scale,
                                                        bool is_causal);

// Run scaled dot-product attention and return ``{output, weights}``.
//
// Identical numerics to ``scaled_dot_product_attention_op`` but exposes
// the post-softmax attention weight matrix
// $W = \text{softmax}(QK^\top \cdot s + M)$ shaped ``(..., L_q, L_k)``
// as a second return value.  Useful for visualisation, head analysis,
// or any caller that needs to inspect or further process attention
// scores.
//
// When autograd is required, two ``ScaledDotProductAttentionBackward``
// nodes are wired — one for the output tensor (the normal SDPA
// backward) and a second, independent node attached to the weights
// tensor so callers may use the weights in downstream loss terms
// without conflating gradients between the two outputs.
//
// Parameters
// ----------
// q : TensorImplPtr
//     Query tensor of shape ``(..., L_q, d_k)``.
// k : TensorImplPtr
//     Key tensor of shape ``(..., L_k, d_k)``.
// v : TensorImplPtr
//     Value tensor of shape ``(..., L_k, d_v)``.
// attn_mask_or_null : TensorImplPtr
//     Optional additive mask broadcastable to ``(..., L_q, L_k)``.
// scale : double
//     Multiplicative factor applied to $QK^\top$ before the softmax.
// is_causal : bool
//     If ``true``, an upper-triangular ``-inf`` mask is applied engine-side.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     A two-element vector ``{output, weights}`` where ``output`` has
//     shape ``(..., L_q, d_v)`` and ``weights`` has shape
//     ``(..., L_q, L_k)``.
//
// Notes
// -----
// On the MLX (GPU) path the fused kernel does not produce a dense
// weight matrix; this entry point therefore recomputes
// $\text{softmax}(QK^\top \cdot s + M)$ to materialise the weights for
// the caller.  On the Accelerate (CPU) path the weight matrix is a
// natural byproduct of the manual softmax and is returned directly.
//
// See Also
// --------
// scaled_dot_product_attention_op : Lighter variant returning only the
//     contracted output, when attention weights are not needed.
LUCID_API std::vector<TensorImplPtr>
scaled_dot_product_attention_with_weights_op(const TensorImplPtr& q,
                                             const TensorImplPtr& k,
                                             const TensorImplPtr& v,
                                             const TensorImplPtr& attn_mask_or_null,
                                             double scale,
                                             bool is_causal);

}  // namespace lucid
