// lucid/_C/nn/Embedding.h
//
// Autograd-aware integer-index embedding lookup plus position-encoding
// helpers used by Transformer-family models.
//
// This header declares three autograd-aware operations and one
// fixed-tensor helper:
//
//   * ``EmbeddingBackward`` — dense (row-gather) lookup
//     $y = W[\text{idx}]$ on a $(V, D)$ weight matrix.  The backward
//     is a sparse scatter-add of ``grad_out`` into a zero-initialized
//     ``dW`` at the positions given by the saved integer indices;
//     rows at ``padding_idx`` are zeroed in both forward and backward.
//   * ``RotaryPosEmbeddingBackward`` — Rotary Position Embedding
//     (RoPE, Su et al. 2021).  Rotates pairs of features in the last
//     dimension by angles derived from precomputed cos/sin tables;
//     the tables are saved during forward and reused during backward
//     to apply the conjugate rotation.
//   * ``sinusoidal_pos_embedding_op`` — generates the fixed
//     non-trainable sinusoidal position-encoding matrix from
//     Vaswani et al. 2017.  No autograd node is wired because the
//     output has no differentiable inputs.
//   * ``embedding_bag_op`` — efficient lookup-and-reduce that fuses
//     a row gather and a sum / mean / max reduction over
//     variable-length bags, avoiding the intermediate ``(B, L, D)``
//     tensor that ``Embedding`` + ``sum`` would materialize.

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the embedding lookup
// $y = W[\text{idx}]$ on a $(V, D)$ weight matrix.
//
// Only the weight matrix is a differentiable input.  Integer indices
// have no meaningful gradient and are not stored in the standard
// ``saved_inputs_`` slot — they live in :member:`saved_indices_`
// instead, alongside their shape and dtype, so the backward can
// scatter-add ``grad_out`` into the correct rows of ``dW``.
//
// Forward dispatches to ``IBackend::embedding_forward`` which gathers
// rows from the weight storage at positions given by ``indices``,
// zeroing any output row whose index equals ``padding_idx``.  Backward
// uses ``IBackend::embedding_backward`` to scatter-add ``grad_out``
// into a zero-initialized ``dW`` at those same positions.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (name ``"embedding"``, version 1,
//     ``AmpPolicy::Promote``).
// padding_idx_ : int
//     Index whose row is masked to zero in both forward and backward.
//     ``-1`` disables the masking entirely.
// weight_shape_ : Shape
//     ``(num_embeddings, embed_dim)``; recovered during backward to
//     size ``dW``.
// saved_indices_ : Storage
//     Integer index tensor saved from the forward pass.  Not subject
//     to AMP because the dtype is integer.
// saved_indices_shape_ : Shape
//     Shape of the saved index tensor.
// saved_indices_dtype_ : Dtype
//     Dtype of the saved index tensor (default ``I64``).
class LUCID_API EmbeddingBackward : public FuncOp<EmbeddingBackward, 1> {
public:
    static const OpSchema schema_v1;
    int padding_idx_ = -1;   // Rows at this index are zeroed and skipped.
    Shape weight_shape_;     // (num_embeddings, embed_dim).
    Storage saved_indices_;  // Integer index tensor from forward.
    Shape saved_indices_shape_;
    Dtype saved_indices_dtype_ = Dtype::I64;

    // Look up rows of ``weight`` at positions given by ``indices``.
    //
    // Parameters
    // ----------
    // weight : TensorImplPtr
    //     Embedding matrix of shape ``(num_embeddings, embed_dim)``.
    //     Must be exactly 2-D.
    // indices : TensorImplPtr
    //     Integer-typed tensor of any shape with values in
    //     ``[0, num_embeddings)``.
    // padding_idx : int
    //     Row whose lookup is forced to zero; pass a negative value to
    //     disable.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output of shape $(\ast\,\text{indices.shape}, \text{embed\_dim})$
    //     — the input index shape with an extra trailing dim of
    //     ``embed_dim``.
    //
    // Math
    // ----
    // $$y[\ldots] = W[\text{idx}[\ldots]],
    //   \qquad y[\ldots] = 0 \text{ if } \text{idx}[\ldots] = \text{padding\_idx}.$$
    //
    // Shape
    // -----
    // - ``weight`` : $(V, D)$
    // - ``indices`` : $(\ast)$ integer
    // - Output : $(\ast, D)$
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``weight`` is not 2-D.
    // DeviceMismatch
    //     If ``weight`` and ``indices`` live on different devices.
    static TensorImplPtr
    forward(const TensorImplPtr& weight, const TensorImplPtr& indices, int padding_idx);

    // Scatter-add backward.
    //
    // Returns a single ``dW`` storage of shape ``weight_shape_``,
    // zero-initialized except where the saved indices accumulate
    // contributions from ``grad_out``.  Rows at ``padding_idx_``
    // accumulate nothing.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient of shape $(\ast, D)$.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-element ``{dW}``.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Autograd node for Rotary Position Embedding (RoPE, Su et al. 2021).
//
// Rotates pairs of features in the last dimension of ``input`` by
// position-dependent angles, encoding absolute position information
// in a way that lets dot-product attention recover relative position.
// During the forward, the backend returns a triple
// ``{rotated_input, cos, sin}``; the cos/sin tables are saved into
// :member:`saved_cos_` and :member:`saved_sin_` so the backward can
// apply the conjugate rotation without recomputing the trigonometry.
//
// :member:`interleaved_` selects the pair layout:
//
//   * ``true``  — adjacent pairs ``(d=0, d=1), (d=2, d=3), …``
//   * ``false`` — split-half ``(d=0, d=D/2), (d=1, d=D/2+1), …``
//
// The chosen layout must match how the model was trained or attention
// scores will be silently corrupted.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (name ``"rotary_pos_embedding"``, version 1,
//     ``AmpPolicy::ForceFP32``).  Forced to F32 because the rotation
//     math is precision-sensitive at long sequence lengths.
// interleaved_ : bool
//     Pair-layout flag (see above).
// orig_shape_ : Shape
//     Saved input shape; used by backward to size the gradient.
// saved_cos_ : Storage
//     Cosine table produced by the forward kernel.
// saved_sin_ : Storage
//     Sine table produced by the forward kernel.
class LUCID_API RotaryPosEmbeddingBackward : public FuncOp<RotaryPosEmbeddingBackward, 1> {
public:
    static const OpSchema schema_v1;
    bool interleaved_ = true;  // Pair layout; must match how the model was trained.
    Shape orig_shape_;

    Storage saved_cos_;  // Cosine table produced during forward.
    Storage saved_sin_;  // Sine table produced during forward.

    // Apply RoPE to ``input``.
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     Input of shape $(\ldots, L, D)$ with ``D`` even.  ``L`` is
    //     the sequence axis and ``D`` is rotated in pairs.
    // position_ids_or_null : TensorImplPtr
    //     Optional integer position-index tensor of shape ``(L,)``.
    //     When ``null``, the kernel uses ``[0, 1, …, L-1]`` implicitly.
    // interleaved : bool
    //     Pair-layout flag (see class docstring).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Rotated output with the same shape as ``input``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``input`` is fewer than 2-D or its last dim is odd.
    // DeviceMismatch
    //     If ``position_ids_or_null`` lives on a different device than ``input``.
    static TensorImplPtr forward(const TensorImplPtr& input,
                                 const TensorImplPtr& position_ids_or_null,
                                 bool interleaved);

    // Apply the conjugate rotation to ``grad_out`` using the saved
    // cos/sin tables.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient with shape :member:`orig_shape_`.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-element ``{grad_in}`` of the same shape.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Integer-index embedding lookup — free-function entry point.
//
// Thin wrapper that delegates to :func:`EmbeddingBackward::forward`.
// Used by the pybind11 binding layer and by other C++ call sites.
//
// Parameters
// ----------
// weight : TensorImplPtr
//     Embedding matrix of shape ``(num_embeddings, embed_dim)``.
// indices : TensorImplPtr
//     Integer-typed tensor of any shape.
// padding_idx : int
//     Index to mask; negative disables masking.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(*indices.shape, embed_dim)``.
LUCID_API TensorImplPtr embedding_op(const TensorImplPtr& weight,
                                     const TensorImplPtr& indices,
                                     int padding_idx);

// Generate the fixed sinusoidal position-encoding matrix from
// Vaswani et al. 2017, "Attention Is All You Need".
//
// Returns the non-trainable matrix
//
// $$\text{PE}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/D}}\right),
//   \quad
//   \text{PE}(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/D}}\right),$$
//
// where $p \in [0, L)$ indexes positions and $i \in [0, D/2)$ indexes
// frequency bands.  No autograd node is created — the result is a
// constant tensor used as an additive bias on token embeddings.
//
// Parameters
// ----------
// seq_len : std::int64_t
//     Sequence length ``L``.  Must be non-negative.
// embed_dim : std::int64_t
//     Embedding dimension ``D``.  Must be strictly positive.
// dtype : Dtype
//     Floating-point dtype of the output.
// device : Device
//     Device on which to allocate the output.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of shape ``(seq_len, embed_dim)``.
//
// Raises
// ------
// LucidError
//     If ``seq_len < 0`` or ``embed_dim <= 0``.
LUCID_API TensorImplPtr sinusoidal_pos_embedding_op(std::int64_t seq_len,
                                                    std::int64_t embed_dim,
                                                    Dtype dtype,
                                                    Device device);

// Apply Rotary Position Embedding to ``input`` — free-function entry point.
//
// Thin wrapper that delegates to :func:`RotaryPosEmbeddingBackward::forward`.
//
// Parameters
// ----------
// input : TensorImplPtr
//     Input of shape $(\ldots, L, D)$ with ``D`` even.
// position_ids_or_null : TensorImplPtr
//     Optional integer position-index tensor of shape ``(L,)``; pass
//     ``null`` to use the implicit ``[0, 1, …, L-1]``.
// interleaved : bool
//     Pair-layout flag (see :class:`RotaryPosEmbeddingBackward`).
//
// Returns
// -------
// TensorImplPtr
//     Rotated output with the same shape as ``input``.
LUCID_API TensorImplPtr rotary_pos_embedding_op(const TensorImplPtr& input,
                                                const TensorImplPtr& position_ids_or_null,
                                                bool interleaved);

// Pooled embedding lookup with per-bag reduction.
//
// Fuses a row gather and a reduction over bags into a single backend
// call, avoiding the intermediate $(B, L, D)$ tensor that
// ``embedding`` followed by ``sum``/``mean``/``max`` would materialize.
// Variable-length bags are encoded as a flat ``indices`` tensor plus
// an ``offsets`` tensor that marks the start of each bag — bag $b$
// consists of ``indices[offsets[b] : offsets[b+1]]`` (the last bag
// runs to the end of ``indices`` when ``include_last_offset=false``).
//
// No autograd node is currently wired for this op (sparse gradients
// are not yet supported on the Lucid backends), so the returned
// tensor is not differentiable with respect to ``weight``.
//
// Parameters
// ----------
// weight : TensorImplPtr
//     Embedding matrix of shape ``(V, D)``.
// indices : TensorImplPtr
//     Flat 1-D integer tensor of all bag contents concatenated.
// offsets : TensorImplPtr
//     1-D integer tensor of shape ``(B,)`` marking the start of each bag.
// mode : int
//     Reduction selector — ``0`` = sum, ``1`` = mean, ``2`` = max.
// padding_idx : int
//     Index that contributes zero to the reduction; negative disables.
// include_last_offset : bool
//     If ``true``, ``offsets`` is treated as ``(B+1,)`` so the last
//     bag also has an explicit upper bound.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, D)`` — one pooled embedding per bag.
//
// Math
// ----
// $$y_b =
//   \begin{cases}
//     \displaystyle\sum_{j\in\mathcal{B}_b} W[j] & \text{mode}=0\\[6pt]
//     \displaystyle\frac{1}{|\mathcal{B}_b|}\sum_{j\in\mathcal{B}_b} W[j]
//       & \text{mode}=1\\[6pt]
//     \displaystyle\max_{j\in\mathcal{B}_b} W[j] & \text{mode}=2
//   \end{cases}$$
//
// See Also
// --------
// embedding_op : Per-token lookup without reduction.
LUCID_API TensorImplPtr embedding_bag_op(const TensorImplPtr& weight,
                                         const TensorImplPtr& indices,
                                         const TensorImplPtr& offsets,
                                         int mode,
                                         int padding_idx,
                                         bool include_last_offset);

}  // namespace lucid
