// lucid/_C/autograd/FusionPass.h
//
// Declares FusionPass, which walks the backward graph immediately after the
// forward pass and replaces sequences of adjacent backward nodes with single
// fused nodes when a known fusible pattern is detected.  Fusing reduces the
// number of kernel launches and intermediate gradient buffers, improving
// throughput for common sub-graphs such as Linear+Activation or
// Convolution+BN+ReLU.
//
// The pass operates on a pre-computed topological node list (raw pointers)
// rather than the shared_ptr graph, so no ownership changes are needed.
// Detected fusions increment per-pattern counters exposed through Stats.

#pragma once

#include <cstddef>
#include <vector>

#include "../api.h"
#include "Node.h"

namespace lucid {

// Enumeration of supported backward fusion patterns.
//
// Each tag names a specific multi-node backward sub-graph that
// :class:`FusionPass` recognises and can collapse into a single node.
//
// Attributes
// ----------
// LinearRelu : enumerator
//     Linear backward followed by ReLU backward.
// LinearGelu : enumerator
//     Linear backward followed by GELU backward.
// LinearSilu : enumerator
//     Linear backward followed by SiLU / Swish backward.
// AddRelu : enumerator
//     Element-wise add backward followed by ReLU backward (residual
//     activations).
// LayerNorm : enumerator
//     Multi-node LayerNorm backward sub-graph (mean / var / affine).
// ScaledDotProduct : enumerator
//     Four-node QK\^T, scale, softmax, attn-V chain produced by
//     attention forward passes.
// ConvBnRelu : enumerator
//     Convolution backward + BatchNorm backward + ReLU backward triple,
//     common in CNN residual blocks.
enum class FusionPattern {
    LinearRelu,
    LinearGelu,
    LinearSilu,
    AddRelu,
    LayerNorm,
    ScaledDotProduct,
    ConvBnRelu,
};

// Backward-graph fusion pass that collapses adjacent fusible nodes.
//
// :class:`FusionPass` is invoked once per backward call, just before the
// Engine begins traversing the graph.  It scans the topologically-ordered
// node list for known multi-node patterns (see :class:`FusionPattern`) and
// rewrites them in place so that subsequent traversal launches fewer
// kernels and allocates fewer intermediate gradient buffers.
//
// The pass is stateless across runs except for the :attr:`stats_` counters,
// which are reset on each call to :func:`run`.  No node ownership changes:
// raw pointers reference nodes owned by their parent edges' ``shared_ptr``.
//
// Attributes
// ----------
// stats_ : Stats
//     Per-pattern fusion counts collected during the most recent
//     :func:`run` call.  Cleared at the entry of each run.
//
// Notes
// -----
// **Cost / benefit trade-off.** Fusing eliminates intermediate gradient
// buffers and kernel-launch overhead but tightens the dependency between
// adjacent ops — the fused node must materialise all saved tensors that
// either sub-op would have needed independently.  For sub-graphs that
// already share most of their saved state (Linear+activation, Conv+BN)
// this is a clear win; for unrelated ops it is not.
//
// **Detection only.** All ``try_fuse_*`` helpers currently return ``true``
// as placeholders — actual kernel substitution will be layered in per
// pattern in subsequent phases.  The counters in :attr:`Stats` therefore
// reflect *detected* opportunities rather than executed fusions.
//
// **Invariants.**
//
// - FusionPass never takes ownership of any :class:`Node`.
// - A successful fusion erases the consumed nodes from the graph vector
//   and leaves the surviving (fused) node in place at the same index.
//
// See Also
// --------
// :func:`run_fusion_pass` — convenience entry point used by
//     ``Engine::backward()``.
class LUCID_API FusionPass {
public:
    // Construct a fresh ``FusionPass`` with zeroed counters.
    //
    // Defaulted; the only state is the :attr:`stats_` ``Stats`` member,
    // whose POD counters are zero-initialised by their in-class default
    // initialisers.  Each :func:`run` call re-zeroes ``stats_`` at entry,
    // so re-using a single instance across backward calls is safe.
    FusionPass() = default;

    // Scan ``graph`` for fusible patterns and apply matches in place.
    //
    // Walks the node list from front to back examining a 2-node window for
    // Linear+Activation fusions and a 4-node window for Scaled Dot-Product
    // Attention fusions.  When a pattern matches and the corresponding
    // ``try_fuse_*`` helper returns ``true``, the consumed (non-surviving)
    // nodes are erased from ``graph`` and the index counter is rewound so
    // the merged node is re-examined on the next iteration.
    //
    // Parameters
    // ----------
    // graph : std::vector<Node*>&
    //     Topologically ordered backward graph, root first.  Modified in
    //     place: fused-away nodes are erased.
    //
    // Returns
    // -------
    // int
    //     Total number of fusions performed during this call
    //     (== :func:`Stats::total`).
    //
    // Notes
    // -----
    // The :attr:`stats_` counters are zeroed at function entry, so calling
    // :func:`run` repeatedly on the same instance is safe.
    int run(std::vector<Node*>& graph);

    // Per-pattern fusion counters populated by :func:`FusionPass::run`.
    //
    // One counter per :class:`FusionPattern` tag.  Reset to zero on every
    // call to :func:`FusionPass::run`.  Exposed read-only via
    // :func:`FusionPass::stats`.
    //
    // Attributes
    // ----------
    // linear_relu_fused : int
    //     Number of Linear+ReLU pairs fused.
    // linear_gelu_fused : int
    //     Number of Linear+GELU pairs fused.
    // linear_silu_fused : int
    //     Number of Linear+SiLU pairs fused.
    // add_relu_fused : int
    //     Number of Add+ReLU residual pairs fused.
    // layer_norm_fused : int
    //     Number of LayerNorm sub-graphs fused.
    // sdpa_fused : int
    //     Number of Scaled Dot-Product Attention chains fused.
    // conv_bn_relu_fused : int
    //     Number of Conv+BN+ReLU triples fused.
    struct Stats {
        int linear_relu_fused = 0;
        int linear_gelu_fused = 0;
        int linear_silu_fused = 0;
        int add_relu_fused = 0;
        int layer_norm_fused = 0;
        int sdpa_fused = 0;
        int conv_bn_relu_fused = 0;

        // Sum the per-pattern fusion counters.
        //
        // Returns
        // -------
        // int
        //     The arithmetic sum of every counter on this :class:`Stats`
        //     instance — the total number of fusions performed during the
        //     most recent :func:`FusionPass::run` call.
        int total() const noexcept {
            return linear_relu_fused + linear_gelu_fused + linear_silu_fused + add_relu_fused +
                   layer_norm_fused + sdpa_fused + conv_bn_relu_fused;
        }
    };

    // Read the per-pattern counters collected by the last :func:`run`.
    //
    // Returns
    // -------
    // const Stats&
    //     Reference to this pass's counters.  Stable until the next call
    //     to :func:`run`, which zeroes them.
    const Stats& stats() const noexcept { return stats_; }

private:
    Stats stats_;

    // Attempt to fuse a Linear backward node with an adjacent activation
    // backward node.
    //
    // Parameters
    // ----------
    // linear_node : Node*
    //     Backward node producing the activation's input (the Linear
    //     layer's matmul backward).
    // act_node : Node*
    //     Activation backward node consuming ``linear_node``'s output.
    // pattern : FusionPattern
    //     Which activation variant this is (``LinearRelu`` / ``LinearGelu``
    //     / ``LinearSilu``).  Selects the fused kernel that will eventually
    //     replace the pair.
    //
    // Returns
    // -------
    // bool
    //     ``true`` when the pair was successfully fused (caller removes
    //     ``act_node`` from the graph), ``false`` to skip.
    //
    // Notes
    // -----
    // Current implementation is a placeholder returning ``true``
    // unconditionally — the actual fused-kernel substitution is staged
    // for a follow-up phase.
    bool try_fuse_linear_activation(Node* linear_node, Node* act_node, FusionPattern pattern);

    // Attempt to fuse a Conv + BatchNorm + ReLU backward triple.
    //
    // Parameters
    // ----------
    // conv : Node*
    //     Conv2d backward node.
    // bn : Node*
    //     BatchNorm backward node consuming ``conv``'s output.
    // relu : Node*
    //     ReLU backward node consuming ``bn``'s output.
    //
    // Returns
    // -------
    // bool
    //     ``true`` on success; caller removes ``bn`` and ``relu`` from the
    //     graph and leaves ``conv`` as the surviving fused node.
    //
    // Notes
    // -----
    // Placeholder — returns ``true`` unconditionally pending the
    // implementation of the fused Conv+BN+ReLU backward kernel.
    bool try_fuse_conv_bn_relu(Node* conv, Node* bn, Node* relu);

    // Attempt to fuse a four-node Scaled Dot-Product Attention backward chain.
    //
    // Parameters
    // ----------
    // mm1 : Node*
    //     Backward node for the :math:`Q K^\top` matmul.
    // scale : Node*
    //     Backward node for the :math:`/\sqrt{d_k}` scaling step.
    // softmax : Node*
    //     Backward node for the row-wise softmax.
    // mm2 : Node*
    //     Backward node for the :math:`\text{attn} \cdot V` matmul.
    //
    // Returns
    // -------
    // bool
    //     ``true`` on success; caller erases ``scale``, ``softmax``, and
    //     ``mm2`` leaving ``mm1`` as the merged SDPA node.
    //
    // Notes
    // -----
    // Placeholder implementation; counts the detection but does not yet
    // substitute a fused attention-backward kernel.
    bool try_fuse_sdpa(Node* mm1, Node* scale, Node* softmax, Node* mm2);

    // Attempt to fuse a sliding window of LayerNorm backward sub-nodes.
    //
    // Parameters
    // ----------
    // window : std::vector<Node*>&
    //     Candidate backward nodes implementing the LayerNorm formula
    //     (mean, variance, normalisation, affine).  The vector may be
    //     modified in place when nodes are consumed.
    //
    // Returns
    // -------
    // bool
    //     ``true`` if the window matched a LayerNorm pattern and was
    //     fused.  Currently always returns ``false`` (LayerNorm fusion
    //     not yet implemented).
    bool try_fuse_layernorm(std::vector<Node*>& window);
};

// Convenience entry point invoked by ``Engine::backward()`` before traversal.
//
// Builds a topological ordering of the backward graph rooted at
// ``root_node`` using an iterative post-order DFS over raw ``Node*``
// (mirroring ``Engine::topo_order`` but avoiding ``shared_ptr`` reference-
// counting overhead during the scan), then constructs a :class:`FusionPass`
// and invokes :func:`FusionPass::run` on the result.
//
// Parameters
// ----------
// root_node : Node*
//     Root of the backward graph (usually the ``grad_fn`` of the loss
//     tensor).  ``nullptr`` is tolerated and short-circuits to ``0``.
//
// Returns
// -------
// int
//     Total number of fusions applied, or ``0`` when ``root_node`` is null
//     or the resulting graph has fewer than two nodes.
//
// See Also
// --------
// :class:`FusionPass` — the underlying pattern-matching pass.
LUCID_API int run_fusion_pass(Node* root_node);

}  // namespace lucid
