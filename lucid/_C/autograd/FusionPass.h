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

// Tags identifying each supported backward fusion pattern.
enum class FusionPattern {
    LinearRelu,
    LinearGelu,
    LinearSilu,
    AddRelu,
    LayerNorm,
    ScaledDotProduct,
    ConvBnRelu,
};

// Graph-level backward fusion pass.
//
// Instantiate once per backward call (or reuse across calls; the pass is
// stateless except for the Stats counters which are reset in run()).  Call
// run() with the topologically ordered node list produced by
// run_fusion_pass() or the Engine's internal topo_order().
//
// Invariants:
//   - FusionPass does not take ownership of any Node.
//   - Fusing a pattern erases the consumed nodes from the graph vector and
//     leaves the surviving (fused) node in place at the same index.
//   - The current implementation stubs out all try_fuse_* methods as
//     always-returning-true placeholders; actual kernel substitution will
//     be added per pattern in subsequent phases.
class LUCID_API FusionPass {
public:
    FusionPass() = default;

    // Scan graph for fusible patterns and apply fusions in-place.
    // Resets stats_ to zero at entry.  Returns the total number of fusions
    // performed (== stats_.total()).
    int run(std::vector<Node*>& graph);

    // Per-pattern fusion counts accumulated during the most recent run().
    struct Stats {
        int linear_relu_fused = 0;
        int linear_gelu_fused = 0;
        int linear_silu_fused = 0;
        int add_relu_fused = 0;
        int layer_norm_fused = 0;
        int sdpa_fused = 0;
        int conv_bn_relu_fused = 0;

        // Sum of all per-pattern counts.
        int total() const noexcept {
            return linear_relu_fused + linear_gelu_fused + linear_silu_fused + add_relu_fused +
                   layer_norm_fused + sdpa_fused + conv_bn_relu_fused;
        }
    };

    // Read the stats collected during the last call to run().
    const Stats& stats() const noexcept { return stats_; }

private:
    Stats stats_;

    // Try to fuse a linear backward node with an adjacent activation backward
    // node (linear_node produces the activation input, act_node applies it).
    // pattern identifies which activation variant this is.  Returns true on
    // success; the caller removes act_node from the graph.
    bool try_fuse_linear_activation(Node* linear_node, Node* act_node, FusionPattern pattern);

    // Try to fuse Conv + BatchNorm + ReLU backward nodes into a single op.
    // Returns true on success.
    bool try_fuse_conv_bn_relu(Node* conv, Node* bn, Node* relu);

    // Try to fuse a four-node Scaled Dot-Product Attention backward sequence:
    // mm1 (QK^T), scale, softmax, mm2 (attn * V).  Returns true on success.
    bool try_fuse_sdpa(Node* mm1, Node* scale, Node* softmax, Node* mm2);

    // Try to fuse a window of backward nodes that implement LayerNorm backward.
    // Returns true on success; the window vector may be modified in-place.
    bool try_fuse_layernorm(std::vector<Node*>& window);
};

// Convenience entry point used by Engine::backward().
//
// Builds a topological ordering of the backward graph rooted at root_node
// using an iterative DFS (same algorithm as Engine::topo_order but operating
// on raw Node* to avoid shared_ptr overhead), then invokes FusionPass::run()
// on the resulting list.
//
// Returns the total number of fusions applied, or 0 if root_node is null or
// the graph has fewer than two nodes.
LUCID_API int run_fusion_pass(Node* root_node);

}  // namespace lucid
