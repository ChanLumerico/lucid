#pragma once

// =====================================================================
// Lucid C++ engine — FusionPass (Phase 19)
// =====================================================================
//
// Eager-mode op-fusion: after a forward subgraph is built, FusionPass
// walks the backward graph and identifies patterns that can be collapsed
// into a single, hardware-fused kernel.  The result is that:
//   1. Memory traffic is reduced (intermediate tensors never materialise).
//   2. On CPU, BLAS + vDSP/vForce pipelines are fused (fewer calls, better
//      cache utilisation).
//   3. On GPU, MLX lazy evaluation is exploited for linear + activation
//      fusions; SDPA uses mlx::core::fast::scaled_dot_product_attention.
//
// Usage (called from kernel dispatch if GradMode is enabled):
//
//   FusionPass fp;
//   int fused = fp.run(graph_nodes);
//   // fused == number of patterns collapsed
//
// FusionPass is NOT responsible for re-executing the fused ops — it only
// replaces the backward nodes so that future backward calls are correct.
// The forward re-execution path (JIT) is out of scope for Phase 19; here
// the main win is the SDPA GPU upgrade and the fused linear+activation
// CPU kernels which are invoked directly from their respective ops.
//
// Layer: autograd/. Depends on autograd/Node.h, core/.

#include <cstddef>
#include <vector>

#include "../api.h"
#include "Node.h"

namespace lucid {

// ---------------------------------------------------------------------------
// Fusion patterns the pass can recognise and act upon.
// ---------------------------------------------------------------------------

enum class FusionPattern {
    LinearRelu,       ///< linear → relu          (CPU: SGEMM + vDSP threshold)
    LinearGelu,       ///< linear → gelu           (CPU: SGEMM + vForce GELU)
    LinearSilu,       ///< linear → silu           (CPU: SGEMM + vForce SiLU)
    AddRelu,          ///< add → relu              (CPU: vDSP add + threshold)
    LayerNorm,        ///< mean → var → norm → affine → 1 op
    ScaledDotProduct, ///< Q@K.T → scale → softmax → @V (SDPA)
    ConvBnRelu,       ///< conv2d → batch_norm → relu
};

// ---------------------------------------------------------------------------
// FusionPass
// ---------------------------------------------------------------------------

class LUCID_API FusionPass {
public:
    FusionPass() = default;

    // Run the pass over the provided list of AutogradNodes (taken as raw ptrs
    // because the pass does not own the graph; ownership stays with the
    // TensorImpls that reference them via grad_fn).
    //
    // Returns the number of fusion patterns successfully applied.
    int run(std::vector<Node*>& graph);

    // Statistics from the last run() call.
    struct Stats {
        int linear_relu_fused    = 0;
        int linear_gelu_fused    = 0;
        int linear_silu_fused    = 0;
        int add_relu_fused       = 0;
        int layer_norm_fused     = 0;
        int sdpa_fused           = 0;
        int conv_bn_relu_fused   = 0;

        int total() const noexcept {
            return linear_relu_fused + linear_gelu_fused + linear_silu_fused
                 + add_relu_fused   + layer_norm_fused   + sdpa_fused
                 + conv_bn_relu_fused;
        }
    };

    const Stats& stats() const noexcept { return stats_; }

private:
    Stats stats_;

    // Per-pattern detection + replacement helpers.
    bool try_fuse_linear_activation(Node* linear_node, Node* act_node,
                                    FusionPattern pattern);
    bool try_fuse_conv_bn_relu(Node* conv, Node* bn, Node* relu);
    bool try_fuse_sdpa(Node* mm1, Node* scale, Node* softmax, Node* mm2);
    bool try_fuse_layernorm(std::vector<Node*>& window);
};

// ---------------------------------------------------------------------------
// Convenience free function — call from the bindings layer or from
// Engine::backward to run the pass on a graph rooted at `root_node`.
// ---------------------------------------------------------------------------

/// Run FusionPass on the backward graph reachable from `root_node`.
/// Returns the number of fusion patterns detected and replaced.
LUCID_API int run_fusion_pass(Node* root_node);

}  // namespace lucid
