#pragma once

#include <cstddef>
#include <vector>

#include "../api.h"
#include "Node.h"

namespace lucid {

enum class FusionPattern {
    LinearRelu,
    LinearGelu,
    LinearSilu,
    AddRelu,
    LayerNorm,
    ScaledDotProduct,
    ConvBnRelu,
};

class LUCID_API FusionPass {
public:
    FusionPass() = default;

    int run(std::vector<Node*>& graph);

    struct Stats {
        int linear_relu_fused = 0;
        int linear_gelu_fused = 0;
        int linear_silu_fused = 0;
        int add_relu_fused = 0;
        int layer_norm_fused = 0;
        int sdpa_fused = 0;
        int conv_bn_relu_fused = 0;

        int total() const noexcept {
            return linear_relu_fused + linear_gelu_fused + linear_silu_fused + add_relu_fused +
                   layer_norm_fused + sdpa_fused + conv_bn_relu_fused;
        }
    };

    const Stats& stats() const noexcept { return stats_; }

private:
    Stats stats_;

    bool try_fuse_linear_activation(Node* linear_node, Node* act_node, FusionPattern pattern);
    bool try_fuse_conv_bn_relu(Node* conv, Node* bn, Node* relu);
    bool try_fuse_sdpa(Node* mm1, Node* scale, Node* softmax, Node* mm2);
    bool try_fuse_layernorm(std::vector<Node*>& window);
};

LUCID_API int run_fusion_pass(Node* root_node);

}  // namespace lucid
