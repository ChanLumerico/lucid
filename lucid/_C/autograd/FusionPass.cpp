#include "FusionPass.h"

#include <algorithm>
#include <string_view>
#include <unordered_set>

#include "../core/ErrorBuilder.h"
#include "Node.h"

namespace lucid {

namespace {

bool sole_consumer(const Node* a, const Node* b) {
    if (!a || !b)
        return false;

    for (const auto& edge : b->next_edges()) {
        if (edge.node.get() == a)
            return true;
    }
    return false;
}

}  // namespace

int FusionPass::run(std::vector<Node*>& graph) {
    stats_ = {};
    if (graph.size() < 2)
        return 0;

    for (std::size_t i = 0; i + 1 < graph.size(); ++i) {
        Node* cur = graph[i];
        Node* next = graph[i + 1];
        if (!cur || !next)
            continue;

        const bool cur_unary = (cur->next_edges().size() == 1);
        const bool next_binary = (next->next_edges().size() >= 2 && next->next_edges().size() <= 3);
        if (cur_unary && next_binary && sole_consumer(next, cur)) {
            if (try_fuse_linear_activation(next, cur, FusionPattern::LinearRelu)) {
                ++stats_.linear_relu_fused;
                graph.erase(graph.begin() + static_cast<std::ptrdiff_t>(i));
                --i;
                continue;
            }
        }

        if (i + 3 < graph.size()) {
            Node* n0 = graph[i];
            Node* n1 = graph[i + 1];
            Node* n2 = graph[i + 2];
            Node* n3 = graph[i + 3];
            if (n0 && n1 && n2 && n3) {
                bool chain =
                    sole_consumer(n1, n0) && sole_consumer(n2, n1) && sole_consumer(n3, n2);
                if (chain && try_fuse_sdpa(n0, n1, n2, n3)) {
                    ++stats_.sdpa_fused;

                    graph.erase(graph.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                                graph.begin() + static_cast<std::ptrdiff_t>(i) + 4);
                    continue;
                }
            }
        }
    }

    return stats_.total();
}

bool FusionPass::try_fuse_linear_activation(Node*, Node*, FusionPattern) {
    return true;
}

bool FusionPass::try_fuse_conv_bn_relu(Node*, Node*, Node*) {
    return true;
}

bool FusionPass::try_fuse_sdpa(Node*, Node*, Node*, Node*) {
    return true;
}

bool FusionPass::try_fuse_layernorm(std::vector<Node*>&) {
    return false;
}

int run_fusion_pass(Node* root_node) {
    if (!root_node)
        return 0;

    std::vector<Node*> graph;
    std::unordered_set<Node*> visited;

    struct Frame {
        Node* node;
        std::size_t edge_idx;
    };
    std::vector<Frame> stack;
    stack.push_back({root_node, 0});
    visited.insert(root_node);

    while (!stack.empty()) {
        auto& f = stack.back();
        const auto& edges = f.node->next_edges();
        if (f.edge_idx < edges.size()) {
            auto* next = edges[f.edge_idx++].node.get();
            if (next && visited.insert(next).second) {
                stack.push_back({next, 0});
            }
        } else {
            graph.push_back(f.node);
            stack.pop_back();
        }
    }
    std::reverse(graph.begin(), graph.end());

    FusionPass fp;
    return fp.run(graph);
}

}  // namespace lucid
