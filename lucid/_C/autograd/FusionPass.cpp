// lucid/_C/autograd/FusionPass.cpp
//
// Implements FusionPass::run() and the pattern-detection helpers.
// Also implements run_fusion_pass(), the free function called by
// Engine::backward() to kick off the pass before graph traversal.

#include "FusionPass.h"

#include <algorithm>
#include <string_view>
#include <unordered_set>

#include "../core/ErrorBuilder.h"
#include "Node.h"

namespace lucid {

namespace {

// Returns true when node b's outgoing edges contain exactly one reference to
// node a, i.e. a is a direct consumer of b.  Used to verify that the two
// adjacent nodes in the backward graph form a simple linear chain with no
// other consumers of b (so b's output is used only by a).
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

// Scan graph for fusible patterns and apply them.
//
// The pass makes two types of searches on each iteration:
//
//   Linear+Activation (2-node window):
//     graph[i] is the activation backward node (cur, 1 output edge) and
//     graph[i+1] is the linear backward node (next, 2-3 output edges).
//     If cur consumes next's output exclusively (sole_consumer), the pair
//     is a candidate.  try_fuse_linear_activation() is called; on success
//     cur is removed and i is decremented so the merged node is re-examined.
//
//   Scaled Dot-Product Attention (4-node window):
//     graph[i..i+3] must form an unbroken chain (each is the sole consumer
//     of the previous).  try_fuse_sdpa() fuses them; on success nodes i+1
//     through i+3 are erased.
//
// The loop index manipulation (--i after erasure) keeps the iterator valid
// because erasing an element shifts all subsequent indices down by one.
int FusionPass::run(std::vector<Node*>& graph) {
    stats_ = {};
    if (graph.size() < 2)
        return 0;

    for (std::size_t i = 0; i + 1 < graph.size(); ++i) {
        Node* cur = graph[i];
        Node* next = graph[i + 1];
        if (!cur || !next)
            continue;

        // 2-node window: potential Linear+Activation fusion.
        // cur is the activation node (unary: one outgoing edge).
        // next is the linear node (binary: 2-3 outgoing edges).
        const bool cur_unary = (cur->next_edges().size() == 1);
        const bool next_binary = (next->next_edges().size() >= 2 && next->next_edges().size() <= 3);
        if (cur_unary && next_binary && sole_consumer(next, cur)) {
            if (try_fuse_linear_activation(next, cur, FusionPattern::LinearRelu)) {
                ++stats_.linear_relu_fused;
                // Remove the activation node (cur) — the linear node absorbs it.
                graph.erase(graph.begin() + static_cast<std::ptrdiff_t>(i));
                // Decrement so the next iteration re-examines the same index
                // (which now points to the merged node's successor).
                --i;
                continue;
            }
        }

        // 4-node window: potential Scaled Dot-Product Attention fusion.
        if (i + 3 < graph.size()) {
            Node* n0 = graph[i];
            Node* n1 = graph[i + 1];
            Node* n2 = graph[i + 2];
            Node* n3 = graph[i + 3];
            if (n0 && n1 && n2 && n3) {
                // Require a strict linear chain: each node consumes only the
                // previous one's output.
                bool chain =
                    sole_consumer(n1, n0) && sole_consumer(n2, n1) && sole_consumer(n3, n2);
                if (chain && try_fuse_sdpa(n0, n1, n2, n3)) {
                    ++stats_.sdpa_fused;
                    // Erase n1, n2, n3; n0 becomes the fused SDPA node.
                    graph.erase(graph.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                                graph.begin() + static_cast<std::ptrdiff_t>(i) + 4);
                    continue;
                }
            }
        }
    }

    return stats_.total();
}

// Placeholder implementation — returns true unconditionally.
// Actual kernel substitution (e.g. replacing two backward nodes with a
// single fused LinearRelu backward kernel) will be added per pattern.
bool FusionPass::try_fuse_linear_activation(Node*, Node*, FusionPattern) {
    return true;
}

// Placeholder — returns true unconditionally.
bool FusionPass::try_fuse_conv_bn_relu(Node*, Node*, Node*) {
    return true;
}

// Placeholder — returns true unconditionally.
bool FusionPass::try_fuse_sdpa(Node*, Node*, Node*, Node*) {
    return true;
}

// Placeholder — LayerNorm fusion not yet implemented; returns false so the
// pattern is never counted as fused.
bool FusionPass::try_fuse_layernorm(std::vector<Node*>&) {
    return false;
}

// Build a topological ordering of the graph rooted at root_node using an
// iterative post-order DFS (raw pointer variant), then run FusionPass on it.
//
// The DFS is identical in structure to Engine's topo_order() but uses raw
// Node* and an unordered_set<Node*> for the visited set to avoid the
// reference-counting overhead of shared_ptr during the scan.
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
            // Post-order: push after all children are done.
            graph.push_back(f.node);
            stack.pop_back();
        }
    }
    // Reverse to root-first order so consecutive forward-adjacent nodes are
    // adjacent in the vector, matching the scanning assumptions in run().
    std::reverse(graph.begin(), graph.end());

    FusionPass fp;
    return fp.run(graph);
}

}  // namespace lucid
