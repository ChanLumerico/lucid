// lucid/_C/autograd/Engine.cpp
//
// Implements Engine::backward().  The algorithm is a standard iterative
// post-order DFS to compute a topological ordering of the backward graph,
// followed by a single-pass loop that executes each node's apply() in that
// order and fans the resulting input gradients out along the edges.

#include "Engine.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "FusionPass.h"
#include "Helpers.h"
#include "Node.h"

namespace lucid {

namespace {

// Returns true when s carries no data (zero nbytes and null pointer/array).
// Used to decide whether to synthesise a ones-valued grad_seed.
bool storage_is_empty(const Storage& s) {
    if (auto* cpu = std::get_if<CpuStorage>(&s)) {
        return cpu->nbytes == 0 && cpu->ptr == nullptr;
    }
    if (auto* gpu = std::get_if<GpuStorage>(&s)) {
        return gpu->nbytes == 0 && gpu->arr == nullptr;
    }
    return false;
}

// Compute a reverse-topological ordering of the backward graph rooted at root.
//
// Uses an iterative DFS with an explicit frame stack to avoid recursion
// overflows on deep networks.  Each frame records the node being visited and
// the index of the next edge to explore, giving the standard iterative
// post-order DFS behaviour:
//   - While the current frame still has unvisited children, push the first
//     unvisited child and advance the edge cursor.
//   - When all children are visited, append the node to `order` (post-order)
//     and pop the frame.
// Nodes already in the visited set are skipped so that shared sub-graphs are
// processed only once.
//
// After the DFS the vector is reversed so that index 0 is the root node
// (last node executed in forward, first in backward).
std::vector<std::shared_ptr<Node>> topo_order(const std::shared_ptr<Node>& root) {
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*> visited;

    struct Frame {
        std::shared_ptr<Node> node;
        std::size_t edge_idx;
    };
    std::vector<Frame> stack;
    stack.push_back({root, 0});
    visited.insert(root.get());

    while (!stack.empty()) {
        auto& f = stack.back();
        const auto& edges = f.node->next_edges();
        if (f.edge_idx < edges.size()) {
            const auto& edge = edges[f.edge_idx++];
            auto next = edge.node;
            if (next && visited.insert(next.get()).second) {
                stack.push_back({std::move(next), 0});
            }
        } else {
            // All children visited: this node is ready to be appended.
            order.push_back(std::move(f.node));
            stack.pop_back();
        }
    }

    // Post-order gives leaf-first ordering; reverse to get root-first.
    std::reverse(order.begin(), order.end());
    return order;
}

}  // namespace

void Engine::backward(const std::shared_ptr<TensorImpl>& root,
                      Storage grad_seed,
                      bool retain_graph) {
    if (!root) {
        ErrorBuilder("Engine::backward").fail("root is null");
    }

    // Leaf-tensor shortcut: if root has no grad_fn it is itself a leaf.
    // Accumulate the seed directly into root->grad and return early; there
    // is no backward graph to walk.
    if (!root->grad_fn()) {
        if (root->requires_grad()) {
            Storage seed = std::move(grad_seed);
            if (storage_is_empty(seed)) {
                seed = make_ones_storage(root->shape(), root->dtype(), root->device());
            }
            auto& grad = root->mutable_grad_storage();
            if (!grad.has_value()) {
                grad = std::move(seed);
            } else {
                accumulate_into(*grad, seed);
            }
        }
        return;
    }

    // Synthesise a ones seed when the caller does not provide one.
    Storage seed = std::move(grad_seed);
    if (storage_is_empty(seed)) {
        seed = make_ones_storage(root->shape(), root->dtype(), root->device());
    }

    // Optionally fuse adjacent backward nodes (e.g. LinearBackward + ReluBackward)
    // into a single fused node before traversal.
    run_fusion_pass(root->grad_fn().get());

    // Build the execution order once; reuse it for the accumulation loop.
    auto order = topo_order(root->grad_fn());

    // pending maps each node to the gradient that has been accumulated for it
    // so far.  Gradients from multiple edges pointing to the same node are
    // summed here before apply() is invoked.
    std::unordered_map<Node*, Storage> pending;
    pending.emplace(root->grad_fn().get(), std::move(seed));

    for (const auto& node : order) {
        auto it = pending.find(node.get());
        if (it == pending.end()) {
            // This node is not reachable from the root through the pending
            // map — it had no gradient contribution, so skip it.
            continue;
        }
        Storage grad_in = std::move(it->second);
        pending.erase(it);

        // Detect in-place mutations that would corrupt the backward pass.
        node->validate_versions();

        // Execute the backward formula for this node.
        const auto input_grads = node->apply(std::move(grad_in));

        // Free saved forward tensors immediately unless the caller needs the
        // graph intact for a second backward call.
        if (!retain_graph)
            node->release_saved();

        // Validate size consistency: the number of returned gradients should
        // equal the number of outgoing edges.  A mismatch is only an error
        // when both sides are non-empty; an empty result with edges is
        // permissible when the node intentionally produces no gradients
        // (e.g. a stop-gradient op).
        const auto& edges = node->next_edges();
        if (input_grads.size() != edges.size()) {
            if (!edges.empty() && !input_grads.empty()) {
                ErrorBuilder("Engine::backward").fail("input_grads/next_edges size mismatch");
            }
        }

        // Distribute computed input gradients to the consumer nodes.
        // If a consumer already has a partial gradient from another path,
        // accumulate in-place; otherwise insert directly.
        for (std::size_t i = 0; i < input_grads.size() && i < edges.size(); ++i) {
            auto next = edges[i].node;
            if (!next)
                continue;
            auto pit = pending.find(next.get());
            if (pit == pending.end()) {
                pending.emplace(next.get(), input_grads[i]);
            } else {
                accumulate_into(pit->second, input_grads[i]);
            }
        }
    }

    // Sever the reference from root back into the graph so the nodes can be
    // garbage-collected.  Skipped when retain_graph is true.
    if (!retain_graph) {
        root->clear_grad_fn();
    }
}

}  // namespace lucid
