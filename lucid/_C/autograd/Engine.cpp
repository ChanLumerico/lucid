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
#include "FusionPass.h"  // Phase 19: op-fusion before backward
#include "Helpers.h"
#include "Node.h"

namespace lucid {

namespace {

bool storage_is_empty(const Storage& s) {
    if (auto* cpu = std::get_if<CpuStorage>(&s)) {
        return cpu->nbytes == 0 && cpu->ptr == nullptr;
    }
    if (auto* gpu = std::get_if<GpuStorage>(&s)) {
        return gpu->nbytes == 0 && gpu->arr == nullptr;
    }
    return false;
}

// DFS-based topological sort returning nodes in *reverse* topological order
// (root's grad_fn first, leaves' grad_fns last).
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
            order.push_back(std::move(f.node));
            stack.pop_back();
        }
    }
    // `order` came out in post-order (leaves before root). Reverse so we
    // process root first.
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
    if (!root->grad_fn()) {
        // Leaf root — no op history to traverse, but PyTorch semantics still
        // require accumulating the seed into the leaf's `grad_storage_` when
        // `requires_grad` is set. Without this branch `backward()` on a leaf
        // tensor is a silent no-op.
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

    // Provide an implicit ones_like(root) seed if caller passed an empty one.
    Storage seed = std::move(grad_seed);
    if (storage_is_empty(seed)) {
        seed = make_ones_storage(root->shape(), root->dtype(), root->device());
    }

    // Phase 19: run fusion pass on the backward graph before executing it.
    // This detects linear+activation and SDPA chains and replaces them with
    // fused backward nodes where available.  The call is a no-op when no
    // patterns are found, so it is safe to always run.
    run_fusion_pass(root->grad_fn().get());

    // Topological order, root first.
    auto order = topo_order(root->grad_fn());

    // Pending gradient per node. The root receives the seed.
    std::unordered_map<Node*, Storage> pending;
    pending.emplace(root->grad_fn().get(), std::move(seed));

    for (const auto& node : order) {
        auto it = pending.find(node.get());
        if (it == pending.end()) {
            // Some branches may receive no gradient (e.g. detached subgraphs).
            continue;
        }
        Storage grad_in = std::move(it->second);
        pending.erase(it);

        // Item #9 — verify saved input version_ counters still match the live
        // tensors. Throws lucid::VersionMismatch if the user did an in-place
        // op between forward and backward. AccumulateGrad's no-op default
        // applies (no inputs to check).
        node->validate_versions();

        const auto input_grads = node->apply(std::move(grad_in));

        // Phase 9.4: release saved tensors immediately after apply() when
        // retain_graph=false. Prevents activation buffers from being held
        // alive until the output TensorImpl destructor runs.
        if (!retain_graph)
            node->release_saved();

        const auto& edges = node->next_edges();
        if (input_grads.size() != edges.size()) {
            // AccumulateGrad returns {} (no edges to forward to).
            if (!edges.empty() && !input_grads.empty()) {
                ErrorBuilder("Engine::backward").fail("input_grads/next_edges size mismatch");
            }
        }

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

    if (!retain_graph) {
        // Break the forward reference from the output tensor back into the
        // graph so repeated backward() without retain_graph=True fails early.
        root->clear_grad_fn();
    }
}

}  // namespace lucid
