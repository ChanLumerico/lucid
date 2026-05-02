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

bool storage_is_empty(const Storage& s) {
    if (auto* cpu = std::get_if<CpuStorage>(&s)) {
        return cpu->nbytes == 0 && cpu->ptr == nullptr;
    }
    if (auto* gpu = std::get_if<GpuStorage>(&s)) {
        return gpu->nbytes == 0 && gpu->arr == nullptr;
    }
    return false;
}

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

    Storage seed = std::move(grad_seed);
    if (storage_is_empty(seed)) {
        seed = make_ones_storage(root->shape(), root->dtype(), root->device());
    }

    run_fusion_pass(root->grad_fn().get());

    auto order = topo_order(root->grad_fn());

    std::unordered_map<Node*, Storage> pending;
    pending.emplace(root->grad_fn().get(), std::move(seed));

    for (const auto& node : order) {
        auto it = pending.find(node.get());
        if (it == pending.end()) {
            continue;
        }
        Storage grad_in = std::move(it->second);
        pending.erase(it);

        node->validate_versions();

        const auto input_grads = node->apply(std::move(grad_in));

        if (!retain_graph)
            node->release_saved();

        const auto& edges = node->next_edges();
        if (input_grads.size() != edges.size()) {
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
        root->clear_grad_fn();
    }
}

}  // namespace lucid
