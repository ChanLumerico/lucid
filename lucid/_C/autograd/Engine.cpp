// lucid/_C/autograd/Engine.cpp
//
// Implements Engine::backward().  The algorithm is a standard iterative
// post-order DFS to compute a topological ordering of the backward graph,
// followed by a single-pass loop that executes each node's apply() in that
// order and fans the resulting input gradients out along the edges.

#include "Engine.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "../ops/gfunc/Gfunc.h"
#include "AccumulateGrad.h"
#include "FusionPass.h"
#include "GraphBarrier.h"
#include "Helpers.h"
#include "Node.h"

namespace lucid {

// Declared where it is defined (ops/bfunc/Add.cpp); graph-mode accumulation
// has to go through it so the sum joins the new graph.
TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);

namespace {

constexpr const char* kSecondPass =
    "trying to backward through the graph a second time: the first backward() or "
    "autograd.grad() freed the tensors it saved — pass retain_graph=True to that call "
    "if the graph is needed again";

// Refuse a node whose saved tensors an earlier pass already freed.  Running
// it read released storage: two losses sharing a graph, each given its own
// backward() without retain_graph, took the process down with a
// segmentation fault.
void refuse_released(const Node& node) {
    if (node.saved_released())
        ErrorBuilder("backward").fail(std::string(kSecondPass) + " (" + node.node_name() + ")");
}

// What a root keeps once its graph has been spent.  Clearing its grad_fn
// instead made it look like a leaf, and a second backward() on the same
// loss quietly wrote the seed into the loss's own .grad.
class SpentGraphNode final : public Node {
public:
    std::vector<Storage> apply(Storage /*grad_out*/) override {
        refuse();
        return {};
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& /*grad_out*/) override {
        refuse();
        return {};
    }

    void validate_versions() override { refuse(); }

    std::string node_name() const override { return "SpentGraph"; }

private:
    static void refuse() { ErrorBuilder("backward").fail(kSecondPass); }
};

// Sever the root from the graph it just ran so the nodes can be freed.
void release_root(const std::shared_ptr<TensorImpl>& root) {
    if (root->is_leaf()) {
        root->clear_grad_fn();
        return;
    }
    root->set_grad_fn(std::make_shared<SpentGraphNode>());
}

// A gradient with a zero derivative of its own.
//
// Many graph-mode formulas build an input's gradient from the incoming
// gradient and a comparison on the saved input — relu6, hardtanh, clamp
// and max/min mask the seed by where x falls.  A comparison carries no
// graph, so when the seed is itself a constant the result came back
// detached: its derivative with respect to x is zero almost everywhere,
// but autograd.grad on it raised "not reachable" and backward() refused
// it, where the reference framework answers zeros (its node still lists
// x as an input).  This node restores that connection with the derivative
// the result actually has — zero, emitted directly rather than as
// ``grad * 0``, which an infinite upstream gradient would turn into NaN.
class ZeroDerivativeNode final : public Node {
public:
    ZeroDerivativeNode(Edge edge, Shape shape, Dtype dtype, Device device)
        : shape_(std::move(shape)), dtype_(dtype), device_(device) {
        set_next_edges({std::move(edge)});
    }

    std::vector<Storage> apply(Storage /*grad_out*/) override {
        return {make_zero_storage(shape_, dtype_, device_)};
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        return {zeros_like_op(grad_out)};
    }

    std::string node_name() const override { return "ZeroDerivativeBackward"; }

private:
    Shape shape_;
    Dtype dtype_;
    Device device_;
};

// Attach every detached gradient in ``grads`` to the edge it is about to
// travel, so the gradient stays part of the graph create_graph is building.
std::vector<TensorImplPtr> keep_in_graph(std::vector<TensorImplPtr> grads,
                                         const std::vector<Edge>& edges) {
    for (std::size_t i = 0; i < grads.size() && i < edges.size(); ++i) {
        const auto& g = grads[i];
        if (!g || g->requires_grad() || !edges[i].node)
            continue;
        auto node =
            std::make_shared<ZeroDerivativeNode>(edges[i], g->shape(), g->dtype(), g->device());
        auto alias = TensorImpl::make_view(g, g->shape(), g->stride(), 0, /*join_family=*/false);
        alias->set_grad_fn(std::move(node));
        alias->set_grad_output_nr(0);
        alias->set_leaf(false);
        alias->set_requires_grad(true);
        grads[i] = std::move(alias);
    }
    return grads;
}

// ── Graph-mode delivery ──────────────────────────────────────────────────────
//
// An ordinary node gets one pending gradient, the add_op sum of everything
// routed to it.  A barrier that implements GraphBarrier takes its gradients
// slot by slot instead, as eager backward hands them over — summing a hooked
// module's two outputs into one tensor was wrong — and ``pending`` holds only
// a placeholder that schedules it.

void seed_for_graph(std::unordered_map<Node*, TensorImplPtr>& pending,
                    const std::shared_ptr<TensorImpl>& root,
                    TensorImplPtr seed) {
    Node* node = root->grad_fn().get();
    if (auto* barrier = graph_barrier_of(node)) {
        barrier->accumulate_barrier_grad_for_graph(root->grad_output_nr(), std::move(seed));
        pending.emplace(node, nullptr);
        return;
    }
    pending.emplace(node, std::move(seed));
}

void route_for_graph(std::unordered_map<Node*, TensorImplPtr>& pending,
                     const Edge& edge,
                     const TensorImplPtr& grad) {
    Node* next = edge.node.get();
    if (next == nullptr || !grad)
        return;
    if (auto* barrier = graph_barrier_of(next)) {
        barrier->accumulate_barrier_grad_for_graph(edge.input_nr, grad);
        pending.emplace(next, nullptr);
        return;
    }
    auto pit = pending.find(next);
    if (pit == pending.end())
        pending.emplace(next, grad);
    else
        pit->second = add_op(pit->second, grad);
}

std::vector<TensorImplPtr> apply_node_for_graph(Node& node, const TensorImplPtr& grad_in) {
    if (auto* barrier = graph_barrier_of(&node))
        return barrier->apply_barrier_for_graph();
    return node.apply_for_graph(grad_in);
}

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

// A gradient buffer its holder alone owns, for a holder that keeps adding
// into it.
//
// The gradient a node hands on is shared — ``pending`` keeps it, a barrier
// parks it — and later arrivals are added into those in place, so a second
// holder of the same CPU buffer would see them too, and its own adds would
// reach the others.  A metal add replaces the array rather than writing into
// it, so a metal gradient can be shared as it is.
Storage own_copy(const Storage& s) {
    if (!storage_is_cpu(s))
        return s;
    const Dtype dt = storage_dtype(s);
    return clone_storage(s, storage_nbytes(s) / dtype_size(dt), dt, Device::CPU);
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

// ── Graph-mode backward (create_graph=true) ───────────────────────────────────
//
// Runs the same topological traversal as backward() but uses TensorImplPtr
// throughout so that every gradient operation is itself recorded in the
// autograd graph.  Gradient accumulation at shared nodes is done via add_op
// (which creates a new backward node) rather than raw storage +=.
static void backward_for_graph(const std::shared_ptr<TensorImpl>& root,
                               TensorImplPtr grad_seed,
                               bool retain_graph) {
    if (!root->grad_fn()) {
        // Leaf shortcut: store the seed directly as grad_impl.
        if (root->requires_grad()) {
            root->accumulate_grad_impl(grad_seed);
        }
        return;
    }

    run_fusion_pass(root->grad_fn().get());
    auto order = topo_order(root->grad_fn());

    std::unordered_map<Node*, TensorImplPtr> pending;
    seed_for_graph(pending, root, std::move(grad_seed));

    for (const auto& node : order) {
        auto it = pending.find(node.get());
        if (it == pending.end())
            continue;
        TensorImplPtr grad_in = std::move(it->second);
        pending.erase(it);

        refuse_released(*node);
        node->validate_versions();
        // What eager backward reads by value, graph mode reads through the
        // tensors the node was handed — and an in-place write since forward
        // may have rewritten one that validate_versions does not check: a
        // saved output, or any input while the check is waived.
        node->restore_saved_for_graph();

        // apply_for_graph throws NotImplementedError if the op doesn't support
        // graph mode — gives the user a clear, actionable message.
        const auto input_grads =
            keep_in_graph(apply_node_for_graph(*node, grad_in), node->next_edges());
        // Collected before release_saved() clears them, as in eager backward.
        const auto retain_ins = node->retainable_inputs();

        if (!retain_graph) {
            node->release_saved();
            node->forget_saved_versions();
        }

        const auto& edges = node->next_edges();
        for (std::size_t i = 0; i < input_grads.size() && i < edges.size(); ++i) {
            // retain_grad: a non-leaf that asked for its gradient gets it here
            // as well, summed by add_op so it carries its graph as a leaf's
            // does.  Graph mode had no such step, and left .grad at None.
            if (i < retain_ins.size() && input_grads[i]) {
                if (auto t = retain_ins[i].lock()) {
                    if (t->retains_grad() && !t->is_leaf())
                        t->accumulate_grad_impl(input_grads[i]);
                }
            }
            route_for_graph(pending, edges[i], input_grads[i]);
        }
    }

    if (!retain_graph)
        release_root(root);
}

void Engine::backward(const std::shared_ptr<TensorImpl>& root,
                      Storage grad_seed,
                      bool retain_graph,
                      bool create_graph) {
    if (!root) {
        ErrorBuilder("Engine::backward").fail("root is null");
    }
    // Barriers discard what an earlier pass left in them (GraphBarrier.h).
    const BackwardPass pass;

    if (create_graph) {
        // create_graph=True implies retain_graph=True: the backward computation
        // re-uses forward nodes (through apply_for_graph's saved_impl_inputs_),
        // and a second backward must be able to traverse those nodes.  Releasing
        // them would cause use-after-free when the second backward reaches nodes
        // that were freed by the first.
        Storage seed = std::move(grad_seed);
        if (storage_is_empty(seed)) {
            seed = make_ones_storage(root->shape(), root->dtype(), root->device());
        }
        auto seed_impl = std::make_shared<TensorImpl>(std::move(seed), root->shape(), root->dtype(),
                                                      root->device(), false);
        backward_for_graph(root, std::move(seed_impl), /*retain_graph=*/true);
        return;
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
    if (root->grad_fn()->is_barrier()) {
        root->grad_fn()->accumulate_barrier_grad(root->grad_output_nr(), std::move(seed));
        pending.emplace(root->grad_fn().get(), Storage{CpuStorage{}});
    } else {
        pending.emplace(root->grad_fn().get(), std::move(seed));
    }

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
        refuse_released(*node);
        node->validate_versions();

        // Execute the backward formula for this node.
        const auto input_grads =
            node->is_barrier() ? node->apply_barrier() : node->apply(std::move(grad_in));

        // Collect retain_grad inputs BEFORE release_saved() clears input_tensors_.
        const auto retain_ins = node->retainable_inputs();

        // Free saved forward tensors immediately unless the caller needs the
        // graph intact for a second backward call.
        if (!retain_graph) {
            node->release_saved();
            node->forget_saved_versions();
        }

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
            // A custom Function's ``None`` arrives as an empty storage: no
            // gradient for that input.  Routed on, it became the leaf's
            // ``.grad`` — the leaf's shape over no memory, read back as
            // whatever the allocator had last left there.
            if (node->empty_grad_is_none() && storage_is_empty(input_grads[i]))
                continue;
            // retain_grad: accumulate into non-leaf tensors that requested it.
            // The retained gradient gets a buffer of its own: pending holds
            // this one too, and adds the next arrival into it.
            if (i < retain_ins.size()) {
                if (auto t = retain_ins[i].lock()) {
                    if (t->retains_grad() && !t->is_leaf()) {
                        auto& g = t->mutable_grad_storage();
                        if (!g.has_value())
                            g = own_copy(input_grads[i]);
                        else
                            accumulate_into(*g, input_grads[i]);
                    }
                }
            }

            auto next = edges[i].node;
            if (!next)
                continue;
            if (next->is_barrier()) {
                next->accumulate_barrier_grad(edges[i].input_nr, input_grads[i]);
                pending.emplace(next.get(), Storage{CpuStorage{}});
                continue;
            }
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
        release_root(root);
    }
}

// ── Functional gradients (no .grad written anywhere) ─────────────────────────
//
// Same traversal as backward(), with one difference that is the whole point:
// an AccumulateGrad node is never executed.  Executing it is what writes a
// leaf's .grad, so intercepting the gradient on its way in leaves every
// tensor's gradient state untouched — including leaves the caller never
// asked about.

namespace {

// Where a requested input should be captured during the traversal.
//
// Everything is keyed by node.  A leaf that has taken part in any op already
// owns an AccumulateGrad as its own ``grad_fn`` (``ensure_grad_fn`` installs
// it), so leaves and interior tensors are both reached the same way — there
// is no separate "match the leaf tensor" case to get wrong.
//
// A node can capture for several requested inputs: the outputs of one
// multi-output node share it, and so does a tensor asked for twice.  Keeping
// one index per node answered only the first of them and called the rest
// unreachable.
struct CaptureTargets {
    std::unordered_map<const Node*, std::vector<std::size_t>> by_node;
    // Only for a tensor with no grad_fn at all, which can be captured solely
    // as the root of the differentiation.
    std::unordered_map<const TensorImpl*, std::vector<std::size_t>> orphans;
};

CaptureTargets build_targets(const std::vector<std::shared_ptr<TensorImpl>>& inputs) {
    CaptureTargets targets;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& inp = inputs[i];
        if (!inp)
            ErrorBuilder("Engine::grad").fail("inputs[" + std::to_string(i) + "] is null");
        if (inp->grad_fn())
            targets.by_node[inp->grad_fn().get()].push_back(i);
        else
            targets.orphans[inp.get()].push_back(i);
    }
    return targets;
}

// The requested inputs this node captures for, or null.
const std::vector<std::size_t>* capture_slots(const Node* node, const CaptureTargets& targets) {
    auto it = targets.by_node.find(node);
    return it == targets.by_node.end() ? nullptr : &it->second;
}

// Whether a gradient arriving at slot ``input_nr`` of ``barrier`` is one a
// requested input asked for.
//
// A barrier takes its gradients slot by slot, and ``pending`` holds only a
// placeholder for it — so the gradient of a tensor a barrier produced (a
// hooked module's output, one output of a multi-output Function) is gathered
// on its way in, from what reaches that tensor's own slot.  Capturing the
// placeholder handed back a tensor over no memory, and reading it crashed
// the process.
bool captures_at_barrier(const Node* barrier,
                         std::uint32_t input_nr,
                         const CaptureTargets& targets,
                         const std::vector<std::shared_ptr<TensorImpl>>& inputs) {
    const auto* slots = capture_slots(barrier, targets);
    if (slots == nullptr)
        return false;
    return std::any_of(slots->begin(), slots->end(),
                       [&](std::size_t s) { return inputs[s]->grad_output_nr() == input_nr; });
}

// Gradients gathered at the requested slots of barriers, by (node, slot).
template <typename Grad>
using BarrierCaptures = std::map<std::pair<const Node*, std::uint32_t>, Grad>;

// The nodes from which some requested input is still reachable.
//
// backward() has to walk the whole graph — every leaf is a destination.
// grad() does not: only the paths between the root and the requested
// inputs carry a gradient anybody asked for, so anything off them is work
// with no result to show for it.  Pruning it is not merely an
// optimisation.  Asking for the gradient with respect to an interior
// tensor is how a vector-Jacobian product is taken mid-graph — the
// divergence of a continuous flow's vector field, a gradient penalty, a
// Hessian-vector product — and continuing past that tensor drags the
// traversal into whatever produced it, which under create_graph must then
// supply a double-backward formula it was never asked to have.
//
// One reverse sweep suffices: ``order`` is topological, so a node's
// successors have already been decided by the time it is examined.
std::unordered_set<const Node*> reaching_nodes(const std::vector<std::shared_ptr<Node>>& order,
                                               const CaptureTargets& targets) {
    std::unordered_set<const Node*> reaching;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const Node* node = it->get();
        bool hit = targets.by_node.find(node) != targets.by_node.end();
        if (!hit) {
            for (const auto& edge : node->next_edges()) {
                if (edge.node && reaching.find(edge.node.get()) != reaching.end()) {
                    hit = true;
                    break;
                }
            }
        }
        if (hit)
            reaching.insert(node);
    }
    return reaching;
}

// Whether this node's gradient has anywhere left to go.
//
// Distinct from membership in ``reaching``: a requested input reaches
// itself, but its gradient is captured rather than propagated unless a
// *further* request sits upstream of it.
bool propagates(const Node* node, const std::unordered_set<const Node*>& reaching) {
    for (const auto& edge : node->next_edges()) {
        if (edge.node && reaching.find(edge.node.get()) != reaching.end())
            return true;
    }
    return false;
}

}  // namespace

std::vector<TensorImplPtr> Engine::grad(const std::shared_ptr<TensorImpl>& root,
                                        Storage grad_seed,
                                        const std::vector<std::shared_ptr<TensorImpl>>& inputs,
                                        bool retain_graph,
                                        bool create_graph) {
    if (!root)
        ErrorBuilder("Engine::grad").fail("root is null");
    // Barriers discard what an earlier pass left in them (GraphBarrier.h).
    const BackwardPass pass;

    const CaptureTargets targets = build_targets(inputs);
    std::vector<TensorImplPtr> results(inputs.size());

    Storage seed = std::move(grad_seed);
    if (storage_is_empty(seed))
        seed = make_ones_storage(root->shape(), root->dtype(), root->device());

    // A leaf root differentiates to itself: dy/dy is the seed.
    if (!root->grad_fn()) {
        auto it = targets.orphans.find(root.get());
        if (it != targets.orphans.end()) {
            for (const std::size_t s : it->second)
                results[s] = std::make_shared<TensorImpl>(seed, root->shape(), root->dtype(),
                                                          root->device(), false);
        }
        return results;
    }

    if (create_graph) {
        auto seed_impl = std::make_shared<TensorImpl>(std::move(seed), root->shape(), root->dtype(),
                                                      root->device(), false);
        auto order = topo_order(root->grad_fn());
        const auto reaching = reaching_nodes(order, targets);

        // What reaches the requested slots of barriers (captures_at_barrier).
        BarrierCaptures<TensorImplPtr> at_barrier;
        auto gather = [&](const Node* next, std::uint32_t input_nr, const TensorImplPtr& grad) {
            if (!grad || !next->is_barrier() ||
                !captures_at_barrier(next, input_nr, targets, inputs))
                return;
            auto [cit, fresh] = at_barrier.try_emplace({next, input_nr}, grad);
            if (!fresh)
                cit->second = add_op(cit->second, grad);
        };

        std::unordered_map<Node*, TensorImplPtr> pending;
        gather(root->grad_fn().get(), root->grad_output_nr(), seed_impl);
        seed_for_graph(pending, root, std::move(seed_impl));

        for (const auto& node : order) {
            auto it = pending.find(node.get());
            if (it == pending.end())
                continue;
            TensorImplPtr grad_in = std::move(it->second);
            pending.erase(it);

            if (const auto* slots = capture_slots(node.get(), targets)) {
                for (const std::size_t s : *slots) {
                    TensorImplPtr captured = grad_in;
                    if (node->is_barrier()) {
                        auto cit = at_barrier.find({node.get(), inputs[s]->grad_output_nr()});
                        captured = cit == at_barrier.end() ? nullptr : cit->second;
                    }
                    results[s] = gradient_in_dtype_of(captured, inputs[s]);
                }
            }
            // Executing an AccumulateGrad is precisely what writes a leaf's
            // .grad, so this path never does — captured or not, stop here.
            if (dynamic_cast<const AccumulateGrad*>(node.get()) != nullptr)
                continue;
            if (!propagates(node.get(), reaching))
                continue;

            refuse_released(*node);
            node->validate_versions();
            node->restore_saved_for_graph();
            const auto input_grads =
                keep_in_graph(apply_node_for_graph(*node, grad_in), node->next_edges());

            const auto& edges = node->next_edges();
            for (std::size_t i = 0; i < input_grads.size() && i < edges.size(); ++i) {
                if (edges[i].node)
                    gather(edges[i].node.get(), edges[i].input_nr, input_grads[i]);
                route_for_graph(pending, edges[i], input_grads[i]);
            }
        }
        return results;
    }

    run_fusion_pass(root->grad_fn().get());
    auto order = topo_order(root->grad_fn());
    const auto reaching = reaching_nodes(order, targets);

    // What reaches the requested slots of barriers (captures_at_barrier), in
    // buffers of their own: the barrier is handed the same gradient and may
    // add into it.
    BarrierCaptures<Storage> at_barrier;
    auto gather = [&](const Node* barrier, std::uint32_t input_nr, const Storage& grad) {
        if (!captures_at_barrier(barrier, input_nr, targets, inputs))
            return;
        auto cit = at_barrier.find({barrier, input_nr});
        if (cit == at_barrier.end())
            at_barrier.emplace(std::make_pair(barrier, input_nr), own_copy(grad));
        else
            accumulate_into(cit->second, grad);
    };

    std::unordered_map<Node*, Storage> pending;
    if (root->grad_fn()->is_barrier()) {
        gather(root->grad_fn().get(), root->grad_output_nr(), seed);
        root->grad_fn()->accumulate_barrier_grad(root->grad_output_nr(), std::move(seed));
        pending.emplace(root->grad_fn().get(), Storage{CpuStorage{}});
    } else {
        pending.emplace(root->grad_fn().get(), std::move(seed));
    }

    for (const auto& node : order) {
        auto it = pending.find(node.get());
        if (it == pending.end())
            continue;
        Storage grad_in = std::move(it->second);
        pending.erase(it);

        if (const auto* slots = capture_slots(node.get(), targets)) {
            for (const std::size_t s : *slots) {
                const auto& inp = inputs[s];
                const Storage* captured = &grad_in;
                if (node->is_barrier()) {
                    auto cit = at_barrier.find({node.get(), inp->grad_output_nr()});
                    captured = cit == at_barrier.end() ? nullptr : &cit->second;
                }
                if (captured != nullptr)
                    results[s] = std::make_shared<TensorImpl>(*captured, inp->shape(), inp->dtype(),
                                                              inp->device(), false);
            }
        }
        // Executing an AccumulateGrad is precisely what writes a leaf's
        // .grad, so this path never does — captured or not, stop here.
        if (dynamic_cast<const AccumulateGrad*>(node.get()) != nullptr)
            continue;
        if (!propagates(node.get(), reaching))
            continue;

        refuse_released(*node);
        node->validate_versions();
        const auto input_grads =
            node->is_barrier() ? node->apply_barrier() : node->apply(std::move(grad_in));

        if (!retain_graph) {
            node->release_saved();
            node->forget_saved_versions();
        }

        const auto& edges = node->next_edges();
        for (std::size_t i = 0; i < input_grads.size() && i < edges.size(); ++i) {
            auto next = edges[i].node;
            // As in ``backward``: from these nodes an empty storage is a
            // ``None``, not a value.
            if (!next || (node->empty_grad_is_none() && storage_is_empty(input_grads[i])))
                continue;
            if (next->is_barrier()) {
                gather(next.get(), edges[i].input_nr, input_grads[i]);
                next->accumulate_barrier_grad(edges[i].input_nr, input_grads[i]);
                pending.emplace(next.get(), Storage{CpuStorage{}});
                continue;
            }
            auto pit = pending.find(next.get());
            if (pit == pending.end())
                pending.emplace(next.get(), input_grads[i]);
            else
                accumulate_into(pit->second, input_grads[i]);
        }
    }

    if (!retain_graph)
        release_root(root);

    return results;
}

}  // namespace lucid
