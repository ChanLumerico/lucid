// lucid/_C/autograd/Engine.cpp
//
// Implements Engine::backward() and Engine::grad().  The algorithm is a
// standard iterative post-order DFS to compute a topological ordering of the
// backward graph, followed by a single-pass loop that executes each node's
// apply() in that order and fans the resulting input gradients out along the
// edges.  The hooks of a tensor run where its gradient is complete: on its
// producer's output slot when that node is popped, or, for a leaf, just
// before AccumulateGrad adds into ``.grad`` (TensorHooks.h).

#include "Engine.h"

#include <algorithm>
#include <map>
#include <optional>
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
#include "TensorHooks.h"

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

// ── Delivery ─────────────────────────────────────────────────────────────────
//
// An ordinary node gets one pending gradient, the sum of everything routed to
// it, and runs once on it — it has one output, so that sum is its slot's
// whole gradient.  A barrier takes its gradients slot by slot instead
// (``accumulate_barrier_grad``) and ``pending`` holds only a placeholder that
// schedules it.  In graph mode only a barrier that implements GraphBarrier
// does — summing a hooked module's two outputs into one tensor was wrong — and
// any other keeps the single pending sum.
//
// A tensor a barrier produced may have hooks or ``retain_grad``, which must see
// its slot's whole gradient, and may replace it, before the barrier does.  The
// gradients of such a slot are held here instead, by (node, slot), summed as
// they arrive, and handed to the barrier through the hooks when it is popped.
// ``grad()`` holds the slots it captures the same way, so a captured gradient
// is the one the hooks returned.

template <typename Grad>
using HeldGrads = std::map<std::pair<const Node*, std::uint32_t>, Grad>;

template <typename Grad>
using SlotGrads = std::vector<std::pair<std::uint32_t, Grad>>;

using Pending = std::unordered_map<Node*, Storage>;
using GraphPending = std::unordered_map<Node*, TensorImplPtr>;

// The first arrival is copied: its producer may hand the same buffer on along
// another edge, where it is added into in place.
void hold(HeldGrads<Storage>& held, const Node* node, std::uint32_t slot, const Storage& grad) {
    auto it = held.find({node, slot});
    if (it == held.end())
        held.emplace(std::make_pair(node, slot), own_grad_copy(grad));
    else
        accumulate_into(it->second, grad);
}

void hold(HeldGrads<TensorImplPtr>& held,
          const Node* node,
          std::uint32_t slot,
          const TensorImplPtr& grad) {
    auto [it, fresh] = held.try_emplace({node, slot}, grad);
    if (!fresh)
        it->second = add_op(it->second, grad);
}

// Take what is held for ``node``'s slots, in slot order.
template <typename Grad>
SlotGrads<Grad> take_held(HeldGrads<Grad>& held, const Node* node) {
    SlotGrads<Grad> out;
    auto it = held.lower_bound({node, 0});
    while (it != held.end() && it->first.first == node) {
        out.emplace_back(it->first.second, std::move(it->second));
        it = held.erase(it);
    }
    return out;
}

template <typename Grad>
const Grad* held_value(const SlotGrads<Grad>& slots, std::uint32_t slot) {
    for (const auto& [k, grad] : slots) {
        if (k == slot)
            return &grad;
    }
    return nullptr;
}

// The gradients held for ``node``'s slots, each through the hooks of the
// tensor on that slot.
SlotGrads<Storage> fire_held(Node& node, HeldGrads<Storage>& held) {
    SlotGrads<Storage> slots = take_held(held, &node);
    for (auto& [slot, grad] : slots)
        grad = run_slot_hooks(node, slot, std::move(grad));
    return slots;
}

SlotGrads<TensorImplPtr> fire_held_for_graph(Node& node, HeldGrads<TensorImplPtr>& held) {
    SlotGrads<TensorImplPtr> slots = take_held(held, &node);
    for (auto& [slot, grad] : slots)
        grad = run_slot_hooks_for_graph(node, slot, std::move(grad));
    return slots;
}

// Route ``grad`` to slot ``slot`` of ``next`` — held when ``hold_slot`` says
// the barrier's slot has to go through hooks or be captured first.
void route(Pending& pending,
           HeldGrads<Storage>& held,
           Node* next,
           std::uint32_t slot,
           Storage grad,
           bool hold_slot) {
    if (next->is_barrier()) {
        if (hold_slot)
            hold(held, next, slot, grad);
        else
            next->accumulate_barrier_grad(slot, std::move(grad));
        pending.try_emplace(next, Storage{CpuStorage{}});
        return;
    }
    auto pit = pending.find(next);
    if (pit == pending.end())
        pending.emplace(next, std::move(grad));
    else
        accumulate_into(pit->second, grad);
}

void route_for_graph(GraphPending& pending,
                     HeldGrads<TensorImplPtr>& held,
                     Node* next,
                     std::uint32_t slot,
                     const TensorImplPtr& grad,
                     bool hold_slot) {
    if (next == nullptr || !grad)
        return;
    if (hold_slot && next->is_barrier()) {
        hold(held, next, slot, grad);
        pending.try_emplace(next, nullptr);
        return;
    }
    if (auto* barrier = graph_barrier_of(next)) {
        barrier->accumulate_barrier_grad_for_graph(slot, grad);
        pending.try_emplace(next, nullptr);
        return;
    }
    auto pit = pending.find(next);
    if (pit == pending.end())
        pending.emplace(next, grad);
    else
        pit->second = pit->second ? add_op(pit->second, grad) : grad;
}

// Hand a graph-mode barrier the gradients of its held slots: slot by slot to
// a GraphBarrier, into the one pending sum otherwise.
void deliver_held_for_graph(Node& node, SlotGrads<TensorImplPtr> slots, TensorImplPtr& grad_in) {
    auto* barrier = graph_barrier_of(&node);
    for (auto& [slot, grad] : slots) {
        if (barrier != nullptr)
            barrier->accumulate_barrier_grad_for_graph(slot, std::move(grad));
        else
            grad_in = grad_in ? add_op(grad_in, grad) : std::move(grad);
    }
}

std::vector<TensorImplPtr> apply_node_for_graph(Node& node, const TensorImplPtr& grad_in) {
    if (auto* barrier = graph_barrier_of(&node))
        return barrier->apply_barrier_for_graph();
    return node.apply_for_graph(grad_in);
}

// Free what ``node`` saved for backward, now that it has run for the last
// time.  A node that saved tensors can never run again after this, so the
// hooks of the tensors it produced go too — which breaks the cycle a hook
// that refers to its own tensor closes through the node.
void release(Node& node) {
    node.release_saved();
    node.forget_saved_versions();
    if (node.saved_released())
        node.release_tensor_hooks();
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
        // Leaf shortcut: the seed is the leaf's gradient.
        accumulate_leaf_for_graph(root, grad_seed);
        return;
    }

    run_fusion_pass(root->grad_fn().get());
    auto order = topo_order(root->grad_fn());

    GraphPending pending;
    HeldGrads<TensorImplPtr> held;
    Node* const root_fn = root->grad_fn().get();
    const std::uint32_t root_slot = root->grad_output_nr();
    route_for_graph(pending, held, root_fn, root_slot, grad_seed,
                    has_slot_hooks(*root_fn, root_slot));

    for (const auto& node : order) {
        auto it = pending.find(node.get());
        if (it == pending.end())
            continue;
        TensorImplPtr grad_in = std::move(it->second);
        pending.erase(it);

        // The hooks of the tensors this node produced, and retain_grad, see
        // their whole gradient before the node does — and before the checks
        // below (see Engine::backward).
        SlotGrads<TensorImplPtr> slots;
        if (node->is_barrier())
            slots = fire_held_for_graph(*node, held);
        else if (node->tensor_hooks() != nullptr)
            grad_in = run_slot_hooks_for_graph(*node, 0, std::move(grad_in));

        refuse_released(*node);
        node->validate_versions();
        // What eager backward reads by value, graph mode reads through the
        // tensors the node was handed — and an in-place write since forward
        // may have rewritten one that validate_versions does not check: a
        // saved output, or any input while the check is waived.
        node->restore_saved_for_graph();

        if (node->is_barrier())
            deliver_held_for_graph(*node, std::move(slots), grad_in);

        // apply_for_graph throws NotImplementedError if the op doesn't support
        // graph mode — gives the user a clear, actionable message.
        const auto input_grads =
            keep_in_graph(apply_node_for_graph(*node, grad_in), node->next_edges());

        if (!retain_graph)
            release(*node);

        const auto& edges = node->next_edges();
        for (std::size_t i = 0; i < input_grads.size() && i < edges.size(); ++i) {
            Node* next = edges[i].node.get();
            route_for_graph(pending, held, next, edges[i].input_nr, input_grads[i],
                            next != nullptr && has_slot_hooks(*next, edges[i].input_nr));
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

    // Synthesise a ones seed when the caller does not provide one.
    Storage seed = std::move(grad_seed);
    if (storage_is_empty(seed)) {
        seed = make_ones_storage(root->shape(), root->dtype(), root->device());
    }

    if (create_graph) {
        // create_graph=True implies retain_graph=True: the backward computation
        // re-uses forward nodes (through apply_for_graph's saved_impl_inputs_),
        // and a second backward must be able to traverse those nodes.  Releasing
        // them would cause use-after-free when the second backward reaches nodes
        // that were freed by the first.
        auto seed_impl = std::make_shared<TensorImpl>(std::move(seed), root->shape(), root->dtype(),
                                                      root->device(), false);
        backward_for_graph(root, std::move(seed_impl), /*retain_graph=*/true);
        return;
    }

    // Leaf-tensor shortcut: if root has no grad_fn it is itself a leaf.  The
    // seed is its gradient — through its hooks, as AccumulateGrad would run
    // them; there is no backward graph to walk.
    if (!root->grad_fn()) {
        accumulate_leaf(*root, std::move(seed));
        return;
    }

    // Optionally fuse adjacent backward nodes (e.g. LinearBackward + ReluBackward)
    // into a single fused node before traversal.
    run_fusion_pass(root->grad_fn().get());

    // Build the execution order once; reuse it for the accumulation loop.
    auto order = topo_order(root->grad_fn());

    // pending maps each node to the gradient that has been accumulated for it
    // so far.  Gradients from multiple edges pointing to the same node are
    // summed here before apply() is invoked.  held keeps the gradients of
    // hooked barrier slots until the barrier's turn (see Delivery above).
    Pending pending;
    HeldGrads<Storage> held;
    Node* const root_fn = root->grad_fn().get();
    const std::uint32_t root_slot = root->grad_output_nr();
    route(pending, held, root_fn, root_slot, std::move(seed), has_slot_hooks(*root_fn, root_slot));

    for (const auto& node : order) {
        auto it = pending.find(node.get());
        if (it == pending.end()) {
            // This node is not reachable from the root through the pending
            // map — it had no gradient contribution, so skip it.
            continue;
        }
        Storage grad_in = std::move(it->second);
        pending.erase(it);

        // The hooks of the tensors this node produced, and retain_grad, see
        // their whole gradient — and may replace it — before the node runs on
        // it.  A node with no hooks pays a null test.
        SlotGrads<Storage> slots;
        if (node->is_barrier())
            slots = fire_held(*node, held);
        else if (node->tensor_hooks() != nullptr)
            grad_in = run_slot_hooks(*node, 0, std::move(grad_in));

        // Detect in-place mutations that would corrupt the backward pass, and
        // a second pass over freed saved state.  After the hooks, not before:
        // a hook can run a backward of its own through this node, which frees
        // what it saved, or write into a tensor it saved — checked first, the
        // node then read freed storage and took the process down.
        refuse_released(*node);
        node->validate_versions();

        std::vector<Storage> input_grads;
        if (node->is_barrier()) {
            for (auto& [slot, grad] : slots)
                node->accumulate_barrier_grad(slot, std::move(grad));
            input_grads = node->apply_barrier();
        } else {
            input_grads = node->apply(std::move(grad_in));
        }

        // Free saved forward tensors immediately unless the caller needs the
        // graph intact for a second backward call.
        if (!retain_graph)
            release(*node);

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
            Node* next = edges[i].node.get();
            if (next == nullptr)
                continue;
            route(pending, held, next, edges[i].input_nr, input_grads[i],
                  has_slot_hooks(*next, edges[i].input_nr));
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
// asked about.  Hooks run on the nodes the traversal visits — those on a path
// to a requested input — and retain_grad keeps what reaches them, as in
// backward().

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
// hooked module's output, one output of a multi-output Function) is held on
// its way in, from what reaches that tensor's own slot.  Capturing the
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
// *further* request sits upstream of it.  An AccumulateGrad has no edges, so
// it never propagates — executing one is precisely what writes a leaf's
// .grad, which this path never does.
bool propagates(const Node* node, const std::unordered_set<const Node*>& reaching) {
    for (const auto& edge : node->next_edges()) {
        if (edge.node && reaching.find(edge.node.get()) != reaching.end())
            return true;
    }
    return false;
}

// The leaf a captured AccumulateGrad stands for, when its hooks are to run
// on what is captured for it — as AccumulateGrad would run them before
// writing ``.grad``.
TensorImplPtr hooked_leaf(const Node& node) {
    const auto* acc = dynamic_cast<const AccumulateGrad*>(&node);
    if (acc == nullptr)
        return nullptr;
    TensorImplPtr leaf = acc->leaf().lock();
    return leaf && leaf->requires_grad() ? leaf : nullptr;
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

    // A leaf root differentiates to itself: dy/dy is the seed, through the
    // leaf's own hooks.
    if (!root->grad_fn()) {
        auto it = targets.orphans.find(root.get());
        if (it == targets.orphans.end())
            return results;
        if (create_graph) {
            const TensorImplPtr value = run_leaf_hooks_for_graph(
                *root, std::make_shared<TensorImpl>(std::move(seed), root->shape(), root->dtype(),
                                                    root->device(), false));
            for (const std::size_t s : it->second)
                results[s] = value;
            return results;
        }
        const Storage value = run_leaf_hooks(*root, std::move(seed));
        for (const std::size_t s : it->second)
            results[s] = std::make_shared<TensorImpl>(value, root->shape(), root->dtype(),
                                                      root->device(), false);
        return results;
    }

    if (!create_graph)
        run_fusion_pass(root->grad_fn().get());
    const auto order = topo_order(root->grad_fn());
    const auto reaching = reaching_nodes(order, targets);
    // Off every path to a requested input, a node is sent nothing: none of it
    // runs in this pass, and a barrier handed a gradient it never ran on kept
    // it — a multi-output Function, which does not discard an earlier pass's
    // gradients, added them into the next backward()'s.
    const auto on_path = [&](const Node* node) {
        return node != nullptr && reaching.find(node) != reaching.end();
    };

    // The barrier slots whose gradient is held (see Delivery): those whose
    // hooks run in this pass, those a requested input reads, and every slot
    // of a barrier that will not run — for the same reason as above.
    const auto hold_at = [&](const Node* next, std::uint32_t slot) {
        return next->is_barrier() &&
               (has_slot_hooks(*next, slot) || captures_at_barrier(next, slot, targets, inputs) ||
                !propagates(next, reaching));
    };

    Node* const root_fn = root->grad_fn().get();
    const std::uint32_t root_slot = root->grad_output_nr();

    if (create_graph) {
        auto seed_impl = std::make_shared<TensorImpl>(std::move(seed), root->shape(), root->dtype(),
                                                      root->device(), false);
        GraphPending pending;
        HeldGrads<TensorImplPtr> held;
        if (on_path(root_fn))
            route_for_graph(pending, held, root_fn, root_slot, seed_impl,
                            hold_at(root_fn, root_slot));

        for (const auto& node : order) {
            auto it = pending.find(node.get());
            if (it == pending.end())
                continue;
            TensorImplPtr grad_in = std::move(it->second);
            pending.erase(it);
            // Off every path to a requested input: nothing of it runs.
            if (!on_path(node.get()))
                continue;

            const bool runs = propagates(node.get(), reaching);

            // Hooks before the checks, as in backward().
            SlotGrads<TensorImplPtr> slots;
            if (node->is_barrier())
                slots = fire_held_for_graph(*node, held);
            else if (node->tensor_hooks() != nullptr)
                grad_in = run_slot_hooks_for_graph(*node, 0, std::move(grad_in));

            if (const auto* wanted = capture_slots(node.get(), targets)) {
                const TensorImplPtr leaf = hooked_leaf(*node);
                TensorImplPtr leaf_grad;
                for (const std::size_t s : *wanted) {
                    TensorImplPtr captured = grad_in;
                    if (node->is_barrier()) {
                        const auto* held_grad = held_value(slots, inputs[s]->grad_output_nr());
                        captured = held_grad == nullptr ? nullptr : *held_grad;
                    }
                    if (leaf && captured) {
                        // Once, however many times the leaf is asked for.
                        if (!leaf_grad)
                            leaf_grad = run_leaf_hooks_for_graph(
                                *leaf, gradient_in_dtype_of(captured, leaf));
                        captured = leaf_grad;
                    }
                    results[s] = gradient_in_dtype_of(captured, inputs[s]);
                }
            }
            if (!runs)
                continue;

            refuse_released(*node);
            node->validate_versions();
            node->restore_saved_for_graph();
            if (node->is_barrier())
                deliver_held_for_graph(*node, std::move(slots), grad_in);
            const auto input_grads =
                keep_in_graph(apply_node_for_graph(*node, grad_in), node->next_edges());

            const auto& edges = node->next_edges();
            for (std::size_t i = 0; i < input_grads.size() && i < edges.size(); ++i) {
                Node* next = edges[i].node.get();
                if (on_path(next))
                    route_for_graph(pending, held, next, edges[i].input_nr, input_grads[i],
                                    hold_at(next, edges[i].input_nr));
            }
        }
        return results;
    }

    Pending pending;
    HeldGrads<Storage> held;
    if (on_path(root_fn))
        route(pending, held, root_fn, root_slot, std::move(seed), hold_at(root_fn, root_slot));

    for (const auto& node : order) {
        auto it = pending.find(node.get());
        if (it == pending.end())
            continue;
        Storage grad_in = std::move(it->second);
        pending.erase(it);
        // Off every path to a requested input: nothing of it runs.
        if (!on_path(node.get()))
            continue;

        const bool runs = propagates(node.get(), reaching);

        // Hooks before the checks, as in backward().
        SlotGrads<Storage> slots;
        if (node->is_barrier())
            slots = fire_held(*node, held);
        else if (node->tensor_hooks() != nullptr)
            grad_in = run_slot_hooks(*node, 0, std::move(grad_in));

        if (const auto* wanted = capture_slots(node.get(), targets)) {
            const TensorImplPtr leaf = hooked_leaf(*node);
            std::optional<Storage> leaf_grad;
            for (const std::size_t s : *wanted) {
                const auto& inp = inputs[s];
                const Storage* captured = &grad_in;
                if (node->is_barrier())
                    captured = held_value(slots, inp->grad_output_nr());
                if (captured == nullptr)
                    continue;
                if (leaf) {
                    // Once, however many times the leaf is asked for.
                    if (!leaf_grad)
                        leaf_grad = run_leaf_hooks(*leaf, *captured);
                    captured = &*leaf_grad;
                }
                // A barrier that runs is handed the held buffer too, and what
                // it does with it is its own business.
                Storage value = runs && node->is_barrier() ? own_grad_copy(*captured) : *captured;
                results[s] = std::make_shared<TensorImpl>(std::move(value), inp->shape(),
                                                          inp->dtype(), inp->device(), false);
            }
        }
        if (!runs)
            continue;

        refuse_released(*node);
        node->validate_versions();
        std::vector<Storage> input_grads;
        if (node->is_barrier()) {
            for (auto& [slot, grad] : slots)
                node->accumulate_barrier_grad(slot, std::move(grad));
            input_grads = node->apply_barrier();
        } else {
            input_grads = node->apply(std::move(grad_in));
        }

        if (!retain_graph)
            release(*node);

        const auto& edges = node->next_edges();
        for (std::size_t i = 0; i < input_grads.size() && i < edges.size(); ++i) {
            Node* next = edges[i].node.get();
            // As in ``backward``: from these nodes an empty storage is a
            // ``None``, not a value.
            if (!on_path(next) || (node->empty_grad_is_none() && storage_is_empty(input_grads[i])))
                continue;
            route(pending, held, next, edges[i].input_nr, std::move(input_grads[i]),
                  hold_at(next, edges[i].input_nr));
        }
    }

    if (!retain_graph)
        release_root(root);

    return results;
}

}  // namespace lucid
