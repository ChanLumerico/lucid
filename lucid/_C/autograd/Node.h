// lucid/_C/autograd/Node.h
//
// Defines the two foundational autograd types: Edge and Node.  Every operation
// that participates in a define-by-run computation graph creates a Node
// subclass and links it to its inputs through a vector of Edges.  After the
// forward pass the engine walks this graph in reverse topological order,
// calling apply() on each node to propagate gradients.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"

namespace lucid {

class Node;

// A directed edge in the backward graph.
//
// An Edge connects a producer node to a specific input slot (input_nr) of a
// consumer node.  The shared_ptr keeps the consumer alive as long as at least
// one producer holds a reference to it, forming an ownership chain that
// mirrors the forward computation.  input_nr is the positional index into
// the consumer's apply() output vector where the arriving gradient should
// land.
struct LUCID_API Edge {
    std::shared_ptr<Node> node;
    std::uint32_t input_nr = 0;

    Edge() = default;
    Edge(std::shared_ptr<Node> n, std::uint32_t i = 0) : node(std::move(n)), input_nr(i) {}

    // Returns true when this edge points to a live node.
    bool is_valid() const { return node != nullptr; }
};

// Abstract base class for every node in the backward computation graph.
//
// Node is the unit of work the Engine executes during backward().  Each
// concrete subclass saves whatever tensors/scalars it needs during the forward
// pass (via its own fields) and implements apply() to compute the input
// gradients from the incoming output gradient.
//
// Lifetime: nodes are jointly owned by the TensorImpl that produced them
// (via grad_fn) and by the Edges of their consumer nodes.  Once backward()
// completes and grad_fn is cleared on the root, the chain of shared_ptr
// references collapses and all temporary nodes are destroyed.
//
// Thread safety: Node is not thread-safe.  The Engine runs the backward graph
// on a single thread; concurrent backward() calls on the same graph are
// undefined behavior.
class LUCID_API Node : public std::enable_shared_from_this<Node> {
public:
    Node();
    virtual ~Node() = default;

    // Compute input gradients from the output gradient grad_out.
    //
    // Returns one Storage per incoming edge (i.e. per input to the forward
    // op).  A null/empty Storage at position i means no gradient flows
    // to that input.  Implementations may assume grad_out is non-null.
    virtual std::vector<Storage> apply(Storage grad_out) = 0;

    // Assert that no saved input tensor has been modified in-place since the
    // forward pass.  The default is a no-op; AutogradNode overrides this to
    // check version counters.
    virtual void validate_versions() {}

    // Free all tensors saved for backward (inputs, outputs, etc.) once the
    // engine has finished executing this node.  Subclasses override to reset
    // their saved Storage arrays to empty CpuStorage.  Called after apply()
    // when retain_graph is false.
    virtual void release_saved() {}

    // Monotonically increasing integer assigned at construction time.
    // The engine uses sequence numbers to break topological ties so that
    // nodes closer to the root are executed first.
    std::uint64_t sequence_nr() const { return sequence_nr_; }
    void set_sequence_nr(std::uint64_t n) { sequence_nr_ = n; }

    // The outgoing edges of this node — one per input tensor of the
    // corresponding forward operation.  Populated at forward time by the op
    // builder, then read-only during backward.
    const std::vector<Edge>& next_edges() const { return next_edges_; }
    void set_next_edges(std::vector<Edge> edges) { next_edges_ = std::move(edges); }

    // Version numbers of input tensors recorded at forward time.
    // Compared against the live tensors in validate_versions() to detect
    // in-place mutations that would corrupt the gradient computation.
    const std::vector<std::int64_t>& saved_versions() const { return saved_versions_; }
    void set_saved_versions(std::vector<std::int64_t> v) { saved_versions_ = std::move(v); }

protected:
    std::uint64_t sequence_nr_;
    std::vector<Edge> next_edges_;
    std::vector<std::int64_t> saved_versions_;
};

// Return the next globally unique sequence number and advance the counter.
// Thread-safe: uses relaxed atomic increment (ordering between different
// nodes' assignments is not required to be sequentially consistent).
LUCID_API std::uint64_t next_sequence_nr();

}  // namespace lucid
