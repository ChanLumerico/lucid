#pragma once

// =====================================================================
// Lucid C++ engine — autograd Node + Edge.
// =====================================================================
//
// Node is the abstract base of every backward operation. Concrete subclasses
// (Phase 3) are CRTP children of FuncOp<D, N_IN>, which provides the boring
// glue (gather_edges, broadcast-undo) so each op writes only its math.
//
// Edges hold weak_ptr<Node> to break the parent→child reference cycle. Owners
// of Node are: TensorImpl::grad_fn_ (one strong ref per result tensor), and
// the engine's local `pending` map during backward (transient).
//
// Layer: autograd/. May depend on core/.

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"

namespace lucid {

class Node;

struct LUCID_API Edge {
    // Strong ref. The autograd graph keeps grad_fns alive even if the
    // intermediate TensorImpls that produced them are released — required
    // for inline composition like `relu(linear(x, W, b))` where the
    // intermediate `linear(...)` TensorImpl dies before backward runs.
    //
    // Cycle prevention: grad_fns never hold strong refs *back* to
    // TensorImpls (AccumulateGrad uses `weak_ptr<TensorImpl>`; FuncOp's
    // `input_tensors_` are weak_ptr too). Storage in saved_inputs_/
    // saved_output_ is independent of TensorImpl lifetime (Storage holds
    // its own shared_ptr to the byte buffer).
    std::shared_ptr<Node> node;
    std::uint32_t input_nr = 0;

    Edge() = default;
    Edge(std::shared_ptr<Node> n, std::uint32_t i = 0)
        : node(std::move(n)), input_nr(i) {}

    bool is_valid() const { return node != nullptr; }
};

class LUCID_API Node : public std::enable_shared_from_this<Node> {
public:
    Node();
    virtual ~Node() = default;

    // Backward step: takes the gradient flowing into this node's output and
    // produces gradients to send to each input (one Storage per next_edge,
    // in order).
    virtual std::vector<Storage> apply(Storage grad_out) = 0;

    // Hook called by Engine::backward right before `apply`. Default no-op;
    // FuncOp<Derived, N_IN> overrides to verify that saved input version_
    // counters still match the live tensors. Throws lucid::VersionMismatch
    // if an in-place op mutated an input between forward and backward.
    virtual void validate_versions() {}

    // Topological identifier. Higher = later in forward = earlier in backward.
    std::uint64_t sequence_nr() const { return sequence_nr_; }
    void set_sequence_nr(std::uint64_t n) { sequence_nr_ = n; }

    const std::vector<Edge>& next_edges() const { return next_edges_; }
    void set_next_edges(std::vector<Edge> edges) {
        next_edges_ = std::move(edges);
    }

    // Saved input/output versions to detect in-place mutation between forward
    // and backward, mirroring the Python `_version` check.
    const std::vector<std::int64_t>& saved_versions() const {
        return saved_versions_;
    }
    void set_saved_versions(std::vector<std::int64_t> v) {
        saved_versions_ = std::move(v);
    }

protected:
    std::uint64_t sequence_nr_;
    std::vector<Edge> next_edges_;
    std::vector<std::int64_t> saved_versions_;
};

LUCID_API std::uint64_t next_sequence_nr();

}  // namespace lucid
