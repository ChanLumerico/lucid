// lucid/_C/autograd/Node.cpp
//
// Implements the global sequence-number counter and the Node constructor.
// The counter is a single process-wide atomic so that every node created
// in any thread receives a unique, monotonically increasing ID.

#include "Node.h"

namespace lucid {

namespace {
// Process-wide counter.  Relaxed ordering is sufficient because the only
// guarantee needed is uniqueness, not synchronisation with other memory.
std::atomic<std::uint64_t> g_seq{0};
}  // namespace

std::uint64_t next_sequence_nr() {
    return g_seq.fetch_add(1, std::memory_order_relaxed);
}

// Assign a sequence number at construction so that every node can be sorted
// deterministically by the engine even before next_edges_ is populated.
Node::Node() : sequence_nr_(next_sequence_nr()) {}

}  // namespace lucid
