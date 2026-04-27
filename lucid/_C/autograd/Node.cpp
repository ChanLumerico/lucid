#include "Node.h"

namespace lucid {

namespace {
std::atomic<std::uint64_t> g_seq{0};
}

std::uint64_t next_sequence_nr() {
    return g_seq.fetch_add(1, std::memory_order_relaxed);
}

Node::Node() : sequence_nr_(next_sequence_nr()) {}

}  // namespace lucid
