#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"

namespace lucid {

class Node;

struct LUCID_API Edge {
    std::shared_ptr<Node> node;
    std::uint32_t input_nr = 0;

    Edge() = default;
    Edge(std::shared_ptr<Node> n, std::uint32_t i = 0) : node(std::move(n)), input_nr(i) {}

    bool is_valid() const { return node != nullptr; }
};

class LUCID_API Node : public std::enable_shared_from_this<Node> {
public:
    Node();
    virtual ~Node() = default;

    virtual std::vector<Storage> apply(Storage grad_out) = 0;

    virtual void validate_versions() {}

    virtual void release_saved() {}

    std::uint64_t sequence_nr() const { return sequence_nr_; }
    void set_sequence_nr(std::uint64_t n) { sequence_nr_ = n; }

    const std::vector<Edge>& next_edges() const { return next_edges_; }
    void set_next_edges(std::vector<Edge> edges) { next_edges_ = std::move(edges); }

    const std::vector<std::int64_t>& saved_versions() const { return saved_versions_; }
    void set_saved_versions(std::vector<std::int64_t> v) { saved_versions_ = std::move(v); }

protected:
    std::uint64_t sequence_nr_;
    std::vector<Edge> next_edges_;
    std::vector<std::int64_t> saved_versions_;
};

LUCID_API std::uint64_t next_sequence_nr();

}  // namespace lucid
