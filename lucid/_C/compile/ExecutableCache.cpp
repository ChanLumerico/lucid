// lucid/_C/compile/ExecutableCache.cpp
//
// Pure-C++ implementation of the executable cache.  Holds raw
// :class:`CompiledExecutable` pointers (the Objective-C++ object is
// hidden behind ``void*`` in :file:`CompiledExecutable.mm`) and routes
// eviction through :func:`destroy_executable`, which is the free
// function defined in the .mm to release the ARC-strong reference.

#include "ExecutableCache.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

namespace lucid::compile {

bool CacheKey::operator==(const CacheKey& other) const noexcept {
    return device == other.device && op_names == other.op_names &&
           output_shapes == other.output_shapes && output_dtypes == other.output_dtypes &&
           op_attrs == other.op_attrs;
}

namespace {
// Boost-style hash combiner — small enough to inline here and used
// uniformly across every CacheKey field.
inline void hash_combine(std::size_t& seed, std::size_t v) noexcept {
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

// Hash a single :type:`AttributeValue`.  The variant is small (i64 /
// vector<i64> / double / bool / string) so a switch over alternatives
// is cheaper than a polymorphic visitor here.
inline std::size_t hash_attribute_value(const AttributeValue& v) noexcept {
    if (const auto* p = std::get_if<std::int64_t>(&v))
        return std::hash<std::int64_t>{}(*p);
    if (const auto* p = std::get_if<std::vector<std::int64_t>>(&v)) {
        std::size_t h = 0xA110C8ULL;
        for (auto d : *p)
            hash_combine(h, std::hash<std::int64_t>{}(d));
        return h;
    }
    if (const auto* p = std::get_if<double>(&v))
        return std::hash<double>{}(*p);
    if (const auto* p = std::get_if<bool>(&v))
        return std::hash<bool>{}(*p);
    if (const auto* p = std::get_if<std::string>(&v))
        return std::hash<std::string>{}(*p);
    return 0;
}
}  // namespace

std::size_t CacheKeyHash::operator()(const CacheKey& key) const noexcept {
    std::size_t h = std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(key.device));
    for (const auto& name : key.op_names)
        hash_combine(h, std::hash<std::string>{}(name));
    for (const auto& shape : key.output_shapes) {
        for (auto d : shape)
            hash_combine(h, std::hash<std::int64_t>{}(d));
        // Trailing sentinel so [a, b] and [a*b] hash differently.
        hash_combine(h, 0xCAFEBABEULL);
    }
    for (auto dt : key.output_dtypes)
        hash_combine(h, std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(dt)));
    // Attributes — sorted-key serialisation keeps the hash
    // deterministic across runs.  Sentinel between ops prevents
    // [{a:1}, {b:2}] colliding with [{a:1,b:2}, {}].
    for (const auto& per_op : key.op_attrs) {
        for (const auto& [k, v] : per_op) {
            hash_combine(h, std::hash<std::string>{}(k));
            hash_combine(h, hash_attribute_value(v));
        }
        hash_combine(h, 0xDEADBEEFULL);
    }
    return h;
}

CacheKey make_cache_key(const TraceGraph& graph) {
    CacheKey key;
    key.op_names.reserve(graph.ops.size());
    key.output_shapes.reserve(graph.ops.size());
    key.output_dtypes.reserve(graph.ops.size());
    key.op_attrs.reserve(graph.ops.size());

    Device device = Device::CPU;
    for (const auto& node : graph.ops) {
        key.op_names.push_back(node.name);
        if (!node.outputs.empty()) {
            const auto& meta = node.outputs[0];
            key.output_shapes.push_back(meta.shape);
            key.output_dtypes.push_back(meta.dtype);
            device = meta.device;
        } else {
            key.output_shapes.emplace_back();
            key.output_dtypes.push_back(Dtype::F32);
        }
        // Serialise attrs in sorted-key order so cache lookup is
        // independent of std::unordered_map insertion order.
        std::vector<std::pair<std::string, AttributeValue>> sorted_attrs;
        sorted_attrs.reserve(node.attrs.size());
        for (const auto& kv : node.attrs)
            sorted_attrs.emplace_back(kv.first, kv.second);
        std::sort(sorted_attrs.begin(), sorted_attrs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        key.op_attrs.push_back(std::move(sorted_attrs));
    }
    key.device = device;
    return key;
}

ExecutableCache::~ExecutableCache() {
    clear();
}

CompiledExecutable* ExecutableCache::find(const CacheKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = map_.find(key);
    if (it == map_.end())
        return nullptr;

    // Bump the matching entry to the back of the LRU vector (most
    // recent).  Linear scan is acceptable because ``max_entries_`` is
    // O(10–100).
    auto vit = std::find(lru_order_.begin(), lru_order_.end(), key);
    if (vit != lru_order_.end()) {
        CacheKey saved = std::move(*vit);
        lru_order_.erase(vit);
        lru_order_.push_back(std::move(saved));
    }
    return it->second;
}

void ExecutableCache::insert(CacheKey key, CompiledExecutable* exec) {
    std::lock_guard<std::mutex> lock(mu_);

    // Replace existing entry under the same key, releasing the prior
    // executable.
    auto it = map_.find(key);
    if (it != map_.end()) {
        destroy_executable(it->second);
        it->second = exec;
    } else {
        map_.emplace(key, exec);
    }

    // Bring the key to the front of "most-recent".
    auto vit = std::find(lru_order_.begin(), lru_order_.end(), key);
    if (vit != lru_order_.end())
        lru_order_.erase(vit);
    lru_order_.push_back(std::move(key));

    // Evict from the front until back at capacity.
    while (lru_order_.size() > max_entries_) {
        const CacheKey victim = std::move(lru_order_.front());
        lru_order_.erase(lru_order_.begin());
        auto mit = map_.find(victim);
        if (mit != map_.end()) {
            destroy_executable(mit->second);
            map_.erase(mit);
        }
    }
}

void ExecutableCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, exec] : map_)
        destroy_executable(exec);
    map_.clear();
    lru_order_.clear();
}

std::size_t ExecutableCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return map_.size();
}

void ExecutableCache::set_max_entries(std::size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    if (n == 0)
        n = 1;
    max_entries_ = n;
}

ExecutableCache& ExecutableCache::session() {
    // Heap-allocated singleton — process-lifetime cache, no shutdown
    // ordering issues with the rest of the engine.
    static ExecutableCache* instance = [] {
        auto* c = new ExecutableCache();
        if (const char* s = std::getenv("LUCID_COMPILE_MAX_CACHE")) {
            char* end = nullptr;
            unsigned long v = std::strtoul(s, &end, 10);
            if (end != s && v > 0)
                c->set_max_entries(static_cast<std::size_t>(v));
        }
        return c;
    }();
    return *instance;
}

}  // namespace lucid::compile
