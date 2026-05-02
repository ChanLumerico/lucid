#include "OpRegistry.h"

#include <map>
#include <mutex>
#include <shared_mutex>

namespace lucid {

namespace {

struct Registry {
    std::map<std::string_view, const OpSchema*> map;
    mutable std::shared_mutex mu;
};

Registry& storage() {
    static Registry r;
    return r;
}

}  // namespace

void OpRegistry::register_op(const OpSchema& schema) {
    auto& r = storage();
    std::unique_lock lock(r.mu);
    r.map[schema.name] = &schema;
}

const OpSchema* OpRegistry::lookup(std::string_view name) {
    auto& r = storage();
    std::shared_lock lock(r.mu);
    auto it = r.map.find(name);
    return it == r.map.end() ? nullptr : it->second;
}

std::vector<const OpSchema*> OpRegistry::all() {
    auto& r = storage();
    std::shared_lock lock(r.mu);
    std::vector<const OpSchema*> out;
    out.reserve(r.map.size());
    for (const auto& [_, schema] : r.map)
        out.push_back(schema);
    return out;
}

std::size_t OpRegistry::size() {
    auto& r = storage();
    std::shared_lock lock(r.mu);
    return r.map.size();
}

std::uint64_t schema_hash(const OpSchema& s) {
    constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ull;
    constexpr std::uint64_t kFnvPrime = 0x100000001b3ull;

    std::uint64_t h = kFnvOffset;
    auto mix = [&](std::uint8_t b) {
        h ^= b;
        h *= kFnvPrime;
    };

    for (char c : s.name)
        mix(static_cast<std::uint8_t>(c));
    mix(0);
    for (int i = 0; i < 4; ++i)
        mix(static_cast<std::uint8_t>((s.version >> (i * 8)) & 0xff));
    mix(static_cast<std::uint8_t>(s.amp_policy));
    mix(s.deterministic ? 1 : 0);
    return h;
}

}  // namespace lucid
