#include "OpRegistry.h"

#include <map>

namespace lucid {

namespace {

// Construct-on-first-use to dodge the static-init-order fiasco. Multiple
// translation units can register before main() in any order.
std::map<std::string_view, const OpSchema*>& storage() {
    static std::map<std::string_view, const OpSchema*> m;
    return m;
}

}  // namespace

void OpRegistry::register_op(const OpSchema& schema) {
    storage()[schema.name] = &schema;
}

const OpSchema* OpRegistry::lookup(std::string_view name) {
    auto& m = storage();
    auto it = m.find(name);
    return it == m.end() ? nullptr : it->second;
}

std::vector<const OpSchema*> OpRegistry::all() {
    auto& m = storage();
    std::vector<const OpSchema*> out;
    out.reserve(m.size());
    for (const auto& [_, schema] : m)
        out.push_back(schema);
    return out;
}

std::size_t OpRegistry::size() {
    return storage().size();
}

// Schema hash: FNV-1a over (name, version, amp_policy, deterministic). Stable
// across architectures because we use little-endian-equivalent byte order
// into the hash regardless of host.
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
    mix(0);  // separator
    for (int i = 0; i < 4; ++i)
        mix(static_cast<std::uint8_t>((s.version >> (i * 8)) & 0xff));
    mix(static_cast<std::uint8_t>(s.amp_policy));
    mix(s.deterministic ? 1 : 0);
    return h;
}

}  // namespace lucid
