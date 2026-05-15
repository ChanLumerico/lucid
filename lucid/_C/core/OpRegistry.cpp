// lucid/_C/core/OpRegistry.cpp
//
// Thread-safe implementation of OpRegistry backed by a std::map<string_view,
// const OpSchema*> guarded by a std::shared_mutex.
//
// The registry is a function-local static (storage()) so that it is
// initialised on first access rather than during the dynamic-initialisation
// phase of the shared library.  This avoids the "static initialisation order
// fiasco" for LUCID_REGISTER_OP macros in other translation units: the map is
// guaranteed to exist before any registration call runs because the
// registration call itself triggers the map's construction.
//
// schema_hash() is also implemented here because it is logically related to
// the registry's change-detection responsibility.

#include "OpRegistry.h"

#include <map>
#include <mutex>
#include <shared_mutex>

namespace lucid {

namespace {

// Holds the mapping and its read-write lock.  Stored as a function-local
// static to guarantee construction before first use (avoids the static-init
// fiasco for translation units that register ops before main() runs).
//
// The map key is std::string_view, which is safe here because all keys point
// into OpSchema::name — a string_view of a constexpr string literal that has
// static storage duration and will never be freed.  Do not insert keys backed
// by temporaries.
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
    // Exclusive lock: prevents concurrent readers from observing a partially
    // constructed map entry.  Registration is rare (happens at static init
    // time), so write contention is not a practical concern.
    std::unique_lock lock(r.mu);
    // schema.name is a string_view into a static/constexpr storage; the
    // map key therefore remains valid for the process lifetime.
    r.map[schema.name] = &schema;
}

const OpSchema* OpRegistry::lookup(std::string_view name) {
    auto& r = storage();
    // Shared lock: multiple concurrent lookups are safe.
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

// FNV-1a 64-bit hash of the schema's name, version bytes, amp_policy byte,
// and deterministic flag.  The null byte between name and version acts as a
// separator to prevent length-extension collisions.
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
    // Null separator between name and version to prevent "ab" from hashing
    // the same as "a" followed by version starting with 'b'.
    mix(0);
    // Hash version as 4 little-endian bytes.
    for (int i = 0; i < 4; ++i)
        mix(static_cast<std::uint8_t>((s.version >> (i * 8)) & 0xff));
    mix(static_cast<std::uint8_t>(s.amp_policy));
    mix(s.deterministic ? 1 : 0);
    return h;
}

}  // namespace lucid
