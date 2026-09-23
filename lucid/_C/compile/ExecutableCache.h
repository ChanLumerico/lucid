// lucid/_C/compile/ExecutableCache.h
//
// Process-global cache of compiled MPSGraph executables, keyed by the
// structural signature of the trace that produced them.  Two traces
// that emit the same op sequence with the same per-op output shapes
// and dtypes on the same device share one executable.
//
// The cache is the owner of each :class:`CompiledExecutable` inserted
// into it: eviction (LRU when ``max_entries`` is exceeded) and
// :func:`clear` both release the entries via
// :func:`destroy_executable`.
//
// Notes
// -----
// Phase 1.2 ships an LRU implementation with default ``max_entries =
// 32``, override-able by setting ``LUCID_COMPILE_MAX_CACHE=N`` in the
// environment before the first :func:`session` call.
//
// The cache is thread-safe (one mutex protects the underlying map +
// LRU vector); calls from the autograd Engine or DataLoader workers
// share one cache.

#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "CompiledExecutable.h"  // forward + destroy_executable
#include "TraceIR.h"

namespace lucid::compile {

// Cache key — the structural signature of a :class:`TraceGraph`.
//
// Two traces share an executable iff every field below matches.  The
// input *values* are not part of the key; only the per-op output meta
// influences the compiled graph (because MPSGraph placeholders are
// shape-static in Phase 1.2 and Phase 1.6 will introduce a separate
// ``dynamic`` key variant).
//
// Attributes
// ----------
// op_names : std::vector<std::string>
//     Op-name sequence in dispatch order.  Compared element-wise.
// op_inputs : std::vector<std::vector<std::int64_t>>
//     Per-op input wiring: every tensor id renumbered by first appearance
//     across the walk, so the numbers depend only on how the graph is
//     connected and not on the ids a particular trace drew.  ``a * b``
//     fed one tensor twice reads ``{0, 0}``, fed two tensors ``{0, 1}``.
//     Without this both hashed alike and one caller ran the other's
//     executable — one feed bound where two were passed.
// output_shapes : std::vector<Shape>
//     Per-op single-output shape.  Phase 1.2 supports one output per
//     node; Phase 1.3+ may append entries for multi-output ops.
// output_dtypes : std::vector<Dtype>
//     Per-op single-output dtype.
// op_attrs : std::vector<std::vector<std::pair<std::string, AttributeValue>>>
//     Per-op attribute map, serialised in sorted-key order so the
//     hash is deterministic.  Required for emitter-divergent
//     attributes that don't show up in shapes (e.g. ``dropout``'s
//     ``training`` / ``p``, ``softmax``'s ``axis``).  Without this
//     two traces with identical structure but different attributes
//     would collide on the same cache entry and one caller would
//     silently receive the other's executable.
// feeds : std::vector<FeedSig>
//     Shape + dtype of every external feed (user input or parameter),
//     ordered by the same first-appearance numbering as ``op_inputs``.
//     The executable's placeholders are typed and shaped by its feeds and
//     ``run_executable`` refuses a mismatch, so two traces whose ops agree
//     but whose inputs do not must not share an entry.  Without this,
//     ``isnan(x) + 1`` traced on float32 and then on int64 — every op
//     output the same shape and dtype — hit the float executable and
//     raised "feed slot 0 expects dtype float32, got int64".
// outputs : std::vector<std::int64_t>
//     The tensors the caller asked for, in order and in the same numbering.
//     The graph alone does not say which of its values are returned: two
//     callables tracing ``(t + 1, t * 2)`` and ``(t * 2, t + 1)`` build the
//     same ops, and without this the second ran the first's executable and
//     got its outputs in the first's order — as ``topk(t)[0]`` got the
//     indices of an earlier ``topk(t)[1]``.
// device : Device
//     Device on which the executable was compiled (all ops in a trace
//     must agree on one device — Phase 1.2 builder enforces this).
struct LUCID_API FeedSig {
    std::int64_t canonical_id = 0;
    Shape shape;
    Dtype dtype = Dtype::F32;
    bool operator==(const FeedSig& other) const noexcept {
        return canonical_id == other.canonical_id && shape == other.shape && dtype == other.dtype;
    }
};

struct LUCID_API CacheKey {
    std::vector<std::string> op_names;
    std::vector<std::vector<std::int64_t>> op_inputs;
    std::vector<Shape> output_shapes;
    std::vector<Dtype> output_dtypes;
    std::vector<std::vector<std::pair<std::string, AttributeValue>>> op_attrs;
    std::vector<FeedSig> feeds;
    std::vector<std::int64_t> outputs;
    Device device = Device::CPU;

    // Structural equality — every field above must match element-wise.
    // Used as the unordered_map bucket-equality predicate.
    bool operator==(const CacheKey& other) const noexcept;
};

// std-style hasher for :class:`CacheKey`; used as the bucket function
// for :class:`ExecutableCache`'s internal map.
struct LUCID_API CacheKeyHash {
    // Hash combine over op_names + output_shapes + output_dtypes +
    // device.  Two structurally-equal keys produce the same hash.
    std::size_t operator()(const CacheKey& key) const noexcept;
};

// Build a :class:`CacheKey` from a :class:`TraceGraph`.
//
// Walks ``graph.ops`` once, copying the name + first-output meta from
// each node.  Device is taken from the most recent non-empty output;
// it is the caller's responsibility to ensure the trace is
// single-device (the MpsBuilder rejects mixed-device traces before
// hashing).
//
// ``feed_meta`` maps each external-feed id to its shape and dtype; it
// fills :attr:`CacheKey::feeds`.  A feed no op reads is left out.
// ``explicit_outputs`` are the trace ids the caller will read back, in
// order; they fill :attr:`CacheKey::outputs`.
LUCID_API CacheKey
make_cache_key(const TraceGraph& graph,
               const std::unordered_map<TensorId, std::pair<Shape, Dtype>>& feed_meta = {},
               const std::vector<TensorId>& explicit_outputs = {});

// Bounded-LRU cache of compiled MPSGraph executables.
//
// Thread-safe.  Stores raw :class:`CompiledExecutable` pointers and
// destroys them via :func:`destroy_executable` on eviction or
// destruction.
class LUCID_API ExecutableCache {
public:
    // Construct an empty cache with the default LRU cap (32 entries).
    // Use :func:`set_max_entries` to override before inserting.
    ExecutableCache() = default;

    // Destructor — releases every cached :class:`CompiledExecutable`
    // via :func:`destroy_executable` before the map is torn down.
    ~ExecutableCache();

    ExecutableCache(const ExecutableCache&) = delete;
    ExecutableCache& operator=(const ExecutableCache&) = delete;
    ExecutableCache(ExecutableCache&&) = delete;
    ExecutableCache& operator=(ExecutableCache&&) = delete;

    // Look up an executable.
    //
    // Returns the borrowed pointer on hit (and bumps the entry to
    // most-recent in the LRU order) or ``nullptr`` on miss.
    CompiledExecutable* find(const CacheKey& key);

    // Insert ``exec`` under ``key``, taking ownership.  Replaces any
    // existing entry (releasing the previous executable).  If the
    // resulting size exceeds ``max_entries``, evicts least-recently
    // used entries until back at capacity.
    void insert(CacheKey key, CompiledExecutable* exec);

    // Release every stored executable.
    void clear();

    // Number of currently-cached entries.
    std::size_t size() const;

    // Override the LRU cap.  Honoured by subsequent inserts; existing
    // entries are not evicted retroactively.  Default is 32.
    void set_max_entries(std::size_t n);

    // Process-global session cache.  First call honours the
    // ``LUCID_COMPILE_MAX_CACHE`` environment variable (positive
    // integer).  Subsequent calls return the same instance.
    static ExecutableCache& session();

private:
    mutable std::mutex mu_;
    std::unordered_map<CacheKey, CompiledExecutable*, CacheKeyHash> map_;
    // Most-recent at the back; front = next eviction candidate.
    std::vector<CacheKey> lru_order_;
    std::size_t max_entries_ = 32;
};

}  // namespace lucid::compile
