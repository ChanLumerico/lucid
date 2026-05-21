// lucid/_C/core/MemoryStats.h
//
// Per-device allocation telemetry for the Lucid engine.
//
// :class:`MemoryTracker` maintains separate atomic counter banks for
// :attr:`Device::CPU` and :attr:`Device::GPU` allocations so that
// diagnostic tools and Python-side memory queries can report live
// usage, peak usage, and allocation / free counts without taking any
// locks.  :func:`MemoryTracker::track_alloc` and
// :func:`MemoryTracker::track_free` are fired from the custom deleter
// installed by :func:`allocate_aligned_bytes` (see ``Allocator.cpp``)
// and should not be called by application code directly.
//
// Notes
// -----
// All counter operations use ``std::memory_order_relaxed``.  The
// counters serve only as approximate diagnostics — there is no
// happens-before edge between a counter update and the compute work
// that follows it, so :func:`MemoryTracker::get_stats` snapshots are
// **eventually consistent** rather than a coherent view of a single
// instant.
//
// See Also
// --------
// :func:`allocate_aligned_bytes` — installs the hooks that drive these
//     counters.
// :class:`Device`                — tag selecting the counter bank.

#pragma once

#include <atomic>
#include <cstddef>

#include "../api.h"
#include "Device.h"

namespace lucid {

// Plain-data snapshot of one device's allocation counters.
//
// Returned by :func:`MemoryTracker::get_stats`.  Copying is trivial —
// the struct is a value-typed snapshot and not connected to the live
// atomic counters once returned.
//
// Attributes
// ----------
// current_bytes : std::size_t
//     Bytes currently held by the allocator on this device
//     (lifetime allocated minus lifetime freed).
// peak_bytes : std::size_t
//     High-water mark of ``current_bytes`` since process start or the
//     last :func:`MemoryTracker::reset_peak` call.
// alloc_count : std::size_t
//     Lifetime total of successful allocation events.
// free_count : std::size_t
//     Lifetime total of deallocation events.
struct LUCID_API MemoryStats {
    std::size_t current_bytes = 0;
    std::size_t peak_bytes = 0;
    std::size_t alloc_count = 0;
    std::size_t free_count = 0;
};

// Atomic per-device allocation counter bank.
//
// All counters live in static storage, one bank per :class:`Device`.
// Updates use ``std::atomic`` so :func:`track_alloc` / :func:`track_free`
// can be invoked from any thread without external locking.  Peak
// tracking uses a compare-exchange loop, avoiding a mutex while still
// guaranteeing a monotonically non-decreasing peak.
//
// Notes
// -----
// All public methods are ``static`` — the class is effectively a
// namespace over the two global counter banks.  There is no instance
// state.
//
// Thread Safety
// -------------
// Counter updates are atomic.  Snapshots returned by
// :func:`get_stats` are eventually consistent — a ``get_stats`` call
// during concurrent allocation may observe ``current_bytes`` slightly
// behind the true total or a peak that lags the true maximum by one
// CAS cycle.
class LUCID_API MemoryTracker {
public:
    // Live atomic counters for a single device.
    //
    // One instance is held in static storage per :class:`Device`.  Not
    // intended for direct use by application code — interact via the
    // static :class:`MemoryTracker` methods instead.
    //
    // Attributes
    // ----------
    // current_bytes : std::atomic<std::size_t>
    //     Bytes currently held by the allocator on this device.
    // peak_bytes : std::atomic<std::size_t>
    //     Monotonically non-decreasing high-water mark of
    //     ``current_bytes`` since the last :func:`reset_peak`.
    // alloc_count : std::atomic<std::size_t>
    //     Number of successful allocation events.
    // free_count : std::atomic<std::size_t>
    //     Number of deallocation events.
    struct Counters {
        std::atomic<std::size_t> current_bytes{0};
        std::atomic<std::size_t> peak_bytes{0};
        std::atomic<std::size_t> alloc_count{0};
        std::atomic<std::size_t> free_count{0};
    };

    // Records an allocation of ``nbytes`` on ``device``.
    //
    // Adds ``nbytes`` to ``current_bytes``, increments ``alloc_count``,
    // and — if the new ``current_bytes`` exceeds the previous peak —
    // advances ``peak_bytes`` via a relaxed compare-exchange loop.
    //
    // Parameters
    // ----------
    // nbytes : std::size_t
    //     Size of the allocation just performed.
    // device : Device
    //     Device whose counter bank should be updated.
    //
    // Notes
    // -----
    // Called from the allocator's deleter; not for direct application
    // use.
    static void track_alloc(std::size_t nbytes, Device device);

    // Records a deallocation of ``nbytes`` on ``device``.
    //
    // Subtracts ``nbytes`` from ``current_bytes`` and increments
    // ``free_count``.  Does not touch ``peak_bytes`` — peaks are
    // historical, by definition.
    //
    // Parameters
    // ----------
    // nbytes : std::size_t
    //     Size of the block being freed.  Must match the value passed
    //     to the corresponding :func:`track_alloc`.
    // device : Device
    //     Device whose counter bank should be updated.
    static void track_free(std::size_t nbytes, Device device);

    // Returns a coherent-looking snapshot of one device's counters.
    //
    // Each field is read separately with ``memory_order_relaxed``, so
    // the four values may not correspond to the same instant under
    // concurrent activity.  Sufficient for diagnostics; not safe to
    // base correctness invariants on.
    //
    // Parameters
    // ----------
    // device : Device
    //     Device whose counters should be sampled.
    //
    // Returns
    // -------
    // MemoryStats
    //     Value-typed snapshot disconnected from the live atomics.
    static MemoryStats get_stats(Device device);

    // Resets ``peak_bytes`` to the current ``current_bytes`` value.
    //
    // Useful at the start of a profiling window — after this call,
    // ``peak_bytes`` reports the maximum live byte total reached since
    // the reset rather than since process start.
    //
    // Parameters
    // ----------
    // device : Device
    //     Device whose peak counter should be cleared.
    //
    // Notes
    // -----
    // Mirrors the reference framework's ``reset_peak_memory_stats`` for
    // API parity.
    static void reset_peak(Device device);

private:
    // Returns the live :class:`Counters` bank for ``device``.
    //
    // Implementation detail — the two banks live in static storage
    // inside the ``.cpp`` file's anonymous namespace.
    //
    // Parameters
    // ----------
    // device : Device
    //     Device whose counter bank should be returned.
    //
    // Returns
    // -------
    // Counters&
    //     Reference to the static bank for the requested device.
    static Counters& counters_for(Device device);
};

}  // namespace lucid
