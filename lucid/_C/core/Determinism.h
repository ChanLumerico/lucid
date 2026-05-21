// lucid/_C/core/Determinism.h
//
// Process-wide determinism flag enforced by the op dispatcher.
//
// When determinism is enabled, every dispatched op is gated through
// :class:`SchemaGuard`, which consults the corresponding
// :class:`OpSchema` entry and rejects any op marked
// ``deterministic == false`` with a :class:`LucidError`.  This lets
// reproducibility-sensitive workflows (model training, gradient
// checking, parity testing) fail loudly the instant they would invoke
// a kernel whose output depends on scheduling or unordered atomics —
// rather than silently producing non-reproducible results.
//
// Notes
// -----
// Unlike :class:`GradMode` and the AMP policy — both of which are
// ``thread_local`` so that independent Python threads can hold
// independent settings — this flag is process-global, backed by a
// ``std::atomic<bool>``.  Reproducibility is typically a whole-process
// concern (the user sets a seed and a determinism mode once at start
// up), so per-thread granularity would be the wrong default.
//
// The atomic backing means :func:`is_enabled` and :func:`set_enabled`
// are individually safe to call from multiple threads concurrently;
// however, the two operations are not packaged into a check-and-set
// pair, so callers must not assume that the flag remains unchanged
// between a read and a subsequent action.
//
// See Also
// --------
// :class:`OpSchema` — per-op metadata that flags non-deterministic
//     kernels.
// :class:`SchemaGuard` — dispatcher hook that enforces the flag.

#pragma once

namespace lucid {

// Process-wide determinism control for the op dispatcher.
//
// All methods are ``static``: there is no instance state — the class
// is a namespace-style holder over a single ``std::atomic<bool>``
// defined in :file:`Determinism.cpp`.  Reads and writes are atomic and
// thread-safe but unordered with respect to each other.
//
// Notes
// -----
// The default value at process startup is ``false`` (non-deterministic
// kernels are permitted).  Reproducibility-sensitive callers should
// explicitly :func:`set_enabled(true)` together with seeding the
// default :class:`Generator`.
//
// See Also
// --------
// :func:`lucid.use_deterministic_algorithms` — Python wrapper.
// :class:`Generator` — companion RNG state required for full
//     reproducibility.
class Determinism {
public:
    // Returns whether deterministic-mode is currently active.
    //
    // Returns
    // -------
    // bool
    //     ``true`` when the dispatcher must reject non-deterministic
    //     ops, ``false`` when they are allowed.
    //
    // Notes
    // -----
    // Reading the flag is a single relaxed atomic load with no fence;
    // the value may have changed by the time the caller acts on it.
    static bool is_enabled();

    // Sets the process-wide determinism flag.
    //
    // Parameters
    // ----------
    // value : bool
    //     ``true`` to require deterministic-only execution (subsequent
    //     non-deterministic ops will raise :class:`LucidError`),
    //     ``false`` to permit all ops.
    //
    // Notes
    // -----
    // The write is atomic but unfenced with respect to ops already in
    // flight on other threads — those ops continue with whichever
    // value they observed at dispatch time.  Setting determinism while
    // other threads are mid-dispatch is therefore racy by design and
    // not recommended: configure the flag at process startup, before
    // launching training threads.
    static void set_enabled(bool value);
};

}  // namespace lucid
