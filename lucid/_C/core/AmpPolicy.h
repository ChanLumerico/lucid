// lucid/_C/core/AmpPolicy.h
//
// Automatic Mixed Precision (AMP) policy enum and thread-local autocast state.
//
// AMP lets a model run most ops in a lower-precision dtype (typically BF16 or
// F16) for memory- and bandwidth-bound speedups, while keeping a small set of
// numerically sensitive ops at F32.  Lucid implements this with two pieces:
//
//   1. ``AmpPolicy`` — a static per-op enum baked into each op's
//      :class:`OpSchema`.  Set once by the op author, never changed at
//      runtime.  Tells the dispatcher whether to cast, ignore the autocast
//      request, or force F32.
//   2. ``amp::AutocastGuard`` — an RAII guard that activates AMP on the
//      current thread for the lifetime of the guard.  The Python
//      ``lucid.amp.autocast`` context manager (see ``lucid/amp/autocast.py``)
//      drives this guard from user code.
//
// :class:`SchemaGuard` combines both at op-dispatch time: it reads
// ``amp::active_dtype()`` and the op's ``AmpPolicy`` to compute the effective
// compute dtype.
//
// References
// ----------
// Micikevicius et al., "Mixed Precision Training" (ICLR 2018) — describes
// the FP16 + FP32 master-weight pattern that motivates the Promote /
// ForceFP32 split.

#pragma once

#include <optional>

#include "../api.h"
#include "Dtype.h"

namespace lucid {

// Mixed-precision dispatch policy attached to each registered op.
//
// Each backward class declares its policy in its ``schema_v1`` constant
// (e.g. ``AmpPolicy::Promote`` for matmul, ``AmpPolicy::ForceFP32`` for
// softmax).  The value never changes for the lifetime of the process —
// it is part of the op's static contract, not a runtime knob.
//
// Attributes
// ----------
// Promote : AmpPolicy
//     Cast inputs to the currently active autocast dtype before forward.
//     Used for compute-bound ops where reduced precision yields a large
//     speedup with negligible accuracy impact — linear, conv, matmul.
//     On the CPU stream, F16 is silently demoted to F32 because Apple
//     Accelerate does not implement native F16 BLAS kernels.
// KeepInput : AmpPolicy
//     Run in the input tensor's native dtype regardless of the autocast
//     setting.  Used for ops that either do not benefit from reduced
//     precision (memory-bound element-wise ops), are dtype-polymorphic
//     (cast, copy), or maintain accumulators that must not be downcast
//     (batch-norm running statistics).
// ForceFP32 : AmpPolicy
//     Always cast to F32 inside the op, regardless of the active autocast
//     dtype.  Used for ops with severe numerical sensitivity — softmax,
//     log, exponentials, loss reductions, normalisation statistics — where
//     BF16/F16 would cause catastrophic cancellation or overflow.
//
// Notes
// -----
// The enum has fixed underlying type ``std::uint8_t`` so it packs tightly
// into :class:`OpSchema` and into the FNV-1a hash computed by
// :func:`schema_hash`.
//
// References
// ----------
// Micikevicius et al., "Mixed Precision Training" (ICLR 2018).
enum class AmpPolicy : std::uint8_t {
    Promote,    // Use autocast dtype (CPU F16 → F32).
    KeepInput,  // Ignore autocast; always use the input tensor's dtype.
    ForceFP32,  // Always use F32, regardless of autocast setting.
};

// Returns the human-readable name of an AMP policy.
//
// Used by diagnostics, error messages, and the profiler to render the
// policy attached to each dispatched op.
//
// Parameters
// ----------
// p : AmpPolicy
//     The policy value to stringify.
//
// Returns
// -------
// const char*
//     A pointer to a static C string — ``"Promote"``, ``"KeepInput"`` or
//     ``"ForceFP32"``.  The pointer is stable for the lifetime of the
//     process.
//
// Raises
// ------
// LucidError
//     Thrown via :class:`ErrorBuilder` when ``p`` is not a known
//     enumerator (defensive guard against bit-rotted casts).
LUCID_API const char* amp_policy_name(AmpPolicy p);

namespace amp {

// Returns the active autocast dtype on the calling thread.
//
// Reads the thread-local AMP state established by the innermost live
// :class:`AutocastGuard`.  Returns ``std::nullopt`` when no guard is
// active — callers can use this to skip the AMP dispatch path entirely.
//
// Returns
// -------
// std::optional<Dtype>
//     The autocast target dtype if AMP is active on this thread, or
//     ``std::nullopt`` otherwise.
//
// Notes
// -----
// All AMP state is ``thread_local``; different Python threads (or OpenMP
// workers) can each maintain independent autocast contexts without
// synchronisation.
LUCID_API std::optional<Dtype> active_dtype();

// Returns true when AMP is active on the calling thread.
//
// Equivalent to ``active_dtype().has_value()`` but avoids constructing
// an ``std::optional`` on the hot dispatch path.
//
// Returns
// -------
// bool
//     ``true`` if at least one :class:`AutocastGuard` is currently live
//     on this thread, ``false`` otherwise.
LUCID_API bool is_active();

// RAII guard that activates AMP with a target dtype for the duration of
// the guard's lifetime.
//
// Construction sets the thread-local autocast state to ``(active=true,
// dtype=target)`` and saves the previous values.  Destruction restores
// the previous values unconditionally, so guards may be nested freely
// (an inner ``ForceFP32`` scope inside an outer ``F16`` scope, etc.).
//
// The Python-side ``lucid.amp.autocast`` context manager wraps this guard
// and is the user-facing entry point — direct C++ use is rare outside
// the engine itself.
//
// Notes
// -----
// Copy and move are deleted: this is a stack-only RAII object, and
// duplicating it would corrupt the saved/restore state machine.
//
// The Python binding exposes ``__enter__`` / ``__exit__`` semantics
// without restoring on exit (the Python ``autocast`` class implements
// proper RAII in Python); the destructor still runs the restore when
// the C++ object is finally collected.
//
// Examples
// --------
// .. code-block:: cpp
//
//     {
//         amp::AutocastGuard guard(Dtype::F16);
//         // ops inside this scope dispatch under AMP
//         auto y = matmul(a, b);
//     } // guard destructor restores previous AMP state
class LUCID_API AutocastGuard {
public:
    // Activates AMP with ``target`` as the effective compute dtype.
    //
    // Saves the previous ``(active, dtype)`` pair so the destructor can
    // restore it.  Does not validate ``target`` — any :class:`Dtype` is
    // accepted (the per-op :class:`AmpPolicy` controls how that dtype is
    // interpreted at dispatch time).
    //
    // Parameters
    // ----------
    // target : Dtype
    //     The autocast target dtype reported by :func:`amp::active_dtype`
    //     while this guard is live.  Typical values are :class:`Dtype::F16`
    //     for GPU half-precision training and :class:`Dtype::BF16` for
    //     bfloat16 training; :class:`Dtype::F32` produces an effectively
    //     neutral guard.
    explicit AutocastGuard(Dtype target);

    // Restores the previous AMP state captured at construction.
    //
    // Runs unconditionally — required for correct nesting semantics.
    ~AutocastGuard();

    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;

private:
    bool prev_active_;
    Dtype prev_dtype_;
};

}  // namespace amp
}  // namespace lucid
