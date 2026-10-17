// lucid/_C/core/AmpPolicy.h
//
// Automatic Mixed Precision (AMP) policy enum, the per-device policy table,
// and thread-local autocast state.
//
// AMP lets a model run most ops in a lower-precision dtype (typically BF16 or
// F16) for memory- and bandwidth-bound speedups, while keeping a small set of
// numerically sensitive ops at F32.  Lucid implements this with three pieces:
//
//   1. ``AmpPolicy`` — what an op does with its floating inputs inside an
//      autocast scope.
//   2. The policy table (``AmpPolicyTable.def``) — one row per op that has
//      a policy, with a column per device.  :func:`amp::policy_for` reads it;
//      an op with no row keeps its inputs' dtype.
//   3. ``amp::AutocastGuard`` — an RAII guard that activates AMP on the
//      current thread for the lifetime of the guard.  The Python
//      ``lucid.amp.autocast`` context manager (see ``lucid/amp/autocast.py``)
//      drives this guard from user code.
//
// :class:`SchemaGuard` applies the table at an engine primitive's entry and
// ``AmpScope`` (``ops/utils/Promote.h``) at an engine composite's.  Each
// turns autocast off on the op's device for as long as it lives, so the ops
// built inside that span compute in the dtype the entry chose instead of
// applying their own rows a second time.  The kernel templates hold their
// guard for the whole forward; ``promote_for_schema`` drops its guard on
// return, so an op that casts through it runs its body with autocast on
// (LCD-500-E2).
//
// References
// ----------
// Micikevicius et al., "Mixed Precision Training" (ICLR 2018) — describes
// the FP16 + FP32 master-weight pattern that motivates the Promote /
// ForceFP32 split.

#pragma once

#include <optional>
#include <span>
#include <string_view>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"

namespace lucid {

// What an op does with its floating inputs inside an autocast scope.
//
// Only float16, bfloat16 and float32 inputs are ever cast; float64, complex,
// integer and bool inputs keep their dtype under every policy.
//
// Attributes
// ----------
// Promote : AmpPolicy
//     Cast inputs to the currently active autocast dtype (the table's
//     ``LOWER``).  Used for compute-bound ops where reduced precision
//     yields a large speedup with negligible accuracy impact — linear,
//     conv, matmul.  On the CPU stream, F16 is computed in F32 because
//     Apple Accelerate does not implement native F16 BLAS kernels.
// KeepInput : AmpPolicy
//     Run in the inputs' own dtype (the table's ``KEEP``, and every op
//     without a row).
// ForceFP32 : AmpPolicy
//     Cast inputs to F32 (the table's ``FP32``).  Used for ops with severe
//     numerical sensitivity — softmax, log, exponentials, loss reductions,
//     normalisation statistics — where BF16/F16 would cause catastrophic
//     cancellation or overflow.
// PromoteWidest : AmpPolicy
//     Cast inputs to the widest floating dtype among them (the table's
//     ``WIDEST``), for ops that take several floating operands and have
//     no kernel for a mixed pair.  For a single input it is ``KeepInput``.
//
// Notes
// -----
// The enum has fixed underlying type ``std::uint8_t`` so it packs tightly
// into :class:`OpSchema` and into the FNV-1a hash computed by
// :func:`schema_hash`.  New values are appended, never inserted.
enum class AmpPolicy : std::uint8_t {
    Promote,        // Use autocast dtype (CPU F16 → F32).
    KeepInput,      // Ignore autocast; always use the input tensor's dtype.
    ForceFP32,      // Always use F32, regardless of autocast setting.
    PromoteWidest,  // Use the widest floating dtype among the inputs.
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
//     A pointer to a static C string — ``"Promote"``, ``"KeepInput"``,
//     ``"ForceFP32"`` or ``"PromoteWidest"``.  The pointer is stable for
//     the lifetime of the process.
//
// Raises
// ------
// LucidError
//     Thrown via :class:`ErrorBuilder` when ``p`` is not a known
//     enumerator (defensive guard against bit-rotted casts).
LUCID_API const char* amp_policy_name(AmpPolicy p);

namespace amp {

// One row of the policy table: an op and its policy on each device.
struct PolicyRow {
    std::string_view name;
    AmpPolicy metal;
    AmpPolicy cpu;
};

// The policy ``op`` takes on ``device`` under autocast.
//
// ``op`` is a schema name or an engine composite's name.  An op without a
// row in ``AmpPolicyTable.def`` keeps its inputs' dtype, so the answer for
// it is ``AmpPolicy::KeepInput`` — whatever its schema declares.
LUCID_API AmpPolicy policy_for(std::string_view op, Device device);

// Every row of the policy table, sorted by name.
LUCID_API std::span<const PolicyRow> policy_table();

// Whether autocast may recast a tensor of this dtype.  float64, complex,
// integer and bool tensors keep their dtype under every policy.
constexpr bool is_autocast_eligible(Dtype dt) noexcept {
    return dt == Dtype::F16 || dt == Dtype::BF16 || dt == Dtype::F32;
}

// The dtype one eligible input computes in under ``policy`` while
// ``autocast_dt`` is active on ``device``.
//
// ``PromoteWidest`` needs the other inputs to mean anything, so for one
// input it is ``KeepInput``; ``AmpScope::cast_widest`` resolves it over all
// of them.  ``Promote`` with float16 on the CPU computes in float32, because
// Accelerate has no float16 arithmetic.
constexpr Dtype
policy_dtype(AmpPolicy policy, Dtype input, Dtype autocast_dt, Device device) noexcept {
    switch (policy) {
    case AmpPolicy::Promote:
        return (device == Device::CPU && autocast_dt == Dtype::F16) ? Dtype::F32 : autocast_dt;
    case AmpPolicy::ForceFP32:
        return Dtype::F32;
    case AmpPolicy::KeepInput:
    case AmpPolicy::PromoteWidest:
        return input;
    }
    return input;
}

// Returns the active autocast dtype for ``device`` on the calling thread.
//
// Reads the thread-local per-device AMP state established by the innermost
// live :class:`AutocastGuard` for ``device``.  Returns ``std::nullopt`` when
// no guard is active for ``device``.
LUCID_API std::optional<Dtype> active_dtype(Device device);

// Returns the active autocast dtype on the calling thread (GPU if active,
// otherwise CPU if active, otherwise ``std::nullopt``).
//
// Notes
// -----
// All AMP state is ``thread_local``; different Python threads (or OpenMP
// workers) can each maintain independent autocast contexts without
// synchronisation.
LUCID_API std::optional<Dtype> active_dtype();

// Returns true when AMP is active for ``device`` on the calling thread.
LUCID_API bool is_active(Device device);

// Returns true when AMP is active on any device on the calling thread.
LUCID_API bool is_active();

// RAII guard that activates (or disables) AMP for a device (or all active
// devices) for the duration of the guard's lifetime.
class LUCID_API AutocastGuard {
public:
    // Sets the autocast target dtype to ``target`` on whichever device(s)
    // currently have AMP active (or both devices if neither is active).
    explicit AutocastGuard(Dtype target);

    // Sets the autocast state for ``device`` to ``(active=enabled, dtype=target)``
    // and restores the previous state for ``device`` on destruction.
    AutocastGuard(Dtype target, Device device, bool enabled = true);

    // Restores the previous AMP state captured at construction.
    //
    // Runs unconditionally — required for correct nesting semantics.
    ~AutocastGuard();

    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;

private:
    std::optional<Device> device_;
    bool prev_cpu_active_ = false;
    Dtype prev_cpu_dtype_ = Dtype::F16;
    bool prev_gpu_active_ = false;
    Dtype prev_gpu_dtype_ = Dtype::F16;
};

}  // namespace amp
}  // namespace lucid
