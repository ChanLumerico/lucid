// lucid/_C/core/SchemaGuard.h
//
// Op-dispatch entry gate that combines determinism enforcement with
// AMP dtype resolution.
//
// Every Lucid op (forward and backward) begins its dispatch sequence by
// constructing a :class:`SchemaGuard` from its :class:`OpSchema` and the
// dtype/device of its primary input.  The guard's constructor does two
// things in one pass:
//
//   1. Determinism gate — if the global :func:`Determinism::is_enabled`
//      flag is set and the schema's ``deterministic`` field is false, a
//      :class:`LucidError` is raised before any computation begins.  The
//      error message includes the op's ``determinism_note`` so users
//      know which aspect of the op is non-reproducible.
//
//   2. Effective dtype resolution — if AMP is active on the current
//      thread (:func:`amp::is_active` returns true), the input dtype is
//      transformed according to the schema's :enum:`AmpPolicy`:
//        Promote   — switch to the autocast dtype, except that
//                    ``(CPU, F16)`` is clamped to F32 because Apple
//                    Accelerate has no native F16 BLAS path.
//        KeepInput — return ``input_dtype`` unchanged.
//        ForceFP32 — always return :class:`Dtype::F32`.
//      When AMP is not active the guard short-circuits and
//      :func:`effective_dtype` simply returns ``input_dtype``.
//
// :func:`check_schema_determinism` exposes the determinism gate as a
// standalone function for callers (e.g. custom-function wrappers,
// torch.compile-style trace builders) that need only the determinism
// check without dtype resolution.
//
// See Also
// --------
// :class:`OpSchema`  — static metadata that drives the guard.
// :enum:`AmpPolicy`  — autocast dispatch policy values.

#pragma once

#include "../api.h"
#include "AmpPolicy.h"
#include "Determinism.h"
#include "Device.h"
#include "Dtype.h"
#include "OpSchema.h"
#include "fwd.h"

namespace lucid {

// Determinism-only gate for a schema.
//
// Equivalent to the first half of :class:`SchemaGuard`'s constructor.
// Exists so callers that do not need AMP dtype resolution can perform
// just the determinism check without paying for the (small) extra work.
//
// Parameters
// ----------
// schema : const OpSchema&
//     The op's schema record.  Only the ``deterministic``,
//     ``determinism_note``, and ``name`` fields are read.
//
// Raises
// ------
// LucidError
//     Thrown when :func:`Determinism::is_enabled` is true and
//     ``schema.deterministic`` is false.  The message starts with
//     ``"non-deterministic op called under set_deterministic(True)"``
//     and appends ``schema.determinism_note`` in parentheses when
//     non-empty.
//
// Notes
// -----
// When determinism is not globally enabled, this function is a no-op
// and adds only a single relaxed atomic load to the call site.
LUCID_API void check_schema_determinism(const OpSchema& schema);

// Combined determinism + AMP dtype-resolution guard.
//
// Constructed once at the top of an op's dispatch routine.  The
// constructor performs both checks; subsequent code reads the resolved
// dtype via :func:`effective_dtype`.
//
// Despite the "Guard" suffix this is not an RAII resource holder — it
// owns no locks, allocations, or thread-local state.  The name reflects
// its role as a gate at the entry of the op, not a scope object.
//
// Examples
// --------
// .. code-block:: cpp
//
//     // Inside a forward op
//     SchemaGuard guard(MyOp::schema_v1, input->dtype(), input->device());
//     const Dtype dt = guard.effective_dtype();
//     auto out = backend->allocate(out_shape, dt);
//     dispatch_kernel(input, out, dt);
//
// See Also
// --------
// :func:`check_schema_determinism` — determinism-only variant.
class LUCID_API SchemaGuard {
public:
    // Constructs the guard and runs both gates.
    //
    // Parameters
    // ----------
    // schema : const OpSchema&
    //     Static metadata for the op being dispatched.
    // input_dtype : Dtype
    //     The dtype of the op's primary input — the starting point for
    //     AMP dtype resolution.
    // device : Device, default=Device::CPU
    //     The device on which the op will execute.  Used to apply the
    //     CPU-F16 → F32 demotion rule under ``AmpPolicy::Promote``,
    //     since Accelerate has no native F16 path.
    //
    // Raises
    // ------
    // LucidError
    //     Thrown when the op is non-deterministic and the global
    //     deterministic mode is active.  See
    //     :func:`check_schema_determinism` for the exact message
    //     format.
    SchemaGuard(const OpSchema& schema, Dtype input_dtype, Device device = Device::CPU);

    // Returns the dtype that the op should use for its computation.
    //
    // May differ from the ``input_dtype`` passed at construction when
    // AMP is active and the schema's :enum:`AmpPolicy` is ``Promote``
    // (returns the autocast dtype, possibly demoted on CPU) or
    // ``ForceFP32`` (returns :class:`Dtype::F32`).  Equals
    // ``input_dtype`` for ``KeepInput`` and when AMP is not active.
    //
    // Returns
    // -------
    // Dtype
    //     The effective compute dtype for the op.
    //
    // Notes
    // -----
    // Marked ``noexcept`` — purely a field read, no allocations or
    // exceptions.
    Dtype effective_dtype() const noexcept { return effective_dtype_; }

private:
    Dtype effective_dtype_;
};

}  // namespace lucid
