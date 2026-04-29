#pragma once

// =====================================================================
// Lucid C++ engine — SchemaGuard: Phase 5 schema enforcement.
// =====================================================================
//
// RAII object constructed at the entry of every kernel forward().
// Two responsibilities:
//
//   1. Determinism gate: if Determinism::is_enabled() and the op's
//      schema declares deterministic=false, throw immediately.
//
//   2. AMP dtype resolution: reads the active AutocastGuard dtype
//      (if any) and the op's AmpPolicy to compute the effective
//      computation dtype.  Callers use `effective_dtype()` instead
//      of the raw input dtype, and `maybe_cast()` to transparently
//      promote inputs before they enter the kernel.
//
// AMP policy semantics:
//   Promote   → use autocast dtype (e.g. F16) when autocast is active.
//   KeepInput → always use the input's native dtype, never cast.
//   ForceFP32 → always F32, even when autocast requests F16.
//
// The cast inserted by `maybe_cast()` is NOT tracked by autograd:
// AMP casts are transparent — the backward sees the original dtype.
//
// Layer: core/. Depends on AmpPolicy.h, Determinism.h, OpSchema.h.

#include "../api.h"
#include "AmpPolicy.h"
#include "Determinism.h"
#include "Device.h"
#include "Dtype.h"
#include "OpSchema.h"
#include "fwd.h"

namespace lucid {

/// Lightweight determinism-only check for ops that manage their own dtype
/// (e.g. NaryKernel ops like dropout that call their own forward()).
/// Throws if Determinism::is_enabled() and schema.deterministic == false.
LUCID_API void check_schema_determinism(const OpSchema& schema);

/// RAII guard: resolves effective dtype (AMP) and gates deterministic ops.
class LUCID_API SchemaGuard {
public:
    /// Construct and immediately enforce schema invariants.
    ///   `schema`       — the op's static schema_v1.
    ///   `input_dtype`  — dtype of the primary input tensor (used for
    ///                    KeepInput and as the baseline for Promote).
    ///   `device`       — device of the primary input (CPU cannot do F16;
    ///                    Promote on CPU with F16 autocast falls back to F32).
    SchemaGuard(const OpSchema& schema, Dtype input_dtype, Device device = Device::CPU);

    /// The dtype that the kernel should use for compute.
    /// Callers should cast inputs with maybe_cast() before reading.
    Dtype effective_dtype() const noexcept { return effective_dtype_; }

    /// Return `t` unchanged if its dtype already equals effective_dtype(),
    /// otherwise return a new TensorImpl with storage cast to effective_dtype().
    /// The cast is performed via IBackend::cast — zero autograd overhead.
    TensorImplPtr maybe_cast(const TensorImplPtr& t) const;

private:
    Dtype effective_dtype_;
};

}  // namespace lucid
