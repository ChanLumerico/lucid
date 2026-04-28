#pragma once

// =====================================================================
// Lucid C++ engine — op schema (P6: checkpoint compatibility contract).
// =====================================================================
//
// Every concrete op declares one `static const OpSchema schema_v1;` and
// registers it via LUCID_REGISTER_OP at module init. The registry becomes
// the source of truth for:
//   - `lucid.compile` op-IR validation (Phase 6)
//   - `Module.state_dict()` checkpoint metadata (Phase 5.5)
//   - `lucid.list_ops()` introspection (Phase 5)
//
// Versioning rule: bump `version` only on observable behavioral change
// (numerics, dtype matrix, attribute semantics). Internal refactors keep the
// same version. See docs/op_versioning.md (TODO Phase 5.5).
//
// Layer: core/.

#include <cstdint>
#include <string_view>

#include "../api.h"
#include "AmpPolicy.h"

namespace lucid {

struct LUCID_API OpSchema {
    std::string_view name;
    int version;
    AmpPolicy amp_policy;
    bool deterministic;
    std::string_view determinism_note;  // why nondeterministic, if applicable

    constexpr OpSchema(
        std::string_view n, int v, AmpPolicy ap, bool det = true, std::string_view note = "")
        : name(n), version(v), amp_policy(ap), deterministic(det), determinism_note(note) {}
};

/// Stable, deterministic hash of the schema. Used by Phase 5.5 to embed an
/// op signature in checkpoint metadata so loading can reject incompatible.
LUCID_API std::uint64_t schema_hash(const OpSchema& s);

}  // namespace lucid
