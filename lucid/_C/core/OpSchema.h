#pragma once

// =====================================================================
// Lucid C++ engine — op schema (Phase 5.4: completeness fields).
// =====================================================================
//
// Every concrete op declares one `static const OpSchema schema_v1;` and
// registers it via LUCID_REGISTER_OP at module init. The registry is
// the source of truth for AMP, determinism, auto-bindings (Phase 6),
// checkpoint compatibility (Phase 5.5), and shape inference (Phase 7+).
//
// Versioning rule: bump `version` only on observable behavioral change.
//
// Layer: core/.

#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <vector>

#include "../api.h"
#include "AmpPolicy.h"

namespace lucid {

/// Immutable metadata record for one registered op (name, AMP policy, arity).
struct LUCID_API OpSchema {
    // ---- Identity -------------------------------------------------------
    std::string_view name;
    int version;

    // ---- AMP / determinism ----------------------------------------------
    AmpPolicy amp_policy;
    bool deterministic;
    std::string_view determinism_note;  // why nondeterministic, if applicable

    // ---- Arity (Phase 5.4) ----------------------------------------------
    /// Number of tensor inputs. -1 = variadic.
    int input_arity = -1;
    /// Number of tensor outputs. Almost always 1.
    int output_arity = 1;

    /// Indices of inputs that are saved for backward (kSavesInput/kSavesOutput
    /// on the CRTP kernel). Used by Phase 5.6 harness and Phase 7 audit.
    /// Convention: index -1 means "saves the output".
    std::vector<int> stable_input_indices;

    // ---- Visibility (Phase 6) -------------------------------------------
    /// true = this schema belongs to an internal backward node that is NOT
    /// user-facing (no Python binding). Examples: TriBackward ("tri"),
    /// ViewBackward ("view"), IndexScatterBackward ("index_scatter").
    /// Excluded from registry coverage checks and pyi stubs.
    bool internal = false;

    // ---- Constructors ---------------------------------------------------

    /// Minimal ctor (backward-compatible with all existing schema_v1 declarations).
    constexpr OpSchema(
        std::string_view n, int v, AmpPolicy ap, bool det = true, std::string_view note = "")
        : name(n), version(v), amp_policy(ap), deterministic(det), determinism_note(note) {}

    /// Full ctor with arity + internal flag (Phase 5.4 / Phase 6).
    OpSchema(std::string_view n,
             int v,
             AmpPolicy ap,
             bool det,
             std::string_view note,
             int in_arity,
             int out_arity,
             std::vector<int> stable_ins,
             bool is_internal = false)
        : name(n),
          version(v),
          amp_policy(ap),
          deterministic(det),
          determinism_note(note),
          input_arity(in_arity),
          output_arity(out_arity),
          stable_input_indices(std::move(stable_ins)),
          internal(is_internal) {}
};

/// Stable, deterministic hash of the schema. Used by Phase 5.5 to embed an
/// op signature in checkpoint metadata so loading can reject incompatible.
LUCID_API std::uint64_t schema_hash(const OpSchema& s);

}  // namespace lucid
