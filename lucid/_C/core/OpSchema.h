// lucid/_C/core/OpSchema.h
//
// Descriptor that encodes the static properties of a registered op.  Each op
// implementation defines a constexpr OpSchema instance (conventionally named
// schema_v1) and registers it with OpRegistry via the LUCID_REGISTER_OP macro
// or a direct call to OpRegistry::register_op().
//
// SchemaGuard (SchemaGuard.h) reads the schema at op-dispatch time to:
//   1. Enforce determinism constraints if Determinism::is_enabled().
//   2. Compute the effective compute dtype under AMP (driven by amp_policy).
//
// The schema_hash() function produces a compact fingerprint of the schema for
// change-detection (e.g. cache invalidation, serialisation versioning).

#pragma once

#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <vector>

#include "../api.h"
#include "AmpPolicy.h"

namespace lucid {

// Static descriptor for an op, stored by pointer in the OpRegistry.
//
// Ownership: OpSchema objects are typically constexpr statics inside op
// implementation files; the registry stores non-owning pointers.  Schemas
// must outlive the registry (process lifetime for constexpr statics).
//
// Fields:
//   name             — canonical op name used as the registry key and in
//                      error messages.
//   version          — monotonic schema version; bump when the signature or
//                      semantics change in a backwards-incompatible way.
//   amp_policy       — controls dtype promotion under AMP (see AmpPolicy.h).
//   deterministic    — false if the op may produce non-reproducible results
//                      (e.g. atomics-based reductions).
//   determinism_note — human-readable explanation of why the op is
//                      non-deterministic; included in error messages.
//   input_arity      — expected number of input tensors; -1 means variadic.
//   output_arity     — number of output tensors produced (default 1).
//   stable_input_indices — indices of inputs that must not be mutated in-place
//                      while their saved tensor versions are held by autograd.
//   internal         — true for ops that should not be callable directly from
//                      Python (implementation details, fused kernels, etc.).
struct LUCID_API OpSchema {
    std::string_view name;
    int version;

    AmpPolicy amp_policy;
    bool deterministic;
    std::string_view determinism_note;

    // -1 indicates the op accepts a variable number of inputs.
    int input_arity = -1;

    int output_arity = 1;

    // Indices into the input list whose values are saved for backward;
    // SchemaGuard / autograd version-check logic uses these to identify
    // which inputs must remain unmodified between forward and backward.
    std::vector<int> stable_input_indices;

    // When true, the op should not be exposed in the public Python API.
    bool internal = false;

    // Minimal constructor for deterministic ops with default arity.
    constexpr OpSchema(
        std::string_view n, int v, AmpPolicy ap, bool det = true, std::string_view note = "")
        : name(n), version(v), amp_policy(ap), deterministic(det), determinism_note(note) {}

    // Full constructor for ops that need explicit arity or internal flag.
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

// Computes a 64-bit FNV-1a hash of the schema's name, version, amp_policy, and
// deterministic flag.  Used for schema change-detection; does not hash the
// stable_input_indices or internal flag since they are not part of the
// behavioural signature.
LUCID_API std::uint64_t schema_hash(const OpSchema& s);

}  // namespace lucid
