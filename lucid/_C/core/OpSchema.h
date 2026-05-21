// lucid/_C/core/OpSchema.h
//
// Static metadata descriptor for a registered op.
//
// Every op in Lucid is described by a constexpr :class:`OpSchema` instance —
// conventionally a public static member named ``schema_v1`` on the op's
// backward class.  The schema records the op's name, version, AMP policy,
// determinism contract, arity, and a list of "stable" inputs that autograd
// must guard against in-place mutation.
//
// The schema is registered with the global :class:`OpRegistry` at static
// initialisation time (see :file:`OpRegistry.h`'s ``LUCID_REGISTER_OP``
// macro) and consulted by :class:`SchemaGuard` at every op dispatch to:
//
//   1. Reject calls that violate the current determinism setting.
//   2. Compute the effective compute dtype under AMP.
//
// Versioning contract
// -------------------
// ``schema_v1`` is the version-1 schema; bumping to ``schema_v2`` is a wire-
// format break — saved checkpoints and traced graphs that reference the
// older version must be migrated.  Bump the ``version`` field whenever the
// op's signature, semantics, or AMP/determinism contract changes in a way
// that an old caller could not handle.
//
// See Also
// --------
// :class:`OpRegistry`     — the global name → schema map.
// :class:`SchemaGuard`    — runtime gate that consumes the schema.
// :enum:`AmpPolicy`       — per-op autocast policy stored in ``amp_policy``.

#pragma once

#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <vector>

#include "../api.h"
#include "AmpPolicy.h"

namespace lucid {

// Per-op metadata record stored by pointer in the :class:`OpRegistry`.
//
// Schemas are intended to be ``constexpr`` statics defined alongside each
// op implementation; the registry stores raw pointers and assumes the
// pointee outlives the registry (true for process-lifetime statics).
// Constructing schemas with dynamic storage duration is supported but
// discouraged.
//
// Attributes
// ----------
// name : std::string_view
//     Canonical op name; used as the :class:`OpRegistry` key and in
//     diagnostic / error messages.  Must point into storage that lives at
//     least as long as the registry (constexpr literals are ideal).
// version : int
//     Monotonic schema version starting at 1.  Bump on backwards-
//     incompatible changes — bumping invalidates checkpoints / traces that
//     reference the previous version.
// amp_policy : AmpPolicy
//     How this op responds to an active autocast context.  See
//     :enum:`AmpPolicy` for the three values.
// deterministic : bool
//     ``true`` if the op produces bit-identical results given identical
//     inputs and configuration.  ``false`` for ops that rely on atomics,
//     non-associative parallel reductions, or other sources of run-to-run
//     variation.  Consulted by :func:`check_schema_determinism` /
//     :class:`SchemaGuard` when ``Determinism::is_enabled()``.
// determinism_note : std::string_view
//     Human-readable explanation of why the op is non-deterministic
//     (e.g. ``"uses atomic scatter-add"``).  Included verbatim in the
//     error message raised under ``set_deterministic(True)``.  Empty for
//     deterministic ops.
// input_arity : int
//     Expected number of input tensors.  ``-1`` means variadic (the op
//     accepts any number of inputs; the op is responsible for its own
//     arity validation).
// output_arity : int
//     Number of output tensors produced.  Defaults to ``1``; most ops are
//     single-output.
// stable_input_indices : std::vector<int>
//     Indices into the input list whose values are saved for backward.
//     Autograd's version-counter logic uses these to detect illegal
//     in-place mutations between forward and backward that would corrupt
//     the saved tensors.
// internal : bool
//     ``true`` for ops that are not part of the public Python API
//     (implementation-detail kernels, fused / internal helpers).  The
//     Python binding layer hides these from ``dir(lucid)`` etc.
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
    //
    // Parameters
    // ----------
    // n : std::string_view
    //     Canonical op name.
    // v : int
    //     Schema version (start at 1).
    // ap : AmpPolicy
    //     AMP dispatch policy.
    // det : bool, default=true
    //     Determinism flag.
    // note : std::string_view, default=""
    //     Non-determinism explanation; ignored when ``det`` is true.
    //
    // Notes
    // -----
    // Leaves ``input_arity = -1`` (variadic), ``output_arity = 1``,
    // ``stable_input_indices`` empty, and ``internal = false``.  Use the
    // full constructor below when these defaults are not appropriate.
    constexpr OpSchema(
        std::string_view n, int v, AmpPolicy ap, bool det = true, std::string_view note = "")
        : name(n), version(v), amp_policy(ap), deterministic(det), determinism_note(note) {}

    // Full constructor for ops that need explicit arity, saved-input
    // tracking, or the ``internal`` flag set.
    //
    // Parameters
    // ----------
    // n : std::string_view
    //     Canonical op name.
    // v : int
    //     Schema version.
    // ap : AmpPolicy
    //     AMP dispatch policy.
    // det : bool
    //     Determinism flag.
    // note : std::string_view
    //     Non-determinism explanation.
    // in_arity : int
    //     Expected input count, or ``-1`` for variadic.
    // out_arity : int
    //     Number of outputs produced.
    // stable_ins : std::vector<int>
    //     Indices of inputs whose values are saved for backward.
    // is_internal : bool, default=false
    //     Hide the op from the public Python surface when true.
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

// Computes a 64-bit FNV-1a fingerprint of an op's behavioural schema.
//
// The hash mixes the schema's ``name``, ``version``, ``amp_policy``, and
// ``deterministic`` flag — fields that together define how the op
// behaves.  ``stable_input_indices``, ``internal``, and arity counts are
// excluded because they describe shape rather than semantics.
//
// Used for change-detection: cached graph compilations, profiler entries,
// and checkpoint metadata embed this hash so a schema bump (or any
// behaviour-relevant change) invalidates the cache automatically.
//
// Parameters
// ----------
// s : const OpSchema&
//     The schema to hash.
//
// Returns
// -------
// std::uint64_t
//     A 64-bit FNV-1a digest.  Stable across processes for the same
//     schema fields; not cryptographic.
//
// Math
// ----
// $$ h_0 = \text{0xcbf29ce484222325}, \qquad
//    h_{i+1} = (h_i \oplus b_i) \cdot \text{0x100000001b3} \pmod{2^{64}}, $$
//
// where $b_i$ ranges over the bytes of ``name``, a null separator,
// the little-endian bytes of ``version``, the ``amp_policy`` byte, and
// finally the ``deterministic`` flag (0 or 1).
//
// Notes
// -----
// The null byte between ``name`` and ``version`` prevents length-
// extension collisions of the form ``"ab"`` vs ``"a" + 'b'``.
LUCID_API std::uint64_t schema_hash(const OpSchema& s);

}  // namespace lucid
