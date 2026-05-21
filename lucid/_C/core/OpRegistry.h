// lucid/_C/core/OpRegistry.h
//
// Global registry that maps op names to their :class:`OpSchema` records.
//
// Every op in Lucid (linear, conv2d, matmul, softmax, ...) publishes a
// constexpr :class:`OpSchema` named ``schema_v1`` on its backward class and
// registers it here at static-initialisation time via the
// :c:macro:`LUCID_REGISTER_OP` helper.  Downstream consumers — the
// :class:`SchemaGuard` dispatch gate, the profiler, diagnostic tooling, and
// the Python introspection bindings — look up schemas by name at runtime.
//
// Thread safety
// -------------
// The registry is backed by a :cpp:class:`std::shared_mutex` (readers-writer
// lock):
//
// * :func:`register_op` acquires a unique (exclusive) write lock.
// * :func:`lookup`, :func:`all`, :func:`size` acquire a shared (read) lock.
//
// Concurrent lookups never block each other; the hot dispatch path therefore
// remains contention-free under realistic workloads where registrations
// happen only at static-init time and reads happen on every op call.
//
// Static-initialisation order
// ---------------------------
// The map is held in a function-local static (see ``OpRegistry.cpp``'s
// ``storage()``) so first-use construction is guaranteed before any
// registration runs.  This sidesteps the static-init-order fiasco for
// :c:macro:`LUCID_REGISTER_OP` invocations in arbitrary translation units.

#pragma once

#include <string_view>
#include <vector>

#include "../api.h"
#include "OpSchema.h"

namespace lucid {

// Thread-safe global op registry.
//
// All methods are static; the underlying map and its lock live in a
// function-local static inside :file:`OpRegistry.cpp`.  Schemas are stored
// by non-owning pointer — the caller (typically a constexpr static in an
// op's translation unit) must guarantee that the pointee outlives the
// registry.
//
// Notes
// -----
// The map keys are :cpp:class:`std::string_view` references into each
// schema's ``name`` field.  Because schemas are required to live for the
// process lifetime, the keys are stable for the same duration — no string
// copying or hashing of string contents per lookup is needed.
class LUCID_API OpRegistry {
public:
    // Registers a schema under ``schema.name``.
    //
    // Acquires an exclusive write lock.  If an op with the same name was
    // already registered the entry is silently overwritten — this is
    // intentional, to let tests inject mocked schemas without going
    // through a deregister API.  Production code registers each schema
    // exactly once at static-init time.
    //
    // Parameters
    // ----------
    // schema : const OpSchema&
    //     A schema whose lifetime exceeds the registry's (constexpr
    //     static is the canonical case).  Only the pointer is stored.
    //
    // Notes
    // -----
    // ``schema.name`` is captured as a :cpp:class:`std::string_view`; do
    // not pass schemas whose ``name`` is backed by a temporary
    // ``std::string``.
    static void register_op(const OpSchema& schema);

    // Looks up a registered schema by canonical op name.
    //
    // Acquires a shared read lock.  Multiple threads may call concurrently.
    //
    // Parameters
    // ----------
    // name : std::string_view
    //     Canonical op name as registered (case-sensitive).
    //
    // Returns
    // -------
    // const OpSchema*
    //     Pointer to the registered schema, or ``nullptr`` when no op of
    //     that name exists.  The returned pointer is stable for the
    //     lifetime of the process.
    static const OpSchema* lookup(std::string_view name);

    // Returns a snapshot of every registered schema.
    //
    // Acquires a shared read lock.  The snapshot is a freshly allocated
    // vector; the schemas it points to remain valid for the process
    // lifetime.
    //
    // Returns
    // -------
    // std::vector<const OpSchema*>
    //     All currently registered schemas, ordered alphabetically by
    //     ``name`` (the backing container is a :cpp:class:`std::map`).
    //
    // Notes
    // -----
    // Intended for diagnostics, profiler dumps, and Python introspection
    // — not for the hot dispatch path, which should use :func:`lookup`.
    static std::vector<const OpSchema*> all();

    // Returns the number of registered ops.
    //
    // Acquires a shared read lock.
    //
    // Returns
    // -------
    // std::size_t
    //     Current size of the registry.
    static std::size_t size();
};

}  // namespace lucid

// Static-initialisation registration helper for op schemas.
//
// Expands to an anonymous-namespace struct whose default constructor
// calls :func:`OpRegistry::register_op` with ``NAME::schema_v1``, plus a
// file-scope static instance of that struct.  The instance's constructor
// runs during shared-library load (before ``main()``), so by the time any
// op dispatch happens the registry already contains every op linked into
// the binary.
//
// Parameters
// ----------
// NAME : identifier
//     The op's backward class.  Must expose a ``static const OpSchema
//     schema_v1`` (or ``static constexpr OpSchema schema_v1``) accessible
//     in the namespace where the macro is expanded.
//
// Examples
// --------
// .. code-block:: cpp
//
//     // In MyOp.cpp
//     const OpSchema MyOpBackward::schema_v1{"my_op", 1, AmpPolicy::Promote};
//     LUCID_REGISTER_OP(MyOpBackward);
//
// Notes
// -----
// Wrapping the helper struct in an anonymous namespace gives the static
// instance internal linkage, so each translation unit that includes
// :file:`OpRegistry.h` and uses the macro gets its own private
// registrar — no ODR collision between translation units that register
// different ops with the same NAME identifier (though sharing op names
// is still illegal).
#define LUCID_REGISTER_OP(NAME)                                                                    \
    namespace {                                                                                    \
    struct LucidRegister_##NAME {                                                                  \
        LucidRegister_##NAME() { ::lucid::OpRegistry::register_op(NAME::schema_v1); }              \
    };                                                                                             \
    static const LucidRegister_##NAME _lucid_register_##NAME{};                                    \
    }
