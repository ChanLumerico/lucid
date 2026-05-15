// lucid/_C/core/OpRegistry.h
//
// Global registry that maps op names to their OpSchema descriptors.
//
// Ops register themselves at static-initialisation time using the
// LUCID_REGISTER_OP convenience macro (or by calling register_op() directly).
// The SchemaGuard and diagnostic tools call lookup() at op-dispatch time.
//
// Thread safety: register_op() acquires a unique_lock (exclusive write) and
// lookup() / all() / size() acquire a shared_lock (concurrent reads allowed).
// std::shared_mutex provides readers-writer semantics so that the hot lookup
// path is not blocked by concurrent reads.
//
// LUCID_REGISTER_OP(NAME) macro:
//   Declares a local struct LucidRegister_##NAME whose default constructor
//   calls register_op(NAME::schema_v1).  A file-scope static instance of the
//   struct is then defined, causing the registration to occur during the
//   shared-library load (before main() runs).

#pragma once

#include <string_view>
#include <vector>

#include "../api.h"
#include "OpSchema.h"

namespace lucid {

// Thread-safe global op registry.  All methods are static; the registry
// itself is a function-local static inside OpRegistry.cpp (see storage()).
class LUCID_API OpRegistry {
public:
    // Registers schema under schema.name.  If an op with the same name was
    // already registered, the new schema silently overwrites the old one
    // (last-writer-wins; intended for test overrides).
    static void register_op(const OpSchema& schema);

    // Returns a pointer to the schema registered under name, or nullptr if no
    // op by that name exists.  The returned pointer is stable for the
    // lifetime of the process.
    static const OpSchema* lookup(std::string_view name);

    // Returns a snapshot of all registered schemas in registration order
    // (actually alphabetical by name because the backing container is a
    // std::map).  The returned pointers are stable for the process lifetime.
    static std::vector<const OpSchema*> all();

    // Returns the number of registered ops.
    static std::size_t size();
};

}  // namespace lucid

// Static-initialisation registration helper.
//
// Usage:  LUCID_REGISTER_OP(MyOp);
// Requirement: MyOp must have a static constexpr OpSchema member named
// schema_v1 that is accessible in the namespace where the macro is expanded.
#define LUCID_REGISTER_OP(NAME)                                                                    \
    namespace {                                                                                    \
    struct LucidRegister_##NAME {                                                                  \
        LucidRegister_##NAME() { ::lucid::OpRegistry::register_op(NAME::schema_v1); }              \
    };                                                                                             \
    static const LucidRegister_##NAME _lucid_register_##NAME{};                                    \
    }
