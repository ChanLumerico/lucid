#pragma once

// =====================================================================
// Lucid C++ engine — global op registry.
// =====================================================================
//
// One process-wide map of `op_name → const OpSchema*`. Populated by static
// initializers in each op's translation unit via LUCID_REGISTER_OP.
//
// Use the construct-on-first-use pattern internally to avoid the static
// initialization order fiasco — registry can be queried before main() and
// from other static initializers.
//
// Layer: core/.

#include <string_view>
#include <vector>

#include "../api.h"
#include "OpSchema.h"

namespace lucid {

/// Process-wide thread-safe op registry: name → OpSchema*.
class LUCID_API OpRegistry {
public:
    static void register_op(const OpSchema& schema);
    static const OpSchema* lookup(std::string_view name);
    static std::vector<const OpSchema*> all();
    static std::size_t size();
};

}  // namespace lucid

/// Register an op's schema at module init. Place at file scope, after the op
/// class definition. Generates a unique-typed dummy struct whose constructor
/// performs the registration.
#define LUCID_REGISTER_OP(NAME)                                                       \
    namespace {                                                                       \
    struct LucidRegister_##NAME {                                                     \
        LucidRegister_##NAME() { ::lucid::OpRegistry::register_op(NAME::schema_v1); } \
    };                                                                                \
    static const LucidRegister_##NAME _lucid_register_##NAME{};                       \
    }
