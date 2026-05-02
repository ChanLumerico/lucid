#pragma once

#include <string_view>
#include <vector>

#include "../api.h"
#include "OpSchema.h"

namespace lucid {

class LUCID_API OpRegistry {
public:
    static void register_op(const OpSchema& schema);
    static const OpSchema* lookup(std::string_view name);
    static std::vector<const OpSchema*> all();
    static std::size_t size();
};

}  // namespace lucid

#define LUCID_REGISTER_OP(NAME)                                                                    \
    namespace {                                                                                    \
    struct LucidRegister_##NAME {                                                                  \
        LucidRegister_##NAME() { ::lucid::OpRegistry::register_op(NAME::schema_v1); }              \
    };                                                                                             \
    static const LucidRegister_##NAME _lucid_register_##NAME{};                                    \
    }
