#pragma once

// =====================================================================
// bfunc internal helpers — shared by Compare/Bitwise/Dot/Inner/Outer/Tensordot.
// Not included by user code. Header-only inline functions to avoid linker
// duplicates.
// =====================================================================

#include <cstring>
#include <stdexcept>
#include <variant>

#include "../../api.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::bfunc_detail {

// Re-exports of the canonical helpers in `core/Helpers.h`. The aliases
// keep existing call sites (`bfunc_detail::allocate_cpu(...)`) working
// while migration to the unqualified `lucid::helpers::` form proceeds.
using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

inline void validate_pair(const TensorImplPtr& a, const TensorImplPtr& b, const char* op) {
    if (!a || !b)
        ErrorBuilder(op).fail("null input");
    if (a->dtype() != b->dtype())
        throw DtypeMismatch(std::string(dtype_name(a->dtype())),
                            std::string(dtype_name(b->dtype())), std::string(op));
    if (a->device() != b->device())
        throw DeviceMismatch(std::string(device_name(a->device())),
                             std::string(device_name(b->device())), std::string(op));
}

inline void validate_pair_eq_shape(const TensorImplPtr& a, const TensorImplPtr& b, const char* op) {
    validate_pair(a, b, op);
    if (a->shape() != b->shape())
        throw ShapeMismatch(a->shape(), b->shape(), std::string(op));
}

}  // namespace lucid::bfunc_detail
