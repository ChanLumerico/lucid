// lucid/_C/ops/bfunc/_Detail.h
//
// Internal utilities shared across all binary-operation translation units.
// Nothing in this header is part of the public API; it is included only by
// *.cpp files inside ops/bfunc/.  The header provides two input-validation
// helpers and re-exports the helpers::allocate_cpu and helpers::fresh
// convenience functions under the bfunc_detail namespace so that each .cpp
// file does not need to repeat the using-declarations.

#pragma once

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

using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

// Verify that a and b are non-null, have the same dtype, and live on the same
// device.  Throws the appropriate typed exception on any violation, embedding
// the caller's op name in every error message.
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

// Stronger variant of validate_pair that additionally requires a and b to have
// identical shapes.  Used by operations that do not support broadcasting (e.g.
// Compare, Bitwise, Floordiv).
inline void validate_pair_eq_shape(const TensorImplPtr& a, const TensorImplPtr& b, const char* op) {
    validate_pair(a, b, op);
    if (a->shape() != b->shape())
        throw ShapeMismatch(a->shape(), b->shape(), std::string(op));
}

}  // namespace lucid::bfunc_detail
