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
