// lucid/_C/ops/composite/Logical.cpp
//
// Coerce each operand to a bool mask via ``not_equal(x, 0)`` then forward to
// the matching ``bitwise_*`` kernel.  ``logical_not`` is just ``equal(x, 0)``.

#include "Logical.h"

#include "../../core/ErrorBuilder.h"
#include "../../kernel/BinaryKernel.h"
#include "../bfunc/Bitwise.h"
#include "../bfunc/Compare.h"
#include "../gfunc/Gfunc.h"

namespace lucid {

namespace {

// Build a bool mask ``x != 0`` with shape/device/dtype matching ``x``.
TensorImplPtr to_bool_mask(const TensorImplPtr& x) {
    auto zero = full_like_op(x, 0.0);
    return not_equal_op(x, zero);
}

}  // namespace

TensorImplPtr logical_and_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    if (!a_in || !b_in)
        ErrorBuilder("logical_and").fail("null input");
    const auto [a, b] = detail::align_binary_devices(a_in, b_in, "logical_and");
    return bitwise_and_op(to_bool_mask(a), to_bool_mask(b));
}

TensorImplPtr logical_or_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    if (!a_in || !b_in)
        ErrorBuilder("logical_or").fail("null input");
    const auto [a, b] = detail::align_binary_devices(a_in, b_in, "logical_or");
    return bitwise_or_op(to_bool_mask(a), to_bool_mask(b));
}

TensorImplPtr logical_xor_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    if (!a_in || !b_in)
        ErrorBuilder("logical_xor").fail("null input");
    const auto [a, b] = detail::align_binary_devices(a_in, b_in, "logical_xor");
    return bitwise_xor_op(to_bool_mask(a), to_bool_mask(b));
}

TensorImplPtr logical_not_op(const TensorImplPtr& a) {
    auto zero = full_like_op(a, 0.0);
    return equal_op(a, zero);
}

}  // namespace lucid
