// lucid/_C/ops/bfunc/Bitwise.cpp
//
// Implements the three bitwise binary operators by routing through a single
// shared dispatch helper.  Floating-point dtypes are rejected before dispatch.

#include "Bitwise.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair_eq_shape;

// Return true for dtypes that support bitwise operations.  Floating-point
// types are excluded because their bit patterns are not meaningful as bitmasks.
bool is_integer_or_bool(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
    case Dtype::I8:
    case Dtype::I16:
    case Dtype::I32:
    case Dtype::I64:
        return true;
    default:
        return false;
    }
}

// Shift operations require true integer (Bool excluded — shifting a bool
// has no defined meaning).
bool is_integer(Dtype dt) {
    switch (dt) {
    case Dtype::I8:
    case Dtype::I16:
    case Dtype::I32:
    case Dtype::I64:
        return true;
    default:
        return false;
    }
}

// Shared implementation for all bitwise binary operations.
//
// Op codes match the backend's bitwise_binary convention:
//   0 = AND (&)
//   1 = OR  (|)
//   2 = XOR (^)
//
// The output dtype matches the input dtype.  No autograd node is attached
// because bitwise operations on integers have no meaningful gradient.
TensorImplPtr
bit_dispatch(const TensorImplPtr& a, const TensorImplPtr& b, const char* name, int op) {
    validate_pair_eq_shape(a, b, name);
    if (!is_integer_or_bool(a->dtype()))
        ErrorBuilder(name).fail("dtype must be integer or bool");
    OpScopeFull scope{name, a->device(), a->dtype(), a->shape()};
    Storage out = backend::Dispatcher::for_device(a->device())
                      .bitwise_binary(a->storage(), b->storage(), a->shape(), a->dtype(), op);
    return fresh(std::move(out), a->shape(), a->dtype(), a->device());
}

}  // namespace

TensorImplPtr bitwise_and_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return bit_dispatch(a, b, "bitwise_and", 0);
}

TensorImplPtr bitwise_or_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return bit_dispatch(a, b, "bitwise_or", 1);
}

TensorImplPtr bitwise_xor_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return bit_dispatch(a, b, "bitwise_xor", 2);
}

namespace {

// Shared dispatcher for the shift ops.  Differs from ``bit_dispatch`` only in
// the dtype guard (Bool not allowed) — kept as its own helper to keep the
// error messages tight.
TensorImplPtr
shift_dispatch(const TensorImplPtr& a, const TensorImplPtr& b, const char* name, int op) {
    validate_pair_eq_shape(a, b, name);
    if (!is_integer(a->dtype()))
        ErrorBuilder(name).fail("dtype must be a signed integer (I8/I16/I32/I64)");
    OpScopeFull scope{name, a->device(), a->dtype(), a->shape()};
    Storage out = backend::Dispatcher::for_device(a->device())
                      .bitwise_binary(a->storage(), b->storage(), a->shape(), a->dtype(), op);
    return fresh(std::move(out), a->shape(), a->dtype(), a->device());
}

}  // namespace

TensorImplPtr bitwise_left_shift_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return shift_dispatch(a, b, "bitwise_left_shift", 3);
}

TensorImplPtr bitwise_right_shift_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return shift_dispatch(a, b, "bitwise_right_shift", 4);
}

}  // namespace lucid
