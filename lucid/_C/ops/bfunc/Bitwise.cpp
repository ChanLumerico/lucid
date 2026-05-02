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

}  // namespace lucid
