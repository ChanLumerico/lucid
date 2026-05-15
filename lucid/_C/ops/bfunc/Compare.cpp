// lucid/_C/ops/bfunc/Compare.cpp
//
// Implements the six element-wise comparison operators by routing through a
// single shared dispatch helper.  The operator is identified by an integer op
// code that the backend uses to select the appropriate comparison kernel.

#include "Compare.h"

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

// Shared implementation for all binary comparisons.
//
// Op codes match the backend's compare_binary convention:
//   0 = equal (==)
//   1 = not_equal (!=)
//   2 = greater (>)
//   3 = greater_equal (>=)
//   4 = less (<)
//   5 = less_equal (<=)
//
// The output dtype is always Bool regardless of the input dtype.  No autograd
// node is attached because comparisons are not differentiable.
TensorImplPtr
cmp_dispatch(const TensorImplPtr& a, const TensorImplPtr& b, const char* name, int op) {
    validate_pair_eq_shape(a, b, name);
    OpScopeFull scope{name, a->device(), a->dtype(), a->shape()};
    Storage out = backend::Dispatcher::for_device(a->device())
                      .compare_binary(a->storage(), b->storage(), a->shape(), a->dtype(), op);
    return fresh(std::move(out), a->shape(), Dtype::Bool, a->device());
}

}  // namespace

TensorImplPtr equal_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(a, b, "equal", 0);
}

TensorImplPtr not_equal_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(a, b, "not_equal", 1);
}

TensorImplPtr greater_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(a, b, "greater", 2);
}

TensorImplPtr greater_equal_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(a, b, "greater_equal", 3);
}

TensorImplPtr less_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(a, b, "less", 4);
}

TensorImplPtr less_equal_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return cmp_dispatch(a, b, "less_equal", 5);
}

}  // namespace lucid
