// lucid/_C/ops/ufunc/Predicate.cpp
//
// Implements isinf, isnan, isfinite, and nan_to_num by routing directly
// through the backend dispatcher.  No autograd node is attached because
// none of these operations are differentiable.

#include "Predicate.h"

#include "../../backend/Dispatcher.h"
#include "../../core/Helpers.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"

namespace lucid {

using helpers::fresh;

namespace {

TensorImplPtr predicate_dispatch(const TensorImplPtr& a, const char* name, int op) {
    Validator::input(a, std::string(name) + ".a").non_null();
    OpScopeFull scope{name, a->device(), a->dtype(), a->shape()};
    auto& be = backend::Dispatcher::for_device(a->device());
    Storage out;
    if (op == 0)
        out = be.isinf(a->storage(), a->shape(), a->dtype());
    else if (op == 1)
        out = be.isnan(a->storage(), a->shape(), a->dtype());
    else
        out = be.isfinite(a->storage(), a->shape(), a->dtype());
    return fresh(std::move(out), a->shape(), Dtype::Bool, a->device());
}

}  // namespace

TensorImplPtr isinf_op(const TensorImplPtr& a) {
    return predicate_dispatch(a, "isinf", 0);
}

TensorImplPtr isnan_op(const TensorImplPtr& a) {
    return predicate_dispatch(a, "isnan", 1);
}

TensorImplPtr isfinite_op(const TensorImplPtr& a) {
    return predicate_dispatch(a, "isfinite", 2);
}

TensorImplPtr nan_to_num_op(const TensorImplPtr& a,
                             double nan_val,
                             double posinf_val,
                             double neginf_val) {
    Validator::input(a, "nan_to_num.a").non_null();
    OpScopeFull scope{"nan_to_num", a->device(), a->dtype(), a->shape()};
    Storage out = backend::Dispatcher::for_device(a->device())
                      .nan_to_num(a->storage(), a->shape(), a->dtype(),
                                  nan_val, posinf_val, neginf_val);
    return fresh(std::move(out), a->shape(), a->dtype(), a->device());
}

TensorImplPtr any_op(const TensorImplPtr& a) {
    Validator::input(a, "any.a").non_null();
    OpScopeFull scope{"any", a->device(), a->dtype(), a->shape()};
    Storage out = backend::Dispatcher::for_device(a->device())
                      .any(a->storage(), a->shape(), a->dtype());
    return fresh(std::move(out), {}, Dtype::Bool, a->device());
}

TensorImplPtr all_op(const TensorImplPtr& a) {
    Validator::input(a, "all.a").non_null();
    OpScopeFull scope{"all", a->device(), a->dtype(), a->shape()};
    Storage out = backend::Dispatcher::for_device(a->device())
                      .all(a->storage(), a->shape(), a->dtype());
    return fresh(std::move(out), {}, Dtype::Bool, a->device());
}

}  // namespace lucid
