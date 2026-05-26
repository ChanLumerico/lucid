// lucid/_C/ops/ufunc/Predicate.cpp
//
// Implements isinf, isnan, isfinite, and nan_to_num by routing directly
// through the backend dispatcher.  No autograd node is attached because
// none of these operations are differentiable.

#include "Predicate.h"

#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
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
    // OpScopeFull records the output dtype as Bool — not a->dtype() —
    // so the tracer attaches the correct dtype meta to the emitted
    // OpNode (otherwise downstream consumers misread the predicate
    // result as the input's float dtype).
    OpScopeFull scope{name, a->device(), Dtype::Bool, a->shape()};
    auto& be = backend::Dispatcher::for_device(a->device());
    Storage out;
    if (op == 0)
        out = be.isinf(a->storage(), a->shape(), a->dtype());
    else if (op == 1)
        out = be.isnan(a->storage(), a->shape(), a->dtype());
    else
        out = be.isfinite(a->storage(), a->shape(), a->dtype());
    auto out_impl = fresh(std::move(out), a->shape(), Dtype::Bool, a->device());
    // 3.5 Phase 1.3: trace hook — without this, the OpNode lands in
    // the trace with ``inputs=[]`` and the compile path treats it as
    // a dead-code header (skipping it entirely).  Downstream
    // consumers (cast / sum / etc.) then look up a never-bound output
    // id and silently misbehave (notably: GradScaler's found_inf
    // detection always reads 0, so overflow steps don't skip update).
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out_impl);
    }
    return out_impl;
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

TensorImplPtr
nan_to_num_op(const TensorImplPtr& a, double nan_val, double posinf_val, double neginf_val) {
    Validator::input(a, "nan_to_num.a").non_null();
    OpScopeFull scope{"nan_to_num", a->device(), a->dtype(), a->shape()};
    scope.set_attr("nan", nan_val);
    scope.set_attr("posinf", posinf_val);
    scope.set_attr("neginf", neginf_val);
    Storage out =
        backend::Dispatcher::for_device(a->device())
            .nan_to_num(a->storage(), a->shape(), a->dtype(), nan_val, posinf_val, neginf_val);
    auto result = fresh(std::move(out), a->shape(), a->dtype(), a->device());
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    return result;
}

TensorImplPtr any_op(const TensorImplPtr& a) {
    Validator::input(a, "any.a").non_null();
    OpScopeFull scope{"any", a->device(), a->dtype(), a->shape()};
    Storage out =
        backend::Dispatcher::for_device(a->device()).any(a->storage(), a->shape(), a->dtype());
    auto result = fresh(std::move(out), {}, Dtype::Bool, a->device());
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    return result;
}

TensorImplPtr all_op(const TensorImplPtr& a) {
    Validator::input(a, "all.a").non_null();
    OpScopeFull scope{"all", a->device(), a->dtype(), a->shape()};
    Storage out =
        backend::Dispatcher::for_device(a->device()).all(a->storage(), a->shape(), a->dtype());
    auto result = fresh(std::move(out), {}, Dtype::Bool, a->device());
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    return result;
}

}  // namespace lucid
