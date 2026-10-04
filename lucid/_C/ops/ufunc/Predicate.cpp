// lucid/_C/ops/ufunc/Predicate.cpp
//
// Implements isinf, isnan and isfinite by routing directly through the
// backend dispatcher.  The predicates carry no autograd node — their Bool
// output has no derivative.  ``nan_to_num`` does: it is the identity on
// finite values, so it lives with the scalar-parameter ops
// (``NanToNumBackward`` in ScalarParam.cpp) and this entry point delegates.

#include "Predicate.h"

#include <limits>

#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/Helpers.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "ScalarParam.h"

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

namespace {

// The largest finite value of ``dt``, which is the reference framework's
// replacement for +inf when none is given.  For complex dtypes it is the
// largest finite value of the parts.  Integer and bool tensors hold nothing
// to replace, so for them the value is never used.  It stays the float32
// figure the default always was, which keeps a traced integer graph as it
// was.
double largest_finite(Dtype dt) {
    switch (dt) {
    case Dtype::F16:
        return 65504.0;
    case Dtype::BF16:
        return 3.3895313892515355e+38;  // 0x7F7F: float32's max with 7 mantissa bits
    case Dtype::F64:
    case Dtype::C128:
        return std::numeric_limits<double>::max();
    default:
        return static_cast<double>(std::numeric_limits<float>::max());
    }
}

}  // namespace

TensorImplPtr nan_to_num_op(const TensorImplPtr& a,
                            std::optional<double> nan_val,
                            std::optional<double> posinf_val,
                            std::optional<double> neginf_val) {
    Validator::input(a, "nan_to_num.a").non_null();
    // The defaults are resolved here, against the input's dtype, so every
    // backend and the trace see the value that was used.
    const double top = largest_finite(a->dtype());
    return NanToNumBackward::forward(a, nan_val.value_or(0.0), posinf_val.value_or(top),
                                     neginf_val.value_or(-top));
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
