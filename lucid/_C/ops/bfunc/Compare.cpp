// lucid/_C/ops/bfunc/Compare.cpp
//
// Implements the six element-wise comparison operators by routing through a
// single shared dispatch helper.  The operator is identified by an integer op
// code that the backend uses to select the appropriate comparison kernel.

#include "Compare.h"

#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "_Broadcast.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::broadcast_pair;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

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
    // dtype / device only — comparisons broadcast NumPy-style (matches the
    // arithmetic ops + MLX / reference-framework semantics), so the shapes need
    // not be identical.  Previously this required equal shapes, which forced
    // ``x < scalar`` to materialise a full-shape constant; broadcasting lets a
    // 0-dim scalar ride through (and is what the symbolic-batch compile path
    // needs, since a full-shape scalar pins the batch).
    validate_pair(a, b, name);
    auto bc = broadcast_pair(a, b);
    OpScopeFull scope{name, a->device(), a->dtype(), bc.shape};
    Storage out = backend::Dispatcher::for_device(a->device())
                      .compare_binary(bc.a->storage(), bc.b->storage(), bc.shape, a->dtype(), op);
    auto result = fresh(std::move(out), bc.shape, Dtype::Bool, a->device());
    // 3.5 Phase 1.3: comparisons are non-differentiable so they bypass
    // ``wire_autograd`` — push the I/O wiring into the tracer manually
    // so cross_entropy's ``target != ignore_index`` mask shows up in the
    // captured graph.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a, b}, result);
    }
    return result;
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
