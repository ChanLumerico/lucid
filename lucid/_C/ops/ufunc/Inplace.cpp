// lucid/_C/ops/ufunc/Inplace.cpp
//
// Implementation of all in-place unary ops.
//
// The anonymous-namespace helper inplace_unary<Fn> runs the out-of-place op
// (fwd_fn) through inplace::run (ops/utils/InplaceGraph.h), which computes it
// outside autocast and writes the result into the source tensor at its own
// shape and dtype, moving its version and its place in the graph.

#include "Inplace.h"

#include <string>

#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../utils/InplaceGraph.h"
#include "Activation.h"
#include "Arith.h"
#include "Discrete.h"
#include "Exponential.h"
#include "Hyperbolic.h"
#include "ScalarParam.h"
#include "Trig.h"

namespace lucid {

namespace {

// Run fwd_fn on a snapshot of `a` and write the result into `a`.  Returns `a`
// (same pointer).
template <typename Fn>
TensorImplPtr inplace_unary(const TensorImplPtr& a, Fn&& fwd_fn, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    inplace::refuse_on_leaf(a, name);
    // The op runs against a *snapshot*, not against ``a`` itself.
    //
    // A node that saves its input keeps a handle on the tensor it was
    // given, and graph-mode backward re-reads it — so once ``a``'s storage
    // slot has been overwritten, ``sin_`` differentiated through
    // ``grad(create_graph=True)`` computed ``cos(sin(x))`` instead of
    // ``cos(x)``.  Eager ``backward()`` did not notice, because it saves a
    // Storage by value at forward time, which is why the two routes
    // disagreed and only in graph mode.
    //
    // The snapshot shares the buffer rather than copying it, and the
    // write replaces ``a``'s slot rather than the buffer, so
    // the original values stay alive and unmutated for as long as the node
    // needs them — and nothing is allocated when no graph is built.
    return inplace::run(a, name, [&] { return fwd_fn(inplace::snapshot(a)); });
}

}  // namespace

// --- Arithmetic ---
TensorImplPtr neg_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &neg_op, "neg_");
}
TensorImplPtr abs_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &abs_op, "abs_");
}
TensorImplPtr sign_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sign_op, "sign_");
}
TensorImplPtr reciprocal_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &reciprocal_op, "reciprocal_");
}
TensorImplPtr square_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &square_op, "square_");
}
TensorImplPtr cube_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &cube_op, "cube_");
}

// --- Exponential / Logarithm ---
TensorImplPtr exp_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &exp_op, "exp_");
}
TensorImplPtr log_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &log_op, "log_");
}
TensorImplPtr log2_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &log2_op, "log2_");
}
TensorImplPtr sqrt_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sqrt_op, "sqrt_");
}

// --- Trigonometric ---
TensorImplPtr sin_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sin_op, "sin_");
}
TensorImplPtr cos_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &cos_op, "cos_");
}
TensorImplPtr tan_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &tan_op, "tan_");
}
TensorImplPtr arcsin_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &arcsin_op, "arcsin_");
}
TensorImplPtr arccos_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &arccos_op, "arccos_");
}
TensorImplPtr arctan_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &arctan_op, "arctan_");
}

// --- Hyperbolic ---
TensorImplPtr sinh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sinh_op, "sinh_");
}
TensorImplPtr cosh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &cosh_op, "cosh_");
}
TensorImplPtr tanh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &tanh_op, "tanh_");
}

// --- Rounding ---
TensorImplPtr round_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &round_op, "round_");
}
TensorImplPtr floor_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &floor_op, "floor_");
}
TensorImplPtr ceil_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &ceil_op, "ceil_");
}

// --- Activation ---
TensorImplPtr sigmoid_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sigmoid_op, "sigmoid_");
}
TensorImplPtr relu_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &relu_op, "relu_");
}

TensorImplPtr clip_inplace_op(const TensorImplPtr& a, double lo, double hi) {
    return inplace_unary(
        a, [lo, hi](const TensorImplPtr& source) { return clip_op(source, lo, hi); }, "clip_");
}

// The same rules as the ops above, for a result computed elsewhere.
// Indexing assignment uses it for a tensor with live views, whose values
// have to land in the buffer the views read — rebinding the tensor, as
// assignment does otherwise, would leave them holding the old ones.
TensorImplPtr
assign_inplace_op(const TensorImplPtr& a, const TensorImplPtr& value, const std::string& name) {
    Validator::input(a, name + ".a").non_null();
    Validator::input(value, name + ".value").non_null();
    inplace::refuse_on_leaf(a, name.c_str());
    a->write_through(*value, name.c_str());
    if (!inplace::adopt_graph_position(a, value))
        inplace::detach_and_bump(a);
    return a;
}

}  // namespace lucid
