// lucid/_C/ops/bfunc/Inplace.cpp
//
// Implements the in-place arithmetic operators.  Each operator delegates to
// the corresponding out-of-place forward function via the shared inplace_apply
// helper, which validates preconditions and hands the forward to inplace::run
// (ops/utils/InplaceGraph.h): computed outside autocast, its result is written
// into the original tensor at that tensor's shape and dtype.

#include "Inplace.h"

#include <string>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../utils/InplaceGraph.h"
#include "Add.h"
#include "Div.h"
#include "Maximum.h"
#include "Minimum.h"
#include "Mul.h"
#include "Pow.h"
#include "Sub.h"

namespace lucid {

namespace {

// Execute fwd_fn(a, b) and write its Storage back into a, preserving a's
// identity (pointer) while updating its contents.
//
// Preconditions checked here:
//   1. Neither a nor b is null.
//   2. a's buffer is not held by anything other than a and its own views —
//      a NumPy array, a storage saved for backward.  The storage swap would
//      leave that holder reading the old values without saying so.  A
//      tensor with live views is written through instead
//      (TensorImpl::write_through), which checks the same thing.  The
//      gradient slot a tensor read from ``.grad`` stands for is not such a
//      holder: that tensor holds the buffer through a pointer of its own,
//      and its write goes into the slot's buffer (take_storage_from).
//
// The result keeps a's shape and dtype (inplace::write_result).
//
// A target whose elements overlap — an expanded view — is refused where the
// result lands (take_storage_from), as the reference refuses it.
//
// The forward runs against a snapshot of a (inplace::snapshot), so this op's
// own node keeps the values a held before the write.  a's version is bumped
// either way (inplace::adopt_graph_position or inplace::detach_and_bump),
// which is what tells another node that saved a that its values moved.
template <typename Fn>
TensorImplPtr
inplace_apply(const TensorImplPtr& a, const TensorImplPtr& b, Fn&& fwd_fn, const char* name) {
    if (!a || !b)
        ErrorBuilder(name).fail("null input");
    // In-place binary ops may promote a 0-d CPU rhs onto a GPU target, but
    // never move ``a`` itself onto ``b``'s device.
    if (a->device() == Device::CPU && b->device() == Device::GPU)
        throw DeviceMismatch(std::string(device_name(a->device())),
                             std::string(device_name(b->device())), name);
    if (a->storage_is_shared() && !a->is_aliased())
        ErrorBuilder(name).fail("in-place op on a tensor that shares storage with something "
                                "other than its views (a tensor saved for backward, a NumPy "
                                "array) — call .clone() first");
    inplace::refuse_on_leaf(a, name);
    // The same three rules the unary family follows — see
    // ``ops/utils/InplaceGraph.h``.  This helper had none of them, so the
    // whole binary family took only the storage and left the derivative
    // behind:
    //
    //     z = x * 1.0; z.mul_(y); z.sum().backward()
    //         dx = 1     (the identity — the gradient z had before)
    //         dy = None  (the second operand was never in the graph)
    //
    //     reference:  dx = y,  dy = x
    //
    // Seven ops, and ``add_`` hid it: its derivative with respect to the
    // first operand really is 1, so the wrong answer and the right one
    // coincided there while ``y`` still received nothing.
    //
    // ``b`` takes part in the snapshot rule: when only ``b`` requires grad,
    // the node still saves ``a`` to differentiate with respect to ``b``.
    // And ``b`` may be ``a`` itself (``x.mul_(x)``), which the node must
    // read as it was too — handed ``a``, it read ``x * x`` for ``x``, and
    // held the tensor whose grad_fn it is.
    return inplace::run(a, name, [&] {
        const auto source = inplace::snapshot(a, b);
        return fwd_fn(source, b.get() == a.get() ? source : b);
    });
}

}  // namespace

TensorImplPtr add_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &add_op, "add_");
}
TensorImplPtr sub_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &sub_op, "sub_");
}
TensorImplPtr mul_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &mul_op, "mul_");
}
TensorImplPtr div_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &div_op, "div_");
}
TensorImplPtr pow_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &pow_op, "pow_");
}
TensorImplPtr maximum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &maximum_op, "maximum_");
}
TensorImplPtr minimum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &minimum_op, "minimum_");
}

}  // namespace lucid
