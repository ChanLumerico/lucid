// lucid/_C/ops/bfunc/Inplace.cpp
//
// Implements the in-place arithmetic operators.  Each operator delegates to
// the corresponding out-of-place forward function via the shared inplace_apply
// helper, which validates preconditions and splices the result Storage back
// into the original tensor.

#include "Inplace.h"

#include <utility>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
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
//   2. a does not share storage with any view tensor (would corrupt the view).
//   3. The out-of-place result has the same shape as a (in-place ops may not
//      change shape; this would also silently break any live views of a).
//
// After the Storage swap, a->bump_version() invalidates any backward nodes that
// hold a saved reference to a's old storage, making stale-gradient bugs loud.
template <typename Fn>
TensorImplPtr
inplace_apply(const TensorImplPtr& a, const TensorImplPtr& b, Fn&& fwd_fn, const char* name) {
    if (!a || !b)
        ErrorBuilder(name).fail("null input");
    if (a->storage_is_shared())
        ErrorBuilder(name).fail("in-place op on a tensor that shares storage with a view — "
                                "call .clone() first or operate on the base tensor");
    auto out = fwd_fn(a, b);
    if (out->shape() != a->shape())
        throw ShapeMismatch(a->shape(), out->shape(),
                            std::string(name) + " (in-place: shape changed)");
    // Splice the new Storage and metadata back into a.
    a->mutable_storage() = std::move(out->mutable_storage());
    a->set_dtype(out->dtype());
    a->set_device(out->device());
    // Increment the version counter so that any backward node that retained a
    // weak reference to a will detect the mutation during validate_versions().
    a->bump_version();
    return a;
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
