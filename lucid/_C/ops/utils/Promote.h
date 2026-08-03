// lucid/_C/ops/utils/Promote.h
//
// One line of the kernel templates, for the ops that do not use them.

#pragma once

#include "../../core/OpSchema.h"
#include "../../core/SchemaGuard.h"
#include "../../core/TensorImpl.h"
#include "../ufunc/Astype.h"

namespace lucid {

// Cast an input to the dtype its schema asks for.
//
// ``UnaryKernel``, ``BinaryKernel`` and ``ReduceKernel`` each open with a
// :class:`SchemaGuard` and cast to ``effective_dtype()``, so every op
// built on them inherits both the autocast policy and the integer
// promotion that goes with ``real_valued`` / ``ForceFP32``.  An op that
// assembles its own forward — softmax, variance, the activations that
// need an extra term — has to ask for the same thing, and the ones that
// did not computed in the input's own dtype.
//
// For an integer input that is not an approximation of the answer, it is
// a different answer: ``softmax`` of an int32 tensor came back
// ``[0, 0, 0]`` on Metal and ``var`` came back ``0``, both integer-typed
// and both silently wrong, while the CPU raised ``NotImplementedError``
// for the same call.  Every one of those schemas already said the result
// was float; nothing was reading it.
//
// ``astype_op`` rather than ``maybe_cast_for_kernel``, for the reason
// ``BatchNorm`` gives: the cast tensor carries an ``AstypeBackward``, so
// a float input cast under an autocast scope keeps its place in the
// graph instead of silently detaching.
//
// Parameters
// ----------
// schema : const OpSchema&
//     The op's schema — supplies the AMP policy and ``real_valued``.
// a : const TensorImplPtr&
//     The input.
//
// Returns
// -------
// TensorImplPtr
//     ``a`` unchanged when it is already at the effective dtype,
//     otherwise a cast copy.
inline TensorImplPtr promote_for_schema(const OpSchema& schema, const TensorImplPtr& a) {
    if (!a)
        return a;
    SchemaGuard sg{schema, a->dtype(), a->device()};
    return astype_op(a, sg.effective_dtype());
}

}  // namespace lucid
