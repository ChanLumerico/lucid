// lucid/_C/ops/bfunc/_Broadcast.h
//
// Shared NumPy-style broadcasting for the binary ops that do NOT ride the
// ``BinaryKernel`` CRTP base — comparison (``Compare.cpp``), bitwise
// (``Bitwise.cpp``) and floor-division (``Floordiv.cpp``).  Those ops are
// non-differentiable / integer-only, so they dispatch straight to the backend
// instead of through ``BinaryKernel::forward`` (which is where the arithmetic
// ops get their broadcasting).  Historically they required identical shapes via
// ``validate_pair_eq_shape``; that forced ``x < scalar`` / ``x & mask`` /
// ``x // k`` to materialise a full-shape constant for the scalar operand, which
// (a) wastes a buffer in eager and (b) pins the trace-time batch in a
// symbolic-batch compile and aborts MPSGraph.  Routing them through
// :func:`broadcast_pair` makes them broadcast like every reference framework.
//
// Header is pure C++ — safe to include from the bfunc ``.cpp`` files.

#pragma once

#include <utility>

#include "../../core/Device.h"
#include "../../core/Shape.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/BinaryKernel.h"  // detail::broadcast_shapes
#include "../utils/Layout.h"            // broadcast_to_op

namespace lucid {
namespace bfunc_detail {

// The (possibly broadcast) operands of a binary op plus their NumPy broadcast
// output shape.
struct BroadcastedPair {
    TensorImplPtr a;
    TensorImplPtr b;
    Shape shape;
};

// Compute the broadcast output shape of ``a`` and ``b`` and return operands
// ready to feed the backend.
//
// GPU (MLX) backends broadcast the two arrays natively, so the operands pass
// through untouched there — importantly, this adds NO node to a trace, so a
// symbolic-batch compile stays abort-free.  The CPU kernels index both operands
// linearly and cannot broadcast, so the expansion is materialised there with
// the eager-only :func:`broadcast_to_op`.  Compile traces on GPU only, so the
// CPU branch never runs under a tracer.
inline BroadcastedPair broadcast_pair(const TensorImplPtr& a, const TensorImplPtr& b) {
    Shape out_shape =
        (a->shape() == b->shape()) ? a->shape() : detail::broadcast_shapes(a->shape(), b->shape());
    TensorImplPtr aa = a;
    TensorImplPtr bb = b;
    if (a->device() == Device::CPU) {
        if (a->shape() != out_shape)
            aa = broadcast_to_op(a, out_shape);
        if (b->shape() != out_shape)
            bb = broadcast_to_op(b, out_shape);
    }
    return {std::move(aa), std::move(bb), std::move(out_shape)};
}

}  // namespace bfunc_detail
}  // namespace lucid
