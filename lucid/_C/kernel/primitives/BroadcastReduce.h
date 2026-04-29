#pragma once

// =====================================================================
// Lucid C++ engine — kernel/primitives/BroadcastReduce.h
// =====================================================================
//
// Primitive: sum-reduce a gradient tensor back to a target shape.
//
// Used by every binary op's backward to undo NumPy-style broadcasting:
//   dx = reduce_grad_to_shape(grad_out, out_shape, a.shape, dtype, device)
//   dy = reduce_grad_to_shape(grad_out, out_shape, b.shape, dtype, device)
//
// Also used by BroadcastBackward::apply() to undo broadcast_to.
//
// Layer: kernel/primitives/. Thin re-export of autograd/Helpers.h so
// kernel/ layer code can include this without reaching into autograd/ directly.
//
// Implementation lives in autograd/Helpers.cpp.

#include "../../autograd/Helpers.h"

// Everything is already declared in autograd/Helpers.h:
//
//   Storage reduce_grad_to_shape(const Storage& grad,
//                                const Shape& grad_shape,
//                                const Shape& target_shape,
//                                Dtype dtype, Device device);
//
//   Storage make_zero_storage(const Shape&, Dtype, Device);
//   Storage make_ones_storage(const Shape&, Dtype, Device);
//   void accumulate_into(Storage& dst, const Storage& src);
//
// Include this header instead of autograd/Helpers.h when writing code at
// kernel/ or ops/ layer that only needs the BroadcastReduce primitive.
