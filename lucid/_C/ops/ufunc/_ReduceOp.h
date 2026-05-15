// lucid/_C/ops/ufunc/_ReduceOp.h
//
// Central alias that maps the public-facing name `ReduceOp<Derived>` to the
// full CRTP implementation `ReduceKernel<Derived>`.  Reduction backward nodes
// (SumBackward, MeanBackward, …) inherit from ReduceOp<Derived> instead of
// ReduceKernel<Derived> directly, keeping the ufunc layer insulated from the
// kernel layer's include graph.

#pragma once

#include "../../kernel/ReduceKernel.h"

namespace lucid {

// Multi-axis reduction op base.  Inherit from this via CRTP.
//
// ReduceKernel<Derived> provides:
//   - static forward(a, axes_user, keepdims) — normalises axes, allocates the
//     reduced output, dispatches to IBackend::dispatch or gpu_kernel /
//     cpu_kernel, and wires autograd with reduce_axes_ / keepdims_ /
//     full_input_shape_ saved on the backward node.
//   - apply(grad_out) — calls Derived::grad_formula(grad_out) and returns the
//     result as a single-element vector.
//
// Derived must supply:
//   static const OpSchema schema_v1;
//   static Storage dispatch(IBackend&, const Storage&, const Shape&,
//                           const std::vector<int>&, bool keepdims, Dtype);
//       — or gpu_kernel / cpu_kernel overloads.
//   Storage grad_formula(const Storage& grad_out);
//
// ReduceKernel also stores reduce_axes_, keepdims_, and full_input_shape_
// fields that grad_formula implementations typically use to broadcast the
// upstream gradient back to the input shape.
template <class Derived>
using ReduceOp = ReduceKernel<Derived>;

}  // namespace lucid
