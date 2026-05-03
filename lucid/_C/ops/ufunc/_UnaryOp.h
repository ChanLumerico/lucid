// lucid/_C/ops/ufunc/_UnaryOp.h
//
// Central alias that maps the public-facing name `UnaryOp<Derived>` to the
// full CRTP implementation `UnaryKernel<Derived>`.  Every backward node in the
// ufunc subsystem (NegBackward, ExpBackward, …) inherits from UnaryOp<Derived>
// rather than UnaryKernel<Derived> directly, so that the include graph stays
// shallow and the public interface of the subsystem is decoupled from the
// kernel machinery in lucid/_C/kernel/.

#pragma once

#include "../../kernel/UnaryKernel.h"

namespace lucid {

// Single-input element-wise op base.  Inherit from this via CRTP.
//
// UnaryKernel<Derived> provides:
//   - static forward(a) — allocates output, dispatches to IBackend or
//     gpu_kernel/cpu_kernel, wires autograd edges, sets kSavesInput /
//     kSavesOutput saved tensors.
//   - apply(grad_out)   — calls Derived::grad_formula(grad_out), then
//     broadcasts the result back to the input shape if needed.
//
// Derived must supply:
//   static const OpSchema schema_v1;
//   static Storage dispatch(IBackend&, const Storage&, const Shape&, Dtype);
//       — or gpu_kernel / cpu_kernel overloads when dispatch() is absent.
//   Storage grad_formula(const Storage& g);
//
// Derived may override the default save policy via constexpr bool members:
//   kSavesInput  (default true)  — stores a_ptr->storage() in saved_inputs_[0].
//   kSavesOutput (default false) — stores out->storage() in saved_output_.
//   kHasGradient (default true)  — skips autograd wiring when false.
template <class Derived>
using UnaryOp = UnaryKernel<Derived>;

}  // namespace lucid
