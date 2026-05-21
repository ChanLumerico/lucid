// lucid/_C/ops/bfunc/_BinaryOp.h
//
// Thin alias re-export of :class:`BinaryKernel` under the shorter name
// :class:`BinaryOp` for use throughout the ``ops/bfunc/`` subsystem.
//
// Every concrete backward node in ``ops/bfunc/`` — :class:`AddBackward`,
// :class:`SubBackward`, :class:`MulBackward`, :class:`DivBackward`,
// :class:`PowBackward`, :class:`MatmulBackward`, etc. — inherits from
// this alias as ``class FooBackward : public BinaryOp<FooBackward>`` so
// that the bfunc translation units do not need to reach back into the
// ``kernel/`` directory directly.  The full CRTP machinery —
// broadcasting, dtype/device validation, autograd graph wiring, the
// ``forward()`` dispatch trampoline, and the ``apply()`` /
// ``apply_for_graph()`` backward entry points — lives in
// :file:`kernel/BinaryKernel.h`.
//
// Notes
// -----
// The alias is the only symbol this header introduces; downstream files
// include it instead of ``kernel/BinaryKernel.h`` directly so that the
// public/internal layering is preserved.

#pragma once

#include "../../kernel/BinaryKernel.h"

namespace lucid {

// CRTP base for binary-op backward nodes — alias of :class:`BinaryKernel`.
//
// Concrete op classes declared as
// ``class FooBackward : public BinaryOp<FooBackward>`` plug into the
// kernel layer's two-input, single-output contract:
//
//   - ``static constexpr OpSchema schema_v1`` — op name + AMP policy.
//   - ``static cpu_kernel(a, b, out_shape, dtype) -> CpuStorage`` and/or
//     ``static gpu_kernel(a, b, out_shape, dtype) -> GpuStorage`` —
//     the per-backend compute hooks (alternatively, ``static dispatch``
//     routes through :class:`backend::IBackend`).
//   - ``grad_formula(grad_out) -> std::tuple<Storage, Storage>`` — the
//     local Jacobian product for both inputs, computed at backward time.
//   - Optional ``grad_formula_impl(grad_out, a_impl, b_impl) ->
//     std::pair<TensorImplPtr, TensorImplPtr>`` — graph-mode (i.e.
//     ``create_graph=True``) backward that retains grad_fn through the
//     gradient computation itself, enabling higher-order derivatives.
//
// The base supplies all the shared bookkeeping: two saved input
// :class:`Storage` slots (via ``AutogradNode<Derived, 2>``), broadcast
// shape inference at forward, ``sum_to_shape`` reduction of each
// gradient branch back to its original input shape at backward, AMP
// dtype promotion forwarding through :class:`SchemaGuard`, and version
// capture for in-place mutation detection.
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete backward class (CRTP self-type).  Must expose the
//     static schema/kernel hooks listed above.
//
// Notes
// -----
// Saved slot count is fixed at 2.  Ops whose backward does not need
// the original inputs (e.g. :class:`AddBackward`, where the local
// Jacobian is identity) opt out by overriding ``kSavesInputs`` to
// ``false`` in the derived class to skip the storage capture at
// forward.  Ops that additionally need the forward *output* (e.g.
// :class:`PowBackward`) save it separately on the derived node.
//
// See Also
// --------
// :class:`BinaryKernel` — the underlying CRTP base; full implementation
//     in :file:`kernel/BinaryKernel.h`.
// :class:`UnaryOp` — sibling alias for elementwise unary ops in
//     :file:`ops/ufunc/_UnaryOp.h`.
// :class:`FuncOp` — the more general autograd CRTP base in
//     :file:`autograd/FuncOp.h`.
template <class Derived>
using BinaryOp = BinaryKernel<Derived>;

}
