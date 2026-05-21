// lucid/_C/ops/ufunc/_UnaryOp.h
//
// Internal alias header that exposes :class:`UnaryOp` — the canonical CRTP
// base every backward node in ``lucid/_C/ops/ufunc/`` inherits from for
// single-input element-wise operations.
//
// The file does nothing more than rename :class:`UnaryKernel` (defined in
// ``lucid/_C/kernel/UnaryKernel.h``) to :class:`UnaryOp` and keep the
// public ufunc headers from leaking the ``kernel/`` include graph.  Every
// concrete unary op in the directory (``NegBackward``, ``LogBackward``,
// ``ReluBackward``, …) is declared as
// ``class FooBackward : public UnaryOp<FooBackward>``, so although the
// filename starts with an underscore the symbol :class:`UnaryOp` itself is
// part of the **public implementation contract** consumed by every sibling
// op header.
//
// Notes
// -----
// Decoupling rationale.  Public ufunc headers (``Arith.h``, ``Exponential.h``,
// ``Activation.h``, …) include only this thin alias header; they never
// pull in :class:`UnaryKernel` directly.  This keeps the include depth of
// the ufunc subsystem one level shallower and means a future swap of the
// kernel-layer base class only has to touch one ``using`` line here.
//
// This header should not be included from non-ufunc translation units —
// downstream consumers (Python bindings, nn modules) call the public
// ``foo_op(a)`` entry points, never the backward classes directly.

#pragma once

#include "../../kernel/UnaryKernel.h"

namespace lucid {

// CRTP base alias for single-input element-wise op backward nodes.
//
// :class:`UnaryOp` is a one-line ``using`` alias for
// :class:`UnaryKernel`; concrete ops inherit through this alias purely
// for include-graph hygiene (see file header).  The full attribute list
// and dispatch contract live on :class:`UnaryKernel` — this comment
// summarises only the surface that ufunc op authors interact with.
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete backward class (CRTP self-type).  Must expose the
//     static / instance members listed under *Required members* below.
//
// Attributes
// ----------
// kSavesInput : static constexpr bool, default ``true``
//     If ``true``, ``UnaryKernel::forward`` snapshots the forward
//     input's :class:`Storage` into ``saved_inputs_[0]`` so that
//     :meth:`grad_formula` can recover $x$ during backward.  Set to
//     ``false`` (e.g. :class:`NegBackward`, :class:`SumBackward`) when
//     the gradient rule is independent of $x$.
// kSavesOutput : static constexpr bool, default ``false``
//     If ``true``, ``UnaryKernel::forward`` snapshots the forward
//     output into ``saved_output_`` instead — used by
//     :class:`SigmoidBackward`, :class:`TanhBackward`, etc. whose
//     analytic derivative is cheaper in terms of $y$ than $x$.
// kHasGradient : static constexpr bool, default ``true``
//     Set to ``false`` for ops with no useful gradient
//     (e.g. :class:`SignBackward`, :class:`RoundBackward`); skips
//     autograd wiring entirely.
//
// Notes
// -----
// **Required members on Derived.**
//
// - ``static const OpSchema schema_v1`` — registered op name + AMP
//   policy.  The ``_v1`` suffix is the schema-version namespace; ops
//   that change wire format add a ``schema_v2`` alongside.
// - Exactly one forward path:
//   ``static Storage dispatch(IBackend&, const Storage&, const Shape&,
//   Dtype)`` *or* a matched pair of
//   ``static {Cpu,Gpu}Storage cpu_kernel / gpu_kernel(...)`` overloads.
// - ``Storage grad_formula(const Storage& grad_out)`` — instance method
//   returning ``dL/dx``.  ``UnaryKernel::apply`` calls it and then
//   broadcasts the result back to the original input shape.
//
// **Dispatch priority inside forward.**
//
// 1. ``Derived::dispatch(IBackend&, …)`` when :concept:`HasUnaryDispatch`
//    is satisfied.
// 2. ``Derived::gpu_kernel(GpuStorage, …)`` when on the GPU stream.
// 3. ``Derived::cpu_kernel(CpuStorage, …)`` on the CPU stream.
//
// **AMP.**  ``schema_v1`` carries the autocast policy; backward inherits
// the casted dtype via the saved :class:`Storage`, so :meth:`grad_formula`
// does not need additional AMP logic.
//
// See Also
// --------
// :class:`UnaryKernel` — full implementation, dispatch priority,
//     SchemaGuard / contiguous-input handling, autograd edge wiring.
// :class:`ReduceOp`    — sibling base for reductions
//     (see ``_ReduceOp.h``).
// :class:`FuncOp`      — the more general autograd CRTP alias used by
//     binary and n-ary ops.
template <class Derived>
using UnaryOp = UnaryKernel<Derived>;

}  // namespace lucid
