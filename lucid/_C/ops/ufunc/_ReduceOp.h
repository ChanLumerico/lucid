// lucid/_C/ops/ufunc/_ReduceOp.h
//
// Internal alias header that exposes :class:`ReduceOp` — the canonical
// CRTP base every multi-axis reduction backward node in
// ``lucid/_C/ops/ufunc/`` inherits from.
//
// Like :class:`UnaryOp` in ``_UnaryOp.h``, this file only renames the
// kernel-layer :class:`ReduceKernel` to a shorter public name so the
// public ufunc headers (``Reductions.h``, ``Trace.h``, ``Var.h``, …) do
// not have to pull in the full ``kernel/`` include graph.  Although the
// filename is underscore-prefixed, the alias :class:`ReduceOp` itself is
// part of the **public implementation contract** consumed by every
// reduction backward class in the directory
// (:class:`SumBackward`, :class:`MeanBackward`, :class:`MaxBackward`, …).
//
// Notes
// -----
// Reductions differ from unary ops in that the output shape collapses
// the requested axes, so the backward pass has to *broadcast* the
// upstream gradient back to the full input extent before applying any
// per-op scaling rule.  :class:`ReduceKernel` therefore stores the
// axis-normalisation result and the original input shape on the
// backward node so ``grad_formula`` can rebuild the forward transform.
//
// This header should not be included from non-ufunc translation units.

#pragma once

#include "../../kernel/ReduceKernel.h"

namespace lucid {

// CRTP base alias for multi-axis reduction backward nodes.
//
// ``ReduceOp<Derived>`` is a one-line ``using`` alias for
// :class:`ReduceKernel<Derived>`; the indirection exists purely to keep
// public ufunc headers decoupled from ``kernel/`` (see file header).
// Concrete reduction backwards inherit through this alias:
//
//     class SumBackward : public ReduceOp<SumBackward> { ... };
//     class MeanBackward : public ReduceOp<MeanBackward> { ... };
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete reduction backward class (CRTP self-type).  Must
//     expose the static / instance members listed under
//     *Required members* below.
//
// Attributes
// ----------
// reduce_axes_ : std::vector<int>
//     Normalised axis list (non-negative, sorted ascending, with
//     duplicates removed).  Populated by ``ReduceKernel::forward`` from
//     the user-supplied ``axes_user`` argument and consulted by
//     :meth:`grad_formula` to broadcast ``grad_out`` back to the
//     pre-reduction extent.
// keepdims_ : bool
//     Whether the forward kept reduced axes as size-1 dimensions.
//     Determines the shape of ``grad_out`` entering :meth:`grad_formula`.
// full_input_shape_ : Shape
//     Snapshot of the input shape *before* reduction; used as the target
//     broadcast extent for the gradient.
// kSavesInput : static constexpr bool, default ``true``
//     If ``true`` (the default on :class:`ReduceKernel`), the forward
//     input :class:`Storage` is saved in ``saved_inputs_[0]`` for use
//     in :meth:`grad_formula`.  Ops with $x$-independent gradient rules
//     (:class:`SumBackward`, :class:`MeanBackward`) override this to
//     ``false``.
// kSavesOutput : static constexpr bool, default ``false``
//     Set to ``true`` by :class:`ProdBackward` so the forward output
//     $y$ is available for the $\partial y/\partial x_i = y / x_i$ rule.
//
// Notes
// -----
// **Required members on Derived.**
//
// - ``static const OpSchema schema_v1`` — registered op name + AMP
//   policy.
// - Exactly one forward path:
//   ``static Storage dispatch(IBackend&, const Storage&, const Shape&,
//   const std::vector<int>& axes, bool keepdims, Dtype)`` *or* a matched
//   pair of ``static {Cpu,Gpu}Storage cpu_kernel / gpu_kernel(...)``
//   overloads.
// - ``Storage grad_formula(const Storage& grad_out)`` — instance method
//   that, after broadcasting ``grad_out`` along the reduced axes, applies
//   the per-op gradient rule (identity for ``sum``, scale-by-$1/N$ for
//   ``mean``, mask for ``max`` / ``min``, ratio-with-output for ``prod``).
// - ``TensorImplPtr scale_graph_grad(const TensorImplPtr& g)`` —
//   optional hook applied by ``apply()`` after the broadcast; defaults
//   to identity but ``MeanBackward`` overrides it to divide by the
//   number of reduced elements.
//
// **Forward responsibilities.**  ``ReduceKernel::forward`` normalises
// ``axes_user`` against the rank of ``a``, allocates the reduced output,
// dispatches via ``IBackend::dispatch`` / ``gpu_kernel`` / ``cpu_kernel``,
// then populates ``reduce_axes_``, ``keepdims_``, and
// ``full_input_shape_`` on the constructed backward node before wiring it
// in as the output's ``grad_fn``.
//
// **AMP.**  Same contract as :class:`UnaryOp` — ``schema_v1`` carries
// the cast policy, backward inherits the dtype via the saved
// :class:`Storage`.
//
// See Also
// --------
// :class:`ReduceKernel` — full implementation, axis normalisation,
//     SchemaGuard / dispatch priority, autograd edge wiring.
// :class:`UnaryOp`     — sibling base for shape-preserving unary ops
//     (see ``_UnaryOp.h``).
// :class:`FuncOp`      — the more general autograd CRTP alias.
template <class Derived>
using ReduceOp = ReduceKernel<Derived>;

}  // namespace lucid
