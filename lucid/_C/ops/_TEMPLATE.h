// lucid/_C/ops/_TEMPLATE.h
//
// Developer scaffolding for adding a new op to the Lucid engine.  This
// file is NOT compiled into the engine — it exists purely as a
// reference pattern.  Copy it to the appropriate ``ops/`` subdirectory,
// rename every occurrence of ``my_op`` / ``MyOp``, fill in the forward
// dispatch and backward formula, and register the backward node with
// ``LUCID_REGISTER_OP``.
//
// Checklist for a new op
// ----------------------
// 1. Create ``MyOp.h`` declaring ``MyOpBackward`` and the free function
//    ``my_op()``.
// 2. Create ``MyOp.cpp`` implementing both.
// 3. Register the backward node with ``LUCID_REGISTER_OP(MyOpBackward)``.
// 4. Add ``linalg_my_op()`` (or the appropriate category) to ``IBackend``
//    and both the CPU (``backend/cpu/``) and GPU (``backend/gpu/``)
//    implementations.
// 5. Expose the op to Python via the ``_bindings/`` layer.
//
// File organisation convention
// ----------------------------
// - Header (``.h``):  class declaration for the backward node + function
//   declaration for the forward entry point.
// - Source (``.cpp``): implementations of both; ``#include`` directives
//   are kept in the ``.cpp`` to reduce build-graph coupling.
//
// Naming conventions
// ------------------
// - Backward node class: ``<Name>Backward``  (e.g. ``MatmulBackward``)
// - Free function:       ``<name>_op``        (e.g. ``matmul_op``)
// - OpSchema tag string: matches the Python-visible op name (lower-snake)
//
// Autograd wiring summary
// -----------------------
// - ``FuncOp<MyOpBackward, N>`` is the CRTP base for a backward node
//   that receives one upstream gradient and produces $N$ input
//   gradients.
// - ``NaryKernel<MyOpBackward, N>::wire_autograd()`` attaches the
//   backward node to the output ``TensorImpl`` and links it to the $N$
//   input ``TensorImpl``\ s.  The final ``bool`` argument
//   (``save_inputs``) controls whether the input storages are copied
//   into ``saved_inputs_`` on the backward node.
// - ``NoGradGuard`` inside ``apply()`` prevents ops called during the
//   backward pass from themselves creating new backward nodes (wrong
//   for first-order, wasteful for second-order).
// - If the backward only needs the forward output (not the inputs),
//   set ``save_inputs=false`` and populate ``bwd->saved_output_``
//   manually after the forward dispatch.  See ``Inv.cpp`` for the
//   canonical example.
// - If the backward needs the original inputs, set ``save_inputs=true``;
//   ``NaryKernel`` will populate ``saved_inputs_[0..N-1]`` automatically.
//   See ``Det.cpp`` for the canonical example.

#pragma once

#include <memory>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Forward entry point for a new op — replace ``my_op`` with the real name.
//
// Every op entry point follows the same four-step pattern.
//
// 1. Validate: call ``Validator::input()`` and ``Validator::pair()`` as
//    needed.  Validation throws ``LucidError`` on bad inputs; it must
//    happen before any allocation so we do not leak storage on error.
// 2. Scope: construct an ``OpScopeFull`` to record profiling and
//    debugging info.
// 3. Dispatch: call ``backend::Dispatcher::for_device(a->device()).<method>()``
//    to obtain the output ``Storage``.  The dispatcher selects the CPU
//    or GPU backend based on the input's device tag.
// 4. Wire autograd: if ``GradMode::is_enabled()`` and any input has
//    ``requires_grad``, construct the backward node, populate
//    ``saved_inputs_`` / ``saved_output_``, and call
//    ``NaryKernel<MyOpBackward, N>::wire_autograd()``.  The final
//    ``bool`` controls whether inputs are saved (true) or only the
//    output is saved (false).
//
// Parameters
// ----------
// a : TensorImplPtr
//     First operand (placeholder).
// b : TensorImplPtr
//     Second operand (placeholder).
//
// Returns
// -------
// TensorImplPtr
//     Output tensor produced by the dispatched kernel.
//
// Notes
// -----
// - ``LUCID_API`` exports the symbol from the shared library so the
//   Python extension module and other engine components can link to it.
// - Header (``.h``) note: declare only the function signature here.
//   Put all ``#include``\ s (backend headers, autograd headers, op
//   helpers) in the ``.cpp`` to avoid cascading recompilation across
//   the whole engine when implementation details change.
LUCID_API TensorImplPtr my_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
