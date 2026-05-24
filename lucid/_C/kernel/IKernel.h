// lucid/_C/kernel/IKernel.h
//
// Abstract polymorphic interface implemented by every op kernel in the
// framework, plus a documentation-only :class:`KernelPolicy` struct that
// enumerates the compile-time flags concrete CRTP kernels may override.
//
// The kernel layer sits between :file:`ops/` (op-level backward nodes)
// and :file:`backend/` (device-specific compute).  Concrete kernels
// inherit :class:`IKernel` alongside one of the CRTP helpers
// (:class:`UnaryKernel`, :class:`BinaryKernel`, :class:`NaryKernel`,
// :class:`VariadicKernel`) so they can be stored as ``IKernel*`` for
// generic dispatch / profiling / testing while still benefiting from the
// CRTP base's typed ``forward()`` and saved-tensor bookkeeping.

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {
namespace kernel {

// Documentation-only enumeration of the compile-time policy flags a
// concrete kernel may override on its CRTP base.
//
// Derived kernels do not instantiate this struct — they override the
// equivalent ``static constexpr bool kSavesInput`` / ``kSavesOutput`` /
// ``kHasGradient`` members on :class:`UnaryKernel` (or sibling bases).
// :class:`KernelPolicy` exists purely so the flag set is documented
// in one place and discoverable via codegen tools.
//
// Attributes
// ----------
// saves_inputs : bool
//     Whether ``forward()`` snapshots the input storages into
//     ``saved_inputs_`` for later use in ``apply()`` / ``grad_formula``.
//     Set ``false`` for ops whose backward formula does not need the
//     forward inputs (e.g. ``Add`` whose backward is the identity).
// saves_output : bool
//     Whether ``forward()`` additionally snapshots the output storage
//     into ``saved_output_``.  Useful for ops whose backward is expressed
//     more cheaply in terms of the output (e.g. ``ReLU``, ``Softmax``,
//     ``Pow``).
// has_gradient : bool
//     Whether the op participates in autograd at all.  Setting ``false``
//     skips graph wiring entirely — used by in-place / non-differentiable
//     ops such as integer cast or copy.
// deterministic : bool
//     Whether the op produces bit-identical outputs for bit-identical
//     inputs.  Currently informational only; consumed by future
//     determinism-mode validators.
//
// See Also
// --------
// :class:`IKernel` — the abstract base these flags refine.
// :class:`UnaryKernel`, :class:`BinaryKernel` — CRTP bases that read
// these flags via ``Derived::kSavesInput`` etc.
struct KernelPolicy {
    bool saves_inputs = true;
    bool saves_output = false;
    bool has_gradient = true;
    bool deterministic = true;
};

// Abstract base interface every Lucid op kernel implements.
//
// :class:`IKernel` exposes a minimal polymorphic surface so the
// autograd engine, profiler, and testing harnesses can manipulate
// kernels without knowing the concrete CRTP type.  Concrete kernels
// inherit :class:`IKernel` indirectly via one of the CRTP helpers
// (:class:`UnaryKernel`, :class:`BinaryKernel`, :class:`NaryKernel`,
// :class:`VariadicKernel`); the CRTP layer supplies correct overrides
// of every virtual below.
//
// Notes
// -----
// The interface is intentionally split: :meth:`compute` is the generic
// polymorphic forward entry point, while :meth:`apply` is the backward
// entry point called by the autograd engine during reverse-mode
// differentiation.  In practice nearly every op routes its forward
// through its CRTP base's typed static ``forward()`` (which handles
// dtype promotion, contiguity, dispatch and graph wiring) and uses
// :meth:`compute` only when invoked through a fully generic harness.
//
// Thread safety: instances are **not** thread-safe.  The framework
// constructs a fresh kernel per forward invocation and never shares
// it across threads.
//
// See Also
// --------
// :class:`UnaryKernel`, :class:`BinaryKernel`, :class:`NaryKernel`,
// :class:`VariadicKernel` — CRTP helpers that provide the typed
// implementations of :meth:`apply` and (when applicable) :meth:`compute`.
// :class:`KernelPolicy` — documentation of the compile-time flags
// concrete kernels expose to the CRTP layer.
class IKernel {
public:
    // Virtual destructor — ensures derived ``UnaryKernel`` /
    // ``BinaryKernel`` / ``ReduceKernel`` destructors run when held by a
    // base-class pointer.  Defaulted; nothing to clean up at this layer.
    virtual ~IKernel() = default;

    // Return the canonical short name of this op (e.g. ``"add"``, ``"relu"``).
    //
    // Returns
    // -------
    // std::string_view
    //     A stable, lifetime-bound view of the op's schema name.  Used
    //     by error messages, profiler scopes and schema lookups.
    //
    // Notes
    // -----
    // The returned view aliases ``Derived::schema_v1.name`` in CRTP
    // kernels, so it is valid for the lifetime of the program.
    virtual std::string_view name() const noexcept = 0;

    // Polymorphic forward entry point.
    //
    // Parameters
    // ----------
    // inputs : const std::vector<Storage>&
    //     Flat ordered list of input storages.  The interpretation
    //     (arity, shapes, dtype) is determined by the concrete kernel.
    //
    // Returns
    // -------
    // Storage
    //     The single output storage.
    //
    // Raises
    // ------
    // std::logic_error
    //     The default implementation always throws — concrete kernels
    //     are normally invoked via their typed static ``forward()``
    //     (which performs full dtype/shape/device handling).  Override
    //     this method only when a kernel must participate in a fully
    //     generic dispatch loop.
    virtual Storage compute(const std::vector<Storage>&) {
        throw std::logic_error(std::string(name()) +
                               ": compute() not implemented — use the concrete static forward()");
    }

    // Backward entry point — invoked by the autograd engine.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     The gradient of the loss with respect to this op's output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One gradient :class:`Storage` per forward input, in the same
    //     order as the inputs were consumed by ``forward()``.
    //
    // Notes
    // -----
    // CRTP bases override this method to delegate to
    // ``Derived::grad_formula(grad_out)`` and apply broadcast-reduction
    // back to the original input shapes; concrete ops only implement
    // the math.
    virtual std::vector<Storage> apply(Storage grad_out) = 0;
};

}  // namespace kernel
}  // namespace lucid
