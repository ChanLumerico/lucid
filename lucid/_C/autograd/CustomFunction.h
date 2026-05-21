// lucid/_C/autograd/CustomFunction.h
//
// Bridges the C++ autograd engine to user-defined Python backward functions
// written against the lucid.autograd.Function API.  Provides two classes and
// one registration helper:
//
//   FunctionCtx  — the context object passed to both the user's forward and
//                  backward methods; stores saved tensors and arbitrary Python
//                  key/value pairs.
//
//   PythonBackwardNode — a Node subclass that wraps a Python callable and
//                  invokes it (with the GIL held) when the Engine calls
//                  apply() during a backward pass.
//
//   register_custom_function() — registers both types and the
//                  _register_python_backward_node helper into a pybind11
//                  module so that Python code can wire custom ops into the
//                  autograd graph.

#pragma once

#include <pybind11/pybind11.h>

#include <memory>
#include <string_view>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "Node.h"

namespace py = pybind11;

namespace lucid {

// Per-call context shared between a custom op's forward and backward passes.
//
// ``FunctionCtx`` is the C++-side counterpart of :class:`lucid.autograd.Function`'s
// Python ``ctx`` object.  A fresh instance is created on each ``Function.apply``
// invocation and is passed unchanged as the first argument to both
// ``forward(ctx, ...)`` and ``backward(ctx, grad_output)``, providing the only
// legal channel for state to cross the forward/backward boundary.
//
// Attributes
// ----------
// saved_tensors_ : std::vector<std::shared_ptr<TensorImpl>>
//     Tensors stored via :func:`save_for_backward` as strong references so
//     they survive until the backward pass executes.
// extras_ : std::unordered_map<std::string, py::object>
//     Generic Python key/value bag mirroring ``ctx.<name> = value`` /
//     ``ctx.<name>`` access from Python.  Used for scalars, shape tuples,
//     hyperparameters, and other non-tensor state.
//
// Notes
// -----
// The object is heap-allocated and jointly owned by the Python wrapper and
// the associated :class:`PythonBackwardNode`, so it outlives the forward
// call and remains valid for the entire backward traversal.  Capturing
// tensors through Python closures instead of saving them through ``ctx``
// bypasses this ownership tracking and leaks memory.
//
// See Also
// --------
// :class:`lucid.autograd.FunctionCtx` — Python-facing wrapper class.
// :class:`PythonBackwardNode` — owns one ``FunctionCtx`` per custom op call.
class LUCID_API FunctionCtx {
public:
    FunctionCtx() = default;

    // Save tensors needed for the backward pass.
    //
    // Stores strong ``shared_ptr`` references so that the underlying
    // :class:`TensorImpl` buffers remain alive until backward completes,
    // even if the Python-side handle goes out of scope.  Mirrors the
    // semantics of :meth:`lucid.autograd.FunctionCtx.save_for_backward`.
    //
    // Parameters
    // ----------
    // tensors : std::vector<std::shared_ptr<TensorImpl>>
    //     Tensors to save, in the order ``saved_tensors()`` should return
    //     them.  Moved into the context.
    //
    // Notes
    // -----
    // Each call overwrites the previously saved list; saving twice does not
    // accumulate.  To stash non-tensor data (shapes, axes, scalars) use the
    // :func:`store` / :func:`load` extras bag instead.
    void save_for_backward(std::vector<std::shared_ptr<TensorImpl>> tensors) {
        saved_tensors_ = std::move(tensors);
    }

    // Read back the tensors persisted by :func:`save_for_backward`.
    //
    // Returns
    // -------
    // const std::vector<std::shared_ptr<TensorImpl>>&
    //     The saved tensors in registration order.  Empty when nothing was
    //     saved.
    const std::vector<std::shared_ptr<TensorImpl>>& saved_tensors() const { return saved_tensors_; }

    // Store an arbitrary Python object under a string key.
    //
    // Used by the pybind11 ``__setattr__`` binding so that Python code
    // assigning ``ctx.shape = x.shape`` lands here.
    //
    // Parameters
    // ----------
    // key : const std::string&
    //     Attribute name.
    // val : py::object
    //     Value to store; ownership is transferred into the extras map.
    void store(const std::string& key, py::object val) { extras_[key] = std::move(val); }

    // Retrieve a previously stored Python object.
    //
    // Parameters
    // ----------
    // key : const std::string&
    //     Attribute name previously passed to :func:`store`.
    //
    // Returns
    // -------
    // py::object
    //     The stored object, or ``py::none()`` when ``key`` is absent.  The
    //     ``__getattr__`` binding translates the ``None`` sentinel into a
    //     Python ``AttributeError``.
    py::object load(const std::string& key) const {
        auto it = extras_.find(key);
        return (it != extras_.end()) ? it->second : py::none();
    }

private:
    std::vector<std::shared_ptr<TensorImpl>> saved_tensors_;
    std::unordered_map<std::string, py::object> extras_;
};

// Autograd graph node that defers backward computation to a Python callable.
//
// When a user subclasses :class:`lucid.autograd.Function` and invokes
// ``MySubclass.apply(...)``, the Python layer constructs a
// ``PythonBackwardNode``, populates :attr:`py_ctx` and :attr:`py_backward_fn`,
// and registers the node as the output tensor's ``grad_fn`` via
// ``_register_python_backward_node``.  The Engine then treats this node like
// any other backward node — when :func:`apply` is called during graph
// traversal, control crosses back into Python with the GIL held.
//
// Attributes
// ----------
// py_ctx : py::object
//     The :class:`FunctionCtx` wrapper carrying saved tensors and extras.
//     Forwarded unchanged as the first argument to ``py_backward_fn``.
// py_backward_fn : py::object
//     The user's ``backward(ctx, grad_output)`` static method.
//     Expected signature: ``(ctx, grad_output: Tensor) -> Tensor |
//     tuple[Tensor | None, ...]``.
// out_shape : Shape
//     Shape of the forward output; used to reconstruct a proper
//     :class:`TensorImpl` wrapper around the raw ``grad_out`` storage.
// out_dtype : Dtype
//     Dtype of the forward output (default ``F32``).
// out_device : Device
//     Device of the forward output (default ``CPU``).
//
// Notes
// -----
// The public ``py_ctx`` / ``py_backward_fn`` data members are exposed
// directly via ``def_readwrite`` so the pybind11 binding can set them
// without accessor boilerplate.
//
// Thread safety: none.  The Engine guarantees single-threaded execution of
// the backward pass, so :func:`apply` does not lock; calling it concurrently
// from multiple threads is undefined.
//
// See Also
// --------
// :class:`lucid.autograd.Function` — Python base class users subclass.
// :class:`FunctionCtx` — context object carrying saved state.
class LUCID_API PythonBackwardNode : public Node {
public:
    // The FunctionCtx object forwarded to py_backward_fn as its first argument.
    py::object py_ctx;

    // The Python callable implementing the backward pass.
    // Signature: (ctx: FunctionCtx, grad_output: Tensor) -> Tensor | tuple[Tensor]
    py::object py_backward_fn;

    // Metadata of the forward output tensor — used to reconstruct a
    // TensorImpl wrapper around the incoming gradient Storage.
    Shape out_shape;
    Dtype out_dtype = Dtype::F32;
    Device out_device = Device::CPU;

    // Human-readable node name for debugger / profiler output.
    //
    // Returns
    // -------
    // std::string_view
    //     The literal ``"PythonBackward"``.
    std::string_view name() const noexcept { return "PythonBackward"; }

    // Invoke the Python backward callable and collect the resulting gradients.
    //
    // Acquires the GIL, wraps ``grad_out`` in a temporary :class:`TensorImpl`
    // matching :attr:`out_shape` / :attr:`out_dtype` / :attr:`out_device`,
    // calls ``py_backward_fn(py_ctx, grad_tensor)``, then unpacks the
    // returned tuple/list/tensor back into a vector of :class:`Storage`
    // values — one per forward input.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient delivered by the Engine; assumed to match the
    //     output dtype/device of the original forward call.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One :class:`Storage` per input edge.  Python ``None`` entries
    //     become empty ``CpuStorage{}`` values representing "no gradient
    //     for this input".
    //
    // Raises
    // ------
    // std::runtime_error
    //     The Python ``backward_fn`` is unset, or it raised an exception
    //     during execution (the original message is preserved).
    std::vector<Storage> apply(Storage grad_out) override;
};

// Register the custom-function bindings into a pybind11 module.
//
// Exposes three Python-visible names under the receiving module:
//
//   - ``FunctionCtx`` — opaque shared-pointer class with
//     ``save_for_backward(*tensors)``, the ``saved_tensors`` property,
//     and permissive ``__setattr__`` / ``__getattr__`` for extras.
//   - ``_PythonBackwardNode`` — internal :class:`Node` subclass with
//     read/write ``ctx`` and ``backward_fn`` fields.
//   - ``_register_python_backward_node(output, node, inputs)`` — wires
//     ``node`` into the autograd graph as ``output``'s ``grad_fn``,
//     materialising :class:`AccumulateGrad` nodes for any leaves that
//     need them and recording version counters for in-place mutation
//     checks.
//
// Parameters
// ----------
// m : py::module_&
//     The pybind11 module to register into.  Called once during engine
//     module initialisation.
//
// See Also
// --------
// :mod:`lucid.autograd._python_node` — Python-side caller of
// ``_register_python_backward_node``.
void register_custom_function(py::module_& m);

}  // namespace lucid
