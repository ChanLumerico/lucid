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

// Context object shared between a custom op's forward and backward passes.
//
// Python code accesses this object as `ctx` inside both `forward()` and
// `backward()`.  It supports two storage mechanisms:
//   - save_for_backward / saved_tensors: the canonical path for saving
//     TensorImpl-backed tensors that the backward formula needs.  These are
//     stored as strong shared_ptrs, keeping the tensors alive until the
//     context is released.
//   - store / load (__setattr__ / __getattr__ from Python): generic key/value
//     store for scalars, shapes, or other Python objects that are not tensors.
//
// Ownership: FunctionCtx is heap-allocated and jointly owned by the Python
// wrapper object and the PythonBackwardNode.  It outlives the forward call so
// that the backward pass can read from it.
class LUCID_API FunctionCtx {
public:
    FunctionCtx() = default;

    // Save a list of TensorImpl shared_ptrs for retrieval during backward.
    // Overwrites any previously saved tensors.
    void save_for_backward(std::vector<std::shared_ptr<TensorImpl>> tensors) {
        saved_tensors_ = std::move(tensors);
    }

    // Access the tensors saved by save_for_backward().
    const std::vector<std::shared_ptr<TensorImpl>>& saved_tensors() const { return saved_tensors_; }

    // Store an arbitrary Python object under key (used by __setattr__).
    void store(const std::string& key, py::object val) { extras_[key] = std::move(val); }

    // Retrieve a Python object by key (used by __getattr__).
    // Returns py::none() when the key is absent; callers (the __getattr__
    // binding) translate that into a Python AttributeError.
    py::object load(const std::string& key) const {
        auto it = extras_.find(key);
        return (it != extras_.end()) ? it->second : py::none();
    }

private:
    std::vector<std::shared_ptr<TensorImpl>> saved_tensors_;
    std::unordered_map<std::string, py::object> extras_;
};

// A Node that invokes a Python callable during backward.
//
// When a user subclasses lucid.autograd.Function and calls .apply(), the
// Python layer creates a PythonBackwardNode, sets py_backward_fn to the
// user's backward() static method, and registers it as the output's grad_fn
// via _register_python_backward_node.
//
// apply() acquires the GIL, wraps the incoming gradient Storage in a
// temporary TensorImpl, calls py_backward_fn(py_ctx, grad_tensor), then
// unpacks the returned tensor(s) back into Storage values.
//
// Fields py_ctx and py_backward_fn are public so that the pybind11 binding
// can set them via .def_readwrite without extra accessors.
//
// Thread safety: none.  apply() must not be called from multiple threads
// simultaneously; the Engine guarantees single-threaded execution.
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

    // Constant name for debugging and error messages.
    std::string_view name() const noexcept { return "PythonBackward"; }

    // Wrap grad_out in a temporary TensorImpl, invoke py_backward_fn, and
    // unpack the result into a vector of Storage values (one per input).
    // None values in the Python result become empty CpuStorage{} entries.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Register FunctionCtx, PythonBackwardNode, and _register_python_backward_node
// into the given pybind11 module.  Called once during module initialisation.
void register_custom_function(py::module_& m);

}  // namespace lucid
