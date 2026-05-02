#pragma once

// =====================================================================
// Lucid C++ engine — CustomFunction (Phase 12)
// =====================================================================
//
// Enables Python code to define custom forward + backward functions that
// participate in the C++ autograd graph.  The pattern mirrors PyTorch's
// `torch.autograd.Function`:
//
//   class MyOp(lucid.autograd.Function):
//       @staticmethod
//       def forward(ctx, x):
//           ctx.save_for_backward(x)
//           return x * 2
//
//       @staticmethod
//       def backward(ctx, grad):
//           (x,) = ctx.saved_tensors
//           return grad * 2
//
//   y = MyOp.apply(x)
//   y.backward()
//
// On the C++ side, when `apply()` runs it:
//   1. Calls `forward()` (Python), which populates a `FunctionCtx`.
//   2. Creates a `PythonBackwardNode` that holds the ctx + backward callable.
//   3. Wires the backward node into the autograd graph.
//
// Layer: autograd/. Depends on autograd/Node.h, core/, pybind11.

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

// ---------------------------------------------------------------------------
// FunctionCtx — the Python-visible context object passed to forward/backward.
// ---------------------------------------------------------------------------
//
// Mirrors `torch.autograd.function.FunctionCtx`: holds saved tensors and any
// extra data the user stashes during forward for use in backward.
//
class LUCID_API FunctionCtx {
public:
    FunctionCtx() = default;

    /// Save tensors needed for backward.  Mirrors ctx.save_for_backward().
    void save_for_backward(std::vector<std::shared_ptr<TensorImpl>> tensors) {
        saved_tensors_ = std::move(tensors);
    }

    /// Retrieve saved tensors during backward.  Mirrors ctx.saved_tensors.
    const std::vector<std::shared_ptr<TensorImpl>>& saved_tensors() const {
        return saved_tensors_;
    }

    /// Store an arbitrary Python object (e.g. a scalar, a shape).
    void store(const std::string& key, py::object val) {
        extras_[key] = std::move(val);
    }
    py::object load(const std::string& key) const {
        auto it = extras_.find(key);
        return (it != extras_.end()) ? it->second : py::none();
    }

private:
    std::vector<std::shared_ptr<TensorImpl>> saved_tensors_;
    std::unordered_map<std::string, py::object> extras_;
};

// ---------------------------------------------------------------------------
// PythonBackwardNode — autograd Node backed by a Python backward callable.
// ---------------------------------------------------------------------------

class LUCID_API PythonBackwardNode : public Node {
public:
    /// Python FunctionCtx object populated during forward.
    py::object py_ctx;

    /// Python static method `backward(ctx, grad_output) → Tensor or tuple`.
    py::object py_backward_fn;

    /// Shape, dtype, device of the forward output — needed to reconstruct
    /// the gradient TensorImpl that is passed to Python backward.
    Shape  out_shape;
    Dtype  out_dtype  = Dtype::F32;
    Device out_device = Device::CPU;

    std::string_view name() const noexcept { return "PythonBackward"; }

    /// Run the Python backward and convert the result to a Storage vector.
    std::vector<Storage> apply(Storage grad_out) override;
};

// ---------------------------------------------------------------------------
// Registration helper (called from bind_autograd.cpp)
// ---------------------------------------------------------------------------

void register_custom_function(py::module_& m);

}  // namespace lucid
