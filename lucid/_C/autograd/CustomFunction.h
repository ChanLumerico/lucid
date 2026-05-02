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

class LUCID_API FunctionCtx {
public:
    FunctionCtx() = default;

    void save_for_backward(std::vector<std::shared_ptr<TensorImpl>> tensors) {
        saved_tensors_ = std::move(tensors);
    }

    const std::vector<std::shared_ptr<TensorImpl>>& saved_tensors() const { return saved_tensors_; }

    void store(const std::string& key, py::object val) { extras_[key] = std::move(val); }
    py::object load(const std::string& key) const {
        auto it = extras_.find(key);
        return (it != extras_.end()) ? it->second : py::none();
    }

private:
    std::vector<std::shared_ptr<TensorImpl>> saved_tensors_;
    std::unordered_map<std::string, py::object> extras_;
};

class LUCID_API PythonBackwardNode : public Node {
public:
    py::object py_ctx;

    py::object py_backward_fn;

    Shape out_shape;
    Dtype out_dtype = Dtype::F32;
    Device out_device = Device::CPU;

    std::string_view name() const noexcept { return "PythonBackward"; }

    std::vector<Storage> apply(Storage grad_out) override;
};

void register_custom_function(py::module_& m);

}  // namespace lucid
