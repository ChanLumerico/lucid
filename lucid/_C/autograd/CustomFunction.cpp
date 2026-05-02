#include "CustomFunction.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"

namespace py = pybind11;

namespace lucid {

namespace {

Storage extract_storage(py::object obj) {
    try {
        auto t = obj.cast<std::shared_ptr<TensorImpl>>();
        if (t)
            return t->storage();
    } catch (...) {
    }

    try {
        auto impl = obj.attr("impl").cast<std::shared_ptr<TensorImpl>>();
        if (impl)
            return impl->storage();
    } catch (...) {
    }

    return Storage{CpuStorage{}};
}

}  // namespace

std::vector<Storage> PythonBackwardNode::apply(Storage grad_out) {
    py::gil_scoped_acquire gil;

    if (!py_backward_fn || py_backward_fn.is_none()) {
        ErrorBuilder("PythonBackwardNode::apply").fail("backward function is not set");
    }

    Dtype grad_dt = storage_dtype(grad_out);
    Device grad_dev = storage_is_gpu(grad_out) ? Device::GPU : Device::CPU;

    Shape grad_shape = out_shape;
    if (grad_shape.empty()) {
        const std::size_t n = storage_nbytes(grad_out) / dtype_size(grad_dt);
        grad_shape = {static_cast<std::int64_t>(n)};
    }

    auto grad_impl =
        std::make_shared<TensorImpl>(std::move(grad_out), grad_shape, grad_dt, grad_dev, false);

    py::object result;
    try {
        result = py_backward_fn(py_ctx, py::cast(grad_impl));
    } catch (py::error_already_set& e) {
        throw std::runtime_error(std::string("PythonBackward raised: ") + e.what());
    }

    std::vector<Storage> storages;

    auto collect_one = [&](py::object item) {
        if (item.is_none()) {
            storages.push_back(Storage{CpuStorage{}});
        } else {
            storages.push_back(extract_storage(std::move(item)));
        }
    };

    if (py::isinstance<py::tuple>(result) || py::isinstance<py::list>(result)) {
        for (auto item : result)
            collect_one(item.cast<py::object>());
    } else {
        collect_one(result);
    }

    return storages;
}

void register_custom_function(py::module_& m) {
    py::class_<FunctionCtx, std::shared_ptr<FunctionCtx>>(m, "FunctionCtx")
        .def(py::init<>())
        .def(
            "save_for_backward",
            [](FunctionCtx& ctx, py::args tensors) {
                std::vector<std::shared_ptr<TensorImpl>> v;
                v.reserve(tensors.size());
                for (auto t : tensors) {
                    try {
                        v.push_back(t.cast<std::shared_ptr<TensorImpl>>());
                    } catch (...) {
                        v.push_back(t.attr("impl").cast<std::shared_ptr<TensorImpl>>());
                    }
                }
                ctx.save_for_backward(std::move(v));
            },
            "Save tensors for backward.  Mirrors ctx.save_for_backward().")
        .def_property_readonly(
            "saved_tensors",
            [](const FunctionCtx& ctx) -> py::tuple {
                py::tuple t(ctx.saved_tensors().size());
                for (std::size_t i = 0; i < ctx.saved_tensors().size(); ++i)
                    t[i] = py::cast(ctx.saved_tensors()[i]);
                return t;
            },
            "Tuple of tensors saved during forward.")
        .def("__setattr__", [](FunctionCtx& ctx, const std::string& key,
                               py::object val) { ctx.store(key, std::move(val)); })
        .def("__getattr__", [](const FunctionCtx& ctx, const std::string& key) -> py::object {
            auto v = ctx.load(key);
            if (v.is_none())
                throw py::attribute_error(key);
            return v;
        });

    py::class_<PythonBackwardNode, Node, std::shared_ptr<PythonBackwardNode>>(m,
                                                                              "_PythonBackwardNode")
        .def(py::init<>())
        .def_readwrite("ctx", &PythonBackwardNode::py_ctx)
        .def_readwrite("backward_fn", &PythonBackwardNode::py_backward_fn);

    m.def(
        "_register_python_backward_node",
        [](std::shared_ptr<TensorImpl> output, std::shared_ptr<PythonBackwardNode> node,
           const std::vector<std::shared_ptr<TensorImpl>>& inputs) {
            if (!output || !node)
                ErrorBuilder("_register_python_backward_node").fail("null argument");

            node->out_shape = output->shape();
            node->out_dtype = output->dtype();
            node->out_device = output->device();

            std::vector<Edge> edges;
            edges.reserve(inputs.size());
            std::vector<std::int64_t> versions;
            versions.reserve(inputs.size());

            for (const auto& inp : inputs) {
                if (!inp || !inp->requires_grad()) {
                    edges.emplace_back(nullptr, 0);
                    versions.push_back(0);
                    continue;
                }
                if (inp->is_leaf()) {
                    if (!inp->grad_fn())
                        inp->set_grad_fn(std::make_shared<AccumulateGrad>(inp));
                }
                edges.emplace_back(inp->grad_fn(), 0);
                versions.push_back(inp->version());
            }

            node->set_next_edges(std::move(edges));
            node->set_saved_versions(std::move(versions));

            output->set_grad_fn(std::move(node));
            output->set_leaf(false);
            output->set_requires_grad(true);
        },
        py::arg("output"), py::arg("node"), py::arg("inputs"),
        "Wire a PythonBackwardNode into the autograd graph for `output`.");
}

}  // namespace lucid
