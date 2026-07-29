// lucid/_C/autograd/CustomFunction.cpp
//
// Implements PythonBackwardNode::apply() and the pybind11 registration
// function register_custom_function().  Together they allow Python code to
// inject a custom backward computation into the C++ autograd graph.

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
#include "Helpers.h"

namespace py = pybind11;

namespace lucid {

namespace {

// Attempt to extract a Storage from a Python object.
//
// Two strategies are tried in order:
//   1. Direct cast to shared_ptr<TensorImpl>: the object is itself a
//      C++-backed TensorImpl exposed to Python.
//   2. Attribute-access cast via obj.impl: the object is a thin Python
//      wrapper (lucid.Tensor) that carries an impl shared_ptr.
// If both fail, returns an empty default-constructed CpuStorage{} so that
// None-valued gradient outputs (stop-gradient inputs) can be represented.
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

// Invoke the Python backward function and collect the resulting gradients.
//
// Steps:
//   1. Acquire the GIL; the Python interpreter may not be active on the
//      backward thread.
//   2. Wrap grad_out in a temporary TensorImpl so Python can treat it as a
//      normal tensor.  out_shape is used as the shape; if it is empty the
//      shape is inferred from nbytes / element size.
//   3. Call py_backward_fn(py_ctx, grad_tensor).  Any Python exception is
//      re-thrown as a C++ std::runtime_error.
//   4. Unpack the result: a tuple or list yields one Storage per item; a
//      single tensor yields one Storage.  Python None entries become empty
//      CpuStorage{} values representing "no gradient for this input".
std::vector<Storage> PythonBackwardNode::apply(Storage grad_out) {
    py::gil_scoped_acquire gil;

    if (!py_backward_fn || py_backward_fn.is_none()) {
        ErrorBuilder("PythonBackwardNode::apply").fail("backward function is not set");
    }

    Dtype grad_dt = storage_dtype(grad_out);
    Device grad_dev = storage_is_gpu(grad_out) ? Device::GPU : Device::CPU;

    // Reconstruct shape for the gradient TensorImpl.  A non-empty out_shape
    // is used directly; otherwise a flat 1-D shape is derived from byte count.
    Shape grad_shape = out_shape;
    if (grad_shape.empty()) {
        const std::size_t n = storage_nbytes(grad_out) / dtype_size(grad_dt);
        grad_shape = {static_cast<std::int64_t>(n)};
    }

    auto grad_impl =
        std::make_shared<TensorImpl>(std::move(grad_out), grad_shape, grad_dt, grad_dev, false);

    py::tuple grads(1);
    grads[0] = py::cast(grad_impl);
    return invoke_backward(grads);
}

void PythonBackwardNode::accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) {
    if (grad_slots_.size() < out_shapes.size())
        grad_slots_.resize(out_shapes.size());
    if (input_nr >= grad_slots_.size())
        return;

    auto& slot = grad_slots_[input_nr];
    if (!slot.has_value())
        slot = std::move(grad);
    else
        accumulate_into(*slot, grad);
}

std::vector<Storage> PythonBackwardNode::apply_barrier() {
    py::gil_scoped_acquire gil;

    if (!py_backward_fn || py_backward_fn.is_none()) {
        ErrorBuilder("PythonBackwardNode::apply_barrier").fail("backward function is not set");
    }

    const std::size_t n = out_shapes.size();
    grad_slots_.resize(n);

    py::tuple grads(n);
    for (std::size_t i = 0; i < n; ++i) {
        // An output the loss never used contributes nothing, but ``backward``
        // is still written against a fixed argument list, so give it zeros
        // rather than ``None`` and let the arithmetic take care of itself.
        Storage g = grad_slots_[i].has_value()
                        ? std::move(*grad_slots_[i])
                        : make_zero_storage(out_shapes[i], out_dtypes[i], out_devices[i]);
        grads[i] = py::cast(std::make_shared<TensorImpl>(std::move(g), out_shapes[i],
                                                         out_dtypes[i], out_devices[i], false));
    }
    grad_slots_.clear();

    return invoke_backward(grads);
}

std::vector<Storage> PythonBackwardNode::invoke_backward(const py::tuple& grads) {
    py::object result;
    try {
        result = py_backward_fn(py_ctx, *grads);
    } catch (py::error_already_set& e) {
        throw std::runtime_error(std::string("PythonBackward raised: ") + e.what());
    }

    std::vector<Storage> storages;

    // Helper that appends one Storage for a single Python return value.
    auto collect_one = [&](py::object item) {
        if (item.is_none()) {
            storages.push_back(Storage{CpuStorage{}});
        } else {
            storages.push_back(extract_storage(std::move(item)));
        }
    };

    // Unpack tuple/list returns (multiple inputs) or a single tensor return.
    if (py::isinstance<py::tuple>(result) || py::isinstance<py::list>(result)) {
        for (auto item : result)
            collect_one(item.cast<py::object>());
    } else {
        collect_one(result);
    }

    return storages;
}

// -------------------------------------------------------------------------
// pybind11 bindings
// -------------------------------------------------------------------------

void register_custom_function(py::module_& m) {
    // FunctionCtx — Python-visible as lucid._C.FunctionCtx.
    // save_for_backward accepts *args of tensors (TensorImpl or Python
    // wrappers with an .impl attribute).  saved_tensors returns a tuple.
    // Arbitrary attributes are stored/retrieved via __setattr__/__getattr__.
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
                        // Fall back to wrapper-with-.impl convention.
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

    // _PythonBackwardNode — internal type; Python sets ctx and backward_fn
    // fields, then passes the node to _register_python_backward_node.
    py::class_<PythonBackwardNode, Node, std::shared_ptr<PythonBackwardNode>>(m,
                                                                              "_PythonBackwardNode")
        .def(py::init<>())
        .def_readwrite("ctx", &PythonBackwardNode::py_ctx)
        .def_readwrite("backward_fn", &PythonBackwardNode::py_backward_fn);

    // _register_python_backward_node(output, node, inputs)
    //
    // Wire node into the autograd graph as the grad_fn of output.
    // For each input in inputs:
    //   - If the input is a leaf requiring grad, create an AccumulateGrad
    //     node for it (if not already present) and record an edge to it.
    //   - If the input does not require grad, record a null edge.
    // Saves the version counters for in-place mutation detection.
    // Marks output as a non-leaf tensor requiring gradients.
    m.def(
        "_register_python_backward_node",
        [](const std::vector<std::shared_ptr<TensorImpl>>& outputs,
           std::shared_ptr<PythonBackwardNode> node,
           const std::vector<std::shared_ptr<TensorImpl>>& inputs) {
            if (!node)
                ErrorBuilder("_register_python_backward_node").fail("null argument");
            if (outputs.empty())
                ErrorBuilder("_register_python_backward_node").fail("no outputs");
            for (const auto& out : outputs) {
                if (!out)
                    ErrorBuilder("_register_python_backward_node").fail("null argument");
            }

            // Single-output metadata is kept as-is so that path stays on the
            // engine's direct (non-barrier) route.  The vectors are what make
            // the node a barrier, and they are filled for every arity so the
            // barrier branch has complete information when it applies.
            node->out_shape = outputs.front()->shape();
            node->out_dtype = outputs.front()->dtype();
            node->out_device = outputs.front()->device();

            node->out_shapes.clear();
            node->out_dtypes.clear();
            node->out_devices.clear();
            if (outputs.size() > 1) {
                node->out_shapes.reserve(outputs.size());
                node->out_dtypes.reserve(outputs.size());
                node->out_devices.reserve(outputs.size());
                for (const auto& out : outputs) {
                    node->out_shapes.push_back(out->shape());
                    node->out_dtypes.push_back(out->dtype());
                    node->out_devices.push_back(out->device());
                }
            }

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
                    // Ensure a leaf has an AccumulateGrad node so that
                    // gradients flowing to it will be accumulated into .grad.
                    if (!inp->grad_fn())
                        inp->set_grad_fn(std::make_shared<AccumulateGrad>(inp));
                }
                edges.emplace_back(inp->grad_fn(), inp->grad_output_nr());
                versions.push_back(inp->version());
            }

            node->set_next_edges(std::move(edges));
            node->set_saved_versions(std::move(versions));

            // Every output shares the one node; ``grad_output_nr`` is what
            // tells the engine which slot an arriving gradient belongs to,
            // and it is the index the barrier reads back.
            for (std::size_t i = 0; i < outputs.size(); ++i) {
                outputs[i]->set_grad_fn(node);
                outputs[i]->set_grad_output_nr(static_cast<std::uint32_t>(i));
                outputs[i]->set_leaf(false);
                outputs[i]->set_requires_grad(true);
            }
        },
        py::arg("outputs"), py::arg("node"), py::arg("inputs"),
        "Wire a PythonBackwardNode into the autograd graph for `outputs`.");
}

}  // namespace lucid
