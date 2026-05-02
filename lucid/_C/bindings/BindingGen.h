#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../core/fwd.h"

namespace py = pybind11;

namespace lucid::bindings {

template <class BackwardNode>
void bind_unary(py::module_& m, TensorImplPtr (*fn)(const TensorImplPtr&), const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"), doc);
}

template <class BackwardNode>
void bind_binary(py::module_& m,
                 TensorImplPtr (*fn)(const TensorImplPtr&, const TensorImplPtr&),
                 const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"), py::arg("b"), doc);
}

template <class BackwardNode, class Fn, class... PyArgs>
void bind_unary_extra(py::module_& m, Fn fn, PyArgs&&... extra_args) {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"),
          std::forward<PyArgs>(extra_args)...);
}

template <class BackwardNode>
void bind_reduce(py::module_& m,
                 TensorImplPtr (*fn)(const TensorImplPtr&, const std::vector<int>&, bool),
                 const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"),
          py::arg("axes") = std::vector<int>{}, py::arg("keepdims") = false, doc);
}

}  // namespace lucid::bindings
