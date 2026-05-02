#pragma once

// =====================================================================
// Lucid C++ engine — BindingGen: schema-driven binding helpers (Phase 6).
// =====================================================================
//
// Each helper extracts the op name from schema_v1 so the binding name is
// always the canonical name and never drifts from the schema.
//
//   bind_unary<NegBackward>(m, &neg_op);
//   bind_binary<AddBackward>(m, &add_op, "Element-wise a + b.");
//   bind_unary_extra<LeakyReluBackward>(m, &leaky_relu_op,
//       py::arg("a"), py::arg("slope") = 0.01);
//
// Complex ops (custom arities, multiple return values) still use m.def
// directly — these helpers cover the ~60% of ops that are truly generic.
//
// Layer: bindings/. Depends on pybind11 and core/OpSchema.h.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../core/fwd.h"

namespace py = pybind11;

namespace lucid::bindings {

// ---- Unary: f(tensor) → tensor -----------------------------------------

template <class BackwardNode>
void bind_unary(py::module_& m, TensorImplPtr (*fn)(const TensorImplPtr&), const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"), doc);
}

// ---- Binary: f(tensor, tensor) → tensor ---------------------------------

template <class BackwardNode>
void bind_binary(py::module_& m,
                 TensorImplPtr (*fn)(const TensorImplPtr&, const TensorImplPtr&),
                 const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"), py::arg("b"), doc);
}

// ---- Unary + extra scalar args ------------------------------------------
// For ops like leaky_relu(a, slope), elu(a, alpha): pass extra py::arg specs
// as variadic args after fn.

template <class BackwardNode, class Fn, class... PyArgs>
void bind_unary_extra(py::module_& m, Fn fn, PyArgs&&... extra_args) {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"),
          std::forward<PyArgs>(extra_args)...);
}

// ---- Reduction: f(tensor, axes, keepdims) → tensor ----------------------

template <class BackwardNode>
void bind_reduce(py::module_& m,
                 TensorImplPtr (*fn)(const TensorImplPtr&, const std::vector<int>&, bool),
                 const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"),
          py::arg("axes") = std::vector<int>{}, py::arg("keepdims") = false, doc);
}

}  // namespace lucid::bindings
