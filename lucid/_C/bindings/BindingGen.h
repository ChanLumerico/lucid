// lucid/_C/bindings/BindingGen.h
//
// Lightweight code-generation helpers used by bind_bfunc.cpp, bind_ufunc.cpp,
// and bind_utils.cpp to register ops whose Python name and argument signature
// are derived directly from the corresponding backward node's OpSchema.  Each
// helper reads `BackwardNode::schema_v1.name` at compile time so the binding
// name is always in sync with the op registry — no string literals to keep
// consistent manually.
//
// All four templates follow the same pattern:
//   1. Extract the Python-visible op name from BackwardNode::schema_v1.name
//      (a compile-time string_view into the static OpSchema).
//   2. Call m.def() with the appropriate argument descriptors.
//   3. Forward any extra pybind11 argument objects unchanged (bind_unary_extra).
//
// Using these helpers instead of raw m.def() calls guarantees that the string
// registered with pybind11 matches the string stored in the OpRegistry, making
// op-schema introspection, .pyi stub generation, and Python-side dispatch
// consistent by construction.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../core/fwd.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers a standard unary op (single TensorImplPtr → TensorImplPtr) under
// the name stored in BackwardNode::schema_v1.  The single argument is exposed
// to Python as "a".
//
// Usage: bind_unary<ExpBackward>(m, &exp_op, "Element-wise exp(a).");
template <class BackwardNode>
void bind_unary(py::module_& m, TensorImplPtr (*fn)(const TensorImplPtr&), const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"), doc);
}

// Registers a standard binary op (two TensorImplPtrs → TensorImplPtr) under
// the name stored in BackwardNode::schema_v1.  Arguments are "a" and "b".
//
// Usage: bind_binary<AddBackward>(m, &add_op, "Element-wise a + b.");
template <class BackwardNode>
void bind_binary(py::module_& m,
                 TensorImplPtr (*fn)(const TensorImplPtr&, const TensorImplPtr&),
                 const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"), py::arg("b"), doc);
}

// Registers a unary op that takes extra keyword arguments beyond the mandatory
// "a" tensor — e.g. leaky_relu(slope=0.01) or elu(alpha=1.0).  Extra pybind11
// argument descriptors are forwarded via the variadic PyArgs pack.
//
// The Fn template parameter is left generic (not constrained to function pointers)
// so that lambdas wrapping the C++ op can also be passed through.
//
// Usage: bind_unary_extra<LeakyReluBackward>(m, &leaky_relu_op,
//                                            py::arg("slope") = 0.01);
template <class BackwardNode, class Fn, class... PyArgs>
void bind_unary_extra(py::module_& m, Fn fn, PyArgs&&... extra_args) {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"),
          std::forward<PyArgs>(extra_args)...);
}

// Registers a reduction op with the standard signature
// (a, dim=[], keepdims=False) → TensorImplPtr, using the name from
// BackwardNode::schema_v1.  The empty `dim` default means "reduce all dims".
//
// The `dim` argument is typed std::vector<int> (not int64_t) because the C++
// reduction ops use plain `int` indices internally; pybind11 converts the
// Python list automatically.  Singular naming matches the standard tensor
// framework convention (``dim`` accepts an int or a list of ints).
//
// Usage: bind_reduce<SumBackward>(m, &sum_op, "Reduce-sum along dim.");
template <class BackwardNode>
void bind_reduce(py::module_& m,
                 TensorImplPtr (*fn)(const TensorImplPtr&, const std::vector<int>&, bool),
                 const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"),
          py::arg("dim") = std::vector<int>{}, py::arg("keepdims") = false, doc);
}

}  // namespace lucid::bindings
