// lucid/_C/bindings/BindingGen.h
//
// Compile-time code-generation helpers that derive pybind11 ``m.def``
// calls from the corresponding backward node's :struct:`OpSchema`.
//
// Used by :file:`bind_bfunc.cpp`, :file:`bind_ufunc.cpp`, and
// :file:`bind_utils.cpp` to register ops whose Python-visible name and
// argument signature come directly from
// ``BackwardNode::schema_v1.name`` — a compile-time ``string_view``
// into the static op registry.  Because the name string is read from
// the same source the ``OpRegistry`` consumes, ``.pyi`` stub
// generation, Python-side dispatch, and op-schema introspection stay
// consistent by construction; there is no second source of truth to
// keep aligned with hand-edited string literals.
//
// Notes
// -----
// All four templates share a common shape:
//
//   1. Extract the Python-visible op name from
//      ``BackwardNode::schema_v1.name``.
//   2. Call ``m.def()`` with the appropriate
//      :func:`py::arg` descriptors.
//   3. Forward any extra pybind11 argument objects unchanged
//      (:func:`bind_unary_extra`).
//
// Argument naming convention matches the standard tensor framework:
// the first tensor operand is ``"a"``, the second (when present)
// ``"b"``, reductions take ``"dim"`` and ``"keepdims"``.
//
// See Also
// --------
// :file:`lucid/_C/core/OpSchema.h` — the schema descriptor read here.
// :file:`lucid/_C/core/OpRegistry.h` — the registry the same name is
//     stored in.

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

// Registers a standard unary op (``TensorImplPtr → TensorImplPtr``)
// on a pybind11 module under the name carried by the corresponding
// backward node.
//
// Reads ``BackwardNode::schema_v1.name`` at compile time and emits a
// ``m.def(name, fn, py::arg("a"), doc)`` call.  The single tensor
// argument is always exposed to Python as ``"a"`` to match the
// reference framework's free-function convention.
//
// Parameters
// ----------
// m : py::module_&
//     Target pybind11 module — typically the engine's bound submodule.
// fn : TensorImplPtr (*)(const TensorImplPtr&)
//     C++ function pointer to the op's forward implementation.
// doc : const char*, optional
//     Python docstring to attach to the binding.  Defaults to the
//     empty string.
//
// Examples
// --------
// Binding an element-wise exponent::
//
//     bind_unary<ExpBackward>(m, &exp_op, "Element-wise exp(a).");
//
// See Also
// --------
// :func:`bind_unary_extra` — variant accepting extra kwargs.
// :func:`bind_binary` — two-argument variant.
template <class BackwardNode>
void bind_unary(py::module_& m, TensorImplPtr (*fn)(const TensorImplPtr&), const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"), doc);
}

// Registers a standard binary op (two TensorImplPtrs → TensorImplPtr) under
// the name stored in BackwardNode::schema_v1.  Arguments are "a" and "b".
//
// Usage: bind_binary<AddBackward>(m, &add_op, "Element-wise a + b.");

// Registers a standard binary op (two ``TensorImplPtr`` operands →
// ``TensorImplPtr``) on a pybind11 module under the name carried by
// the corresponding backward node.
//
// Reads ``BackwardNode::schema_v1.name`` at compile time and emits a
// ``m.def(name, fn, py::arg("a"), py::arg("b"), doc)`` call.  The
// two tensor arguments are exposed as ``"a"`` and ``"b"`` to match
// the reference framework's free-function convention.
//
// Parameters
// ----------
// m : py::module_&
//     Target pybind11 module.
// fn : TensorImplPtr (*)(const TensorImplPtr&, const TensorImplPtr&)
//     C++ function pointer to the op's forward implementation.
// doc : const char*, optional
//     Python docstring to attach to the binding.
//
// Examples
// --------
// Binding element-wise addition::
//
//     bind_binary<AddBackward>(m, &add_op, "Element-wise a + b.");
//
// See Also
// --------
// :func:`bind_unary` — single-argument variant.
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

// Registers a unary op that accepts extra keyword arguments beyond
// the mandatory ``"a"`` tensor operand.
//
// Variadic in the extra-argument pack so that arbitrary
// :func:`py::arg`-style descriptors with default values can be
// forwarded unchanged.  Typical use cases are activation functions
// with hyperparameters (``leaky_relu(slope=0.01)``,
// ``elu(alpha=1.0)``).  The callable template parameter ``Fn`` is
// intentionally unconstrained so that both plain function pointers
// and lambdas wrapping the C++ op can be passed.
//
// Parameters
// ----------
// m : py::module_&
//     Target pybind11 module.
// fn : Fn
//     Callable implementing the op.  Function pointer or lambda; the
//     argument arity must match ``"a"`` plus the forwarded extra
//     descriptors.
// extra_args : PyArgs&&...
//     Zero or more pybind11 argument descriptors
//     (:func:`py::arg("name") = default`) appended after the implicit
//     ``py::arg("a")``.
//
// Examples
// --------
// Binding leaky ReLU with a slope hyperparameter::
//
//     bind_unary_extra<LeakyReluBackward>(m, &leaky_relu_op,
//                                         py::arg("slope") = 0.01);
//
// See Also
// --------
// :func:`bind_unary` — without extra kwargs.
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

// Registers a reduction op with the canonical
// ``(a, dim=[], keepdims=False) → TensorImplPtr`` signature.
//
// Reads ``BackwardNode::schema_v1.name`` at compile time and emits an
// ``m.def`` with the three argument descriptors: ``"a"`` (the tensor),
// ``"dim"`` (defaulting to an empty list meaning "reduce all
// dimensions"), and ``"keepdims"`` (defaulting to ``false``).
//
// Parameters
// ----------
// m : py::module_&
//     Target pybind11 module.
// fn : TensorImplPtr (*)(const TensorImplPtr&, const std::vector<int>&, bool)
//     C++ function pointer to the reduction op.  Note ``int`` rather
//     than ``int64_t``: the engine's reduction kernels use plain
//     ``int`` for axis indices; pybind11 converts the Python list
//     automatically.
// doc : const char*, optional
//     Python docstring to attach to the binding.
//
// Notes
// -----
// Singular ``"dim"`` (rather than plural ``"dims"``) matches the
// reference framework convention — the argument accepts either a
// single ``int`` or a list of ``int``\ s.
//
// Examples
// --------
// Binding a reduce-sum op::
//
//     bind_reduce<SumBackward>(m, &sum_op, "Reduce-sum along dim.");
//
// See Also
// --------
// :func:`bind_unary` — non-reducing unary ops.
template <class BackwardNode>
void bind_reduce(py::module_& m,
                 TensorImplPtr (*fn)(const TensorImplPtr&, const std::vector<int>&, bool),
                 const char* doc = "") {
    m.def(BackwardNode::schema_v1.name.data(), fn, py::arg("a"),
          py::arg("dim") = std::vector<int>{}, py::arg("keepdims") = false, doc);
}

}  // namespace lucid::bindings
