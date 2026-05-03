// lucid/_C/bindings/bind_linalg.cpp
//
// Registers dense linear-algebra ops on the `lucid._C.engine.linalg`
// sub-module (created in bind.cpp).  All ops dispatch through the backend
// Dispatcher: CPU uses LAPACK (Accelerate) and GPU uses MLX linalg.
//
// Ops that return multiple tensors (qr → (Q, R), eig/eigh → (values, vectors))
// are wrapped in lambdas that unpack the C++ std::vector result into Python
// tuples, since pybind11 cannot auto-convert a fixed-size vector to tuple.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/linalg/Cholesky.h"
#include "../ops/linalg/Det.h"
#include "../ops/linalg/Eig.h"
#include "../ops/linalg/Eigh.h"
#include "../ops/linalg/Inv.h"
#include "../ops/linalg/MatrixPower.h"
#include "../ops/linalg/Norm.h"
#include "../ops/linalg/Pinv.h"
#include "../ops/linalg/QR.h"
#include "../ops/linalg/SVD.h"
#include "../ops/linalg/Solve.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers all linalg ops on the `linalg` sub-module.
void register_linalg(py::module_& m) {
    // Single-output ops delegate directly to their C++ op functions.
    m.def("inv", &inv_op, py::arg("a"));
    m.def("det", &det_op, py::arg("a"));
    m.def("solve", &solve_op, py::arg("a"), py::arg("b"));
    m.def("cholesky", &cholesky_op, py::arg("a"), py::arg("upper") = false);

    // norm accepts an axis list; empty list means full-tensor norm.
    // std::move transfers the locally-constructed vector into the C++ op to
    // avoid a redundant copy.
    m.def(
        "norm",
        [](const TensorImplPtr& a, double ord, std::vector<int> axis, bool keepdims) {
            return norm_op(a, ord, std::move(axis), keepdims);
        },
        py::arg("a"), py::arg("ord") = 2.0, py::arg("axis") = std::vector<int>{},
        py::arg("keepdims") = false);

    // qr returns (Q, R); the C++ op returns a 2-element vector.
    m.def(
        "qr",
        [](const TensorImplPtr& a) {
            auto r = qr_op(a);
            return py::make_tuple(r[0], r[1]);
        },
        py::arg("a"));

    // svd is registered directly; it returns a tuple internally via its own
    // Python-friendly signature.
    m.def("svd", &svd_op, py::arg("a"), py::arg("compute_uv") = true);
    m.def("matrix_power", &matrix_power_op, py::arg("a"), py::arg("n"));
    m.def("pinv", &pinv_op, py::arg("a"));

    // eig returns (eigenvalues, eigenvectors) for general (possibly complex)
    // square matrices.
    m.def(
        "eig",
        [](const TensorImplPtr& a) {
            auto r = eig_op(a);
            return py::make_tuple(r[0], r[1]);
        },
        py::arg("a"));

    // eigh is the symmetric/Hermitian specialisation; it guarantees real
    // eigenvalues sorted in ascending order and is faster than eig.
    m.def(
        "eigh",
        [](const TensorImplPtr& a) {
            auto r = eigh_op(a);
            return py::make_tuple(r[0], r[1]);
        },
        py::arg("a"),
        "Symmetric/Hermitian eigendecomposition.\n"
        "Returns (eigenvalues, eigenvectors) with eigenvalues sorted ascending.\n"
        "Input must be a real symmetric square matrix.\n"
        "CPU: LAPACK ssyev/dsyev.  GPU: mlx::core::linalg::eigh.");
}

}  // namespace lucid::bindings
