#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Mirrors `lucid/linalg/`. One header per op for full per-op modularity.
#include "../ops/linalg/Inv.h"
#include "../ops/linalg/Det.h"
#include "../ops/linalg/Solve.h"
#include "../ops/linalg/Cholesky.h"
#include "../ops/linalg/Norm.h"
#include "../ops/linalg/QR.h"
#include "../ops/linalg/SVD.h"
#include "../ops/linalg/MatrixPower.h"
#include "../ops/linalg/Pinv.h"
#include "../ops/linalg/Eig.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_linalg(py::module_& m) {
    m.def("inv",      &inv_op,      py::arg("a"));
    m.def("det",      &det_op,      py::arg("a"));
    m.def("solve",    &solve_op,    py::arg("a"), py::arg("b"));
    m.def("cholesky", &cholesky_op, py::arg("a"), py::arg("upper") = false);

    m.def("norm",
          [](const TensorImplPtr& a, double ord, std::vector<int> axis,
             bool keepdims) { return norm_op(a, ord, std::move(axis), keepdims); },
          py::arg("a"), py::arg("ord") = 2.0,
          py::arg("axis") = std::vector<int>{}, py::arg("keepdims") = false);

    m.def("qr",
          [](const TensorImplPtr& a) {
              auto [Q, R] = qr_op(a);
              return py::make_tuple(Q, R);
          },
          py::arg("a"));

    m.def("svd", &svd_op, py::arg("a"), py::arg("compute_uv") = true);
    m.def("matrix_power", &matrix_power_op, py::arg("a"), py::arg("n"));
    m.def("pinv",         &pinv_op,         py::arg("a"));
    m.def("eig",
          [](const TensorImplPtr& a) {
              auto [w, v] = eig_op(a);
              return py::make_tuple(w, v);
          },
          py::arg("a"));
}

}  // namespace lucid::bindings
