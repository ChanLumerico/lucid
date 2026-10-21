// lucid/_C/bindings/bind_complex.cpp
//
// Registers the six complex-viewing ops on the top-level engine module:
// ``real`` / ``imag`` / ``view_as_real`` / ``view_as_complex`` / ``complex`` /
// ``conj``.  Each backend has its own native implementation (CPU = vDSP /
// interleaved walk, GPU = mlx native / zero-copy lane reinterpret); the
// bindings just expose the C++ op functions.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/complex/Complex.h"
#include "../ops/complex/Conj.h"
#include "../ops/complex/Imag.h"
#include "../ops/complex/Real.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_complex(py::module_& m) {
    m.def("real", &real_op, py::arg("a"),
          "Real part of a complex tensor — returns a zero-copy real view of the same shape.");
    m.def("imag", &imag_op, py::arg("a"),
          "Imaginary part of a complex tensor — returns a zero-copy real view of the same shape.");
    m.def("view_as_real", &view_as_real_op, py::arg("a"),
          "Reinterpret a complex tensor as a real view with a trailing (..., 2) pair axis.");
    m.def("view_as_complex", &view_as_complex_op, py::arg("a"),
          "Reinterpret a real (..., 2) tensor as a complex view of shape (...).");
    m.def("complex", &complex_op, py::arg("real"), py::arg("imag"),
          "Build a complex tensor from two real tensors of identical shape.");
    m.def("conj", &conj_op, py::arg("a"),
          "Element-wise complex conjugate.  Identity for real dtypes; "
          "negates the imaginary part for complex dtypes.");
}

}  // namespace lucid::bindings
