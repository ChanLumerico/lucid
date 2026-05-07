// lucid/_C/bindings/bind_complex.cpp
//
// Registers the four complex-viewing ops on the top-level engine module:
// ``real`` / ``imag`` / ``complex`` / ``conj``.  Each backend has its own
// native implementation (CPU = vDSP / interleaved walk, GPU = mlx native);
// the bindings just expose the C++ op functions.

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
          "Real part of a C64 tensor — returns F32 of the same shape.");
    m.def("imag", &imag_op, py::arg("a"),
          "Imaginary part of a C64 tensor — returns F32 of the same shape.");
    m.def("complex", &complex_op, py::arg("real"), py::arg("imag"),
          "Build a C64 tensor from two F32 tensors of identical shape.");
    m.def("conj", &conj_op, py::arg("a"),
          "Element-wise complex conjugate.  Identity for real dtypes; "
          "negates the imaginary part for C64.");
}

}  // namespace lucid::bindings
