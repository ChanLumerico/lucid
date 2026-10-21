// lucid/_C/bindings/bind_fft.cpp
//
// Registers FFT ops on the `lucid._C.engine.fft` sub-module (created in
// bind.cpp).  Only the four base ops are exposed here — fftn/ifftn/rfftn/irfftn.
// 1-D / 2-D / shift / freq / Hermitian variants are composed in pure Python
// inside lucid/fft/__init__.py on top of these primitives.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/fft/Fftn.h"
#include "../ops/fft/Ifftn.h"
#include "../ops/fft/Irfftn.h"
#include "../ops/fft/Rfftn.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_fft(py::module_& m) {
    m.def(
        "fftn",
        [](const TensorImplPtr& a, std::vector<std::int64_t> n, std::vector<int> dim) {
            return fftn_op(a, n, dim);
        },
        py::arg("a"), py::arg("s") = std::vector<std::int64_t>{},
        py::arg("dim") = std::vector<int>{},
        "N-dimensional discrete Fourier transform (complex output).");

    m.def(
        "ifftn",
        [](const TensorImplPtr& a, std::vector<std::int64_t> n, std::vector<int> dim) {
            return ifftn_op(a, n, dim);
        },
        py::arg("a"), py::arg("s") = std::vector<std::int64_t>{},
        py::arg("dim") = std::vector<int>{}, "N-dimensional inverse discrete Fourier transform.");

    m.def(
        "rfftn",
        [](const TensorImplPtr& a, std::vector<std::int64_t> n, std::vector<int> dim) {
            return rfftn_op(a, n, dim);
        },
        py::arg("a"), py::arg("s") = std::vector<std::int64_t>{},
        py::arg("dim") = std::vector<int>{},
        "N-dimensional real-input DFT.  Output dtype is C64; last axis size is n//2+1.");

    m.def(
        "irfftn",
        [](const TensorImplPtr& a, std::vector<std::int64_t> n, std::vector<int> dim) {
            return irfftn_op(a, n, dim);
        },
        py::arg("a"), py::arg("s") = std::vector<std::int64_t>{},
        py::arg("dim") = std::vector<int>{},
        "N-dimensional inverse real DFT.  Output dtype is F32.");
}

}  // namespace lucid::bindings
