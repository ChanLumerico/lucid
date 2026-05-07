// lucid/_C/bindings/bind_fft.cpp
//
// Registers FFT ops on the `lucid._C.engine.fft` sub-module (created in
// bind.cpp).  Only the four base ops are exposed here — fftn/ifftn/rfftn/irfftn.
// 1-D / 2-D / shift / freq / Hermitian variants are composed in pure Python
// inside lucid/fft/__init__.py on top of these primitives.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../ops/fft/Fftn.h"
#include "../ops/fft/Ifftn.h"
#include "../ops/fft/Irfftn.h"
#include "../ops/fft/Rfftn.h"
#include "../ops/fft/_Detail.h"

namespace py = pybind11;

namespace lucid::bindings {

// Helper used by lucid.fft.hfft / ihfft (Python composites).  Conjugates a
// complex tensor; for real tensors this is a no-op identity.  Lives in the
// fft binding rather than as a top-level op because it has only one consumer.
static TensorImplPtr conj_complex(const TensorImplPtr& a) {
    if (a->dtype() != Dtype::C64)
        return a;  // real conjugate ≡ identity
    OpScopeFull scope{"fft._conj_complex", a->device(), a->dtype(), a->shape()};
    auto in_arr = fft_detail::as_mlx_input(a);
    auto out = ::mlx::core::conjugate(in_arr, fft_detail::kMlxFftStream);
    Storage out_st = fft_detail::finalise_result(std::move(out), a->dtype(), a->shape(), a->device());
    return fft_detail::fresh(std::move(out_st), a->shape(), a->dtype(), a->device());
}

// Scalar multiplication helper used by lucid.fft for normalisation scaling.
// Works for any dtype — including C64, where the standard ``Tensor * float``
// path fails because CpuBackend lacks ``full(C64)`` (so the broadcast scalar
// can't be constructed).  Going through MLX sidesteps that by building the
// scalar directly as an mlx array.
static TensorImplPtr scale_op(const TensorImplPtr& a, double s) {
    if (s == 1.0)
        return a;
    OpScopeFull scope{"fft._scale", a->device(), a->dtype(), a->shape()};
    auto in_arr = fft_detail::as_mlx_input(a);
    ::mlx::core::array scalar_arr =
        ::mlx::core::astype(::mlx::core::array(static_cast<float>(s)),
                            ::lucid::gpu::to_mlx_dtype(a->dtype()));
    auto out = ::mlx::core::multiply(in_arr, scalar_arr, fft_detail::kMlxFftStream);
    Storage out_st = fft_detail::finalise_result(std::move(out), a->dtype(), a->shape(), a->device());
    return fft_detail::fresh(std::move(out_st), a->shape(), a->dtype(), a->device());
}

void register_fft(py::module_& m) {
    m.def("_conj_complex", &conj_complex, py::arg("a"),
          "Complex conjugate.  No-op on real tensors.  Used by hfft/ihfft.");
    m.def("_scale", &scale_op, py::arg("a"), py::arg("s"),
          "Multiply tensor by a real scalar.  Routes through MLX so it works "
          "for C64 inputs (which CpuBackend's full(C64) does not support).");

    m.def(
        "fftn",
        [](const TensorImplPtr& a, std::vector<std::int64_t> n, std::vector<int> dim) {
            return fftn_op(a, std::move(n), std::move(dim));
        },
        py::arg("a"),
        py::arg("s") = std::vector<std::int64_t>{},
        py::arg("dim") = std::vector<int>{},
        "N-dimensional discrete Fourier transform (complex output).");

    m.def(
        "ifftn",
        [](const TensorImplPtr& a, std::vector<std::int64_t> n, std::vector<int> dim) {
            return ifftn_op(a, std::move(n), std::move(dim));
        },
        py::arg("a"),
        py::arg("s") = std::vector<std::int64_t>{},
        py::arg("dim") = std::vector<int>{},
        "N-dimensional inverse discrete Fourier transform.");

    m.def(
        "rfftn",
        [](const TensorImplPtr& a, std::vector<std::int64_t> n, std::vector<int> dim) {
            return rfftn_op(a, std::move(n), std::move(dim));
        },
        py::arg("a"),
        py::arg("s") = std::vector<std::int64_t>{},
        py::arg("dim") = std::vector<int>{},
        "N-dimensional real-input DFT.  Output dtype is C64; last axis size is n//2+1.");

    m.def(
        "irfftn",
        [](const TensorImplPtr& a, std::vector<std::int64_t> n, std::vector<int> dim) {
            return irfftn_op(a, std::move(n), std::move(dim));
        },
        py::arg("a"),
        py::arg("s") = std::vector<std::int64_t>{},
        py::arg("dim") = std::vector<int>{},
        "N-dimensional inverse real DFT.  Output dtype is F32.");
}

}  // namespace lucid::bindings
