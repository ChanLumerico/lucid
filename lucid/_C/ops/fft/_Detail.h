// lucid/_C/ops/fft/_Detail.h
//
// Internal helpers shared across all FFT ops.  Mirrors
// ``ops/linalg/_Detail.h``: header-only inline helpers, MLX interop,
// output-shape arithmetic, and dtype guards.  Nothing in this header
// is part of the public API.
//
// Architecture note (carve-out)
// -----------------------------
// FFT bypasses ``IBackend`` dispatch and calls ``mlx::core::fft``
// directly from the ops layer, mirroring the linalg pattern
// (``as_mlx_array_gpu`` + ``kMlxLinalgStream``).  For FFT specifically,
// MLX wraps Apple Accelerate vDSP on the CPU stream, so dispatching
// MLX from both ``Device::CPU`` and ``Device::GPU`` paths is consistent
// with the H3 spirit ("CPU = Accelerate, GPU = MLX") — Accelerate is
// reached transitively through MLX.

#pragma once

#include <cstdint>
#include <vector>

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/fft.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::fft_detail {

using ::lucid::helpers::fresh;

// MLX stream constant used for every FFT dispatch.
//
// Notes
// -----
// MLX has no Metal FFT kernel on Apple Silicon; it routes through
// Accelerate vDSP on the CPU stream.  GPU tensors continue to hold
// data in GPU-accessible memory — only the dispatch lane is CPU, and
// MLX handles the host/device transfer transparently.
inline const ::mlx::core::Device kMlxFftStream{::mlx::core::Device::cpu};

// Resolve the output dtype for ``fftn`` / ``ifftn``.
//
// Returns
// -------
// Dtype
//     Always ``Dtype::C64``.
//
// Raises
// ------
// LucidError
//     If ``in`` is not in ``{C64, F32, F16}``.
//
// Notes
// -----
// Real inputs (F16/F32) are accepted and promoted to complex by MLX.
inline Dtype dtype_for_complex_fft(Dtype in) {
    if (in == Dtype::C64 || in == Dtype::F32 || in == Dtype::F16)
        return Dtype::C64;
    ErrorBuilder("fft").not_implemented("fftn/ifftn requires F16/F32/C64 input, got " +
                                        std::string(dtype_name(in)));
    return Dtype::C64;
}

// Resolve the output dtype for ``rfftn``.
//
// Returns
// -------
// Dtype
//     Always ``Dtype::C64``.
//
// Raises
// ------
// LucidError
//     If ``in`` is not ``F16`` or ``F32`` (rfftn is a real-input op).
inline Dtype dtype_for_rfft(Dtype in) {
    if (in == Dtype::F32 || in == Dtype::F16)
        return Dtype::C64;
    ErrorBuilder("rfftn").not_implemented("rfftn requires F16/F32 input, got " +
                                          std::string(dtype_name(in)));
    return Dtype::C64;
}

// Resolve the output dtype for ``irfftn``.
//
// Returns
// -------
// Dtype
//     Always ``Dtype::F32``.
//
// Raises
// ------
// LucidError
//     If ``in`` is not ``C64`` (irfftn consumes a complex Hermitian half).
inline Dtype dtype_for_irfft(Dtype in) {
    if (in == Dtype::C64)
        return Dtype::F32;
    ErrorBuilder("irfftn").not_implemented("irfftn requires C64 input, got " +
                                           std::string(dtype_name(in)));
    return Dtype::F32;
}

// Normalise a possibly-negative axis list to non-negative form, in-place.
//
// Parameters
// ----------
// axes : vector<int>&
//     Axes to rewrite.  Each negative axis $a$ becomes $a + r$ where
//     $r$ is the input rank.
// rank : int
//     Input rank used as the modulus.
// op : const char*
//     Op name embedded into the error message on failure.
//
// Raises
// ------
// LucidError
//     If any normalised axis falls outside $[0, r)$.
inline void normalise_axes(std::vector<int>& axes, int rank, const char* op) {
    for (auto& ax : axes) {
        int orig = ax;
        if (ax < 0)
            ax += rank;
        if (ax < 0 || ax >= rank)
            ErrorBuilder(op).fail("axis " + std::to_string(orig) + " out of range for rank " +
                                  std::to_string(rank));
    }
}

// Populate ``axes`` with ``[0, 1, ..., rank-1]`` when empty.
//
// Notes
// -----
// MLX's ``fftn`` accepts an empty axis list with the same meaning, but
// Lucid pre-fills so subsequent shape arithmetic (output-shape helpers
// below) can index ``axes`` directly without branching.
inline void default_axes_all(std::vector<int>& axes, int rank) {
    if (!axes.empty())
        return;
    axes.reserve(rank);
    for (int i = 0; i < rank; ++i)
        axes.push_back(i);
}

// Compute the output shape for a complex FFT (``fftn`` / ``ifftn``).
//
// Parameters
// ----------
// in_shape : Shape
//     Input shape.
// n : vector<int64_t>
//     Per-axis transform lengths.  Empty means "preserve input shape".
// axes : vector<int>
//     Already normalised, non-empty (when ``n`` is non-empty).
// op : const char*
//     Op name embedded into the error message.
//
// Returns
// -------
// Shape
//     ``in_shape`` with each ``axes[i]`` overwritten by ``n[i]``, or
//     ``in_shape`` unchanged when ``n`` is empty.
//
// Raises
// ------
// LucidError
//     If ``len(n) != len(axes)`` when ``n`` is non-empty.
inline Shape complex_fft_out_shape(const Shape& in_shape,
                                   const std::vector<std::int64_t>& n,
                                   const std::vector<int>& axes,
                                   const char* op) {
    Shape out = in_shape;
    if (n.empty())
        return out;
    if (n.size() != axes.size())
        ErrorBuilder(op).fail("len(s) must match len(axes)");
    for (std::size_t i = 0; i < axes.size(); ++i)
        out[static_cast<std::size_t>(axes[i])] = n[i];
    return out;
}

// Compute the output shape for a real FFT (``rfftn``).
//
// The last entry of ``axes`` is the Hermitian-trimmed axis: it shrinks
// to $n_{\text{last}}/2 + 1$.  Other transformed axes follow ``n``
// verbatim (or the input shape when ``n`` is empty).
//
// Parameters
// ----------
// in_shape : Shape
//     Input shape (real-valued tensor).
// n : vector<int64_t>
//     Per-axis transform lengths.  Empty means "use input sizes".
// axes : vector<int>
//     Must be non-empty; the back element is the Hermitian-trimmed axis.
// op : const char*
//     Op name embedded into the error message.
//
// Returns
// -------
// Shape
//     ``in_shape`` with ``axes[i]`` rewritten as described above.
//
// Raises
// ------
// LucidError
//     If ``axes`` is empty, or ``len(n) != len(axes)`` when ``n`` is
//     non-empty.
inline Shape rfft_out_shape(const Shape& in_shape,
                            const std::vector<std::int64_t>& n,
                            const std::vector<int>& axes,
                            const char* op) {
    if (axes.empty())
        ErrorBuilder(op).fail("rfftn requires at least one axis");
    Shape out = in_shape;
    if (n.empty()) {
        // Default: keep all sizes, only reduce last axis.
        const int last = axes.back();
        out[static_cast<std::size_t>(last)] = in_shape[static_cast<std::size_t>(last)] / 2 + 1;
        return out;
    }
    if (n.size() != axes.size())
        ErrorBuilder(op).fail("len(s) must match len(axes)");
    for (std::size_t i = 0; i + 1 < axes.size(); ++i)
        out[static_cast<std::size_t>(axes[i])] = n[i];
    const int last = axes.back();
    out[static_cast<std::size_t>(last)] = n.back() / 2 + 1;
    return out;
}

// Compute the output shape for an inverse real FFT (``irfftn``).
//
// The last entry of ``axes`` is expanded back to a real-domain length:
// ``n.back()`` if supplied, otherwise the even-length default
// $2 \cdot (\text{in\_shape}[\text{axes.back()}] - 1)$.
//
// Parameters
// ----------
// in_shape : Shape
//     Input shape (Hermitian half complex tensor).
// n : vector<int64_t>
//     Per-axis transform lengths.  Empty means "use defaults".
// axes : vector<int>
//     Must be non-empty.
// op : const char*
//     Op name embedded into the error message.
//
// Returns
// -------
// Shape
//     ``in_shape`` with ``axes[i]`` rewritten as described above.
//
// Raises
// ------
// LucidError
//     If ``axes`` is empty, or ``len(n) != len(axes)`` when ``n`` is
//     non-empty.
inline Shape irfft_out_shape(const Shape& in_shape,
                             const std::vector<std::int64_t>& n,
                             const std::vector<int>& axes,
                             const char* op) {
    if (axes.empty())
        ErrorBuilder(op).fail("irfftn requires at least one axis");
    Shape out = in_shape;
    if (n.empty()) {
        const int last = axes.back();
        const std::int64_t in_last = in_shape[static_cast<std::size_t>(last)];
        out[static_cast<std::size_t>(last)] = (in_last - 1) * 2;
        return out;
    }
    if (n.size() != axes.size())
        ErrorBuilder(op).fail("len(s) must match len(axes)");
    for (std::size_t i = 0; i < axes.size(); ++i)
        out[static_cast<std::size_t>(axes[i])] = n[i];
    return out;
}

// Convert a Lucid per-axis length vector to MLX's int32 ``Shape``.
//
// Parameters
// ----------
// n : vector<int64_t>
//     Per-axis transform lengths.  Empty produces an empty MLX shape
//     (signal to MLX that defaults should be used).
//
// Returns
// -------
// mlx::core::Shape
//     The same values cast down to ``int32_t``.
//
// Raises
// ------
// LucidError
//     If any entry exceeds ``INT32_MAX`` (MLX's FFT length type).
inline ::mlx::core::Shape mlx_n_from_lucid(const std::vector<std::int64_t>& n) {
    ::mlx::core::Shape out;
    out.reserve(n.size());
    for (auto v : n) {
        if (v > std::numeric_limits<std::int32_t>::max())
            ErrorBuilder("fft").fail("FFT length exceeds INT32_MAX");
        out.push_back(static_cast<std::int32_t>(v));
    }
    return out;
}

// Wrap an MLX FFT result as a Lucid ``Storage`` on the target device.
//
// Parameters
// ----------
// out : mlx::core::array&&
//     The freshly produced FFT result (will be made contiguous before
//     wrapping).
// out_dtype : Dtype
//     Lucid-side dtype tag.
// out_shape : Shape
//     Logical Lucid shape for the result.
// dev : Device
//     Target device.  ``GPU`` wraps in-place; ``CPU`` downloads to host.
//
// Returns
// -------
// Storage
//     A ``GpuStorage`` wrapping the MLX array, or a ``CpuStorage``
//     containing a downloaded copy.
inline Storage
finalise_result(::mlx::core::array&& out, Dtype out_dtype, const Shape& out_shape, Device dev) {
    if (dev == Device::GPU)
        return Storage{::lucid::gpu::wrap_mlx_array(::mlx::core::contiguous(out), out_dtype)};
    auto wrapped = ::lucid::gpu::wrap_mlx_array(::mlx::core::contiguous(out), out_dtype);
    return Storage{::lucid::gpu::download_gpu_to_cpu(wrapped, out_shape)};
}

// Materialise a Lucid input tensor as an ``mlx::core::array``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor (CPU or GPU).
//
// Returns
// -------
// mlx::core::array
//     For GPU tensors the underlying MLX array is returned by value
//     (cheap shared handle).  For CPU tensors the buffer is uploaded
//     via ``MlxBridge`` (one host-to-device copy).
inline ::mlx::core::array as_mlx_input(const TensorImplPtr& a) {
    if (a->device() == Device::GPU) {
        const auto& g = std::get<::lucid::GpuStorage>(a->storage());
        return *g.arr;
    }
    const auto& cs = std::get<::lucid::CpuStorage>(a->storage());
    auto gs = ::lucid::gpu::upload_cpu_to_gpu(cs, a->shape());
    return *gs.arr;
}

}  // namespace lucid::fft_detail
