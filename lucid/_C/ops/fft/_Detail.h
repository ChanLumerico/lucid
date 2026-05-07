// lucid/_C/ops/fft/_Detail.h
//
// Internal helpers shared across all FFT ops.  Mirrors ops/linalg/_Detail.h:
// header-only inline helpers, MLX interop, output-shape arithmetic, dtype
// guards.  Nothing in this header is part of the public API.
//
// Architecture note (carve-out):
//   FFT bypasses IBackend dispatch and calls mlx::core::fft directly from the
//   ops layer.  The same pattern is used by linalg (_Detail.h: as_mlx_array_gpu
//   + kMlxLinalgStream).  For FFT, MLX itself wraps Apple Accelerate vDSP on
//   the CPU stream, so calling MLX from both Device::CPU and Device::GPU paths
//   is consistent with H3 ("CPU = Accelerate, GPU = MLX") in spirit — Accelerate
//   is reached transitively through MLX.

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

// MLX FFT runs on the CPU stream — Apple Silicon does not yet have a Metal
// FFT kernel, so MLX dispatches to vDSP under the hood.  GPU tensors still
// store data in GPU-accessible memory; only the dispatch lane is CPU.
inline const ::mlx::core::Device kMlxFftStream{::mlx::core::Device::cpu};

// Output dtype rules:
//   complex DFT (fftn / ifftn): input must be C64 → output C64.
//                                Real input (F32) is accepted and promoted.
//   real DFT (rfftn): input F32 → output C64.
//   inverse real DFT (irfftn): input C64 → output F32.
inline Dtype dtype_for_complex_fft(Dtype in) {
    if (in == Dtype::C64 || in == Dtype::F32 || in == Dtype::F16)
        return Dtype::C64;
    ErrorBuilder("fft").not_implemented(
        "fftn/ifftn requires F16/F32/C64 input, got " + std::string(dtype_name(in)));
    return Dtype::C64;
}

inline Dtype dtype_for_rfft(Dtype in) {
    if (in == Dtype::F32 || in == Dtype::F16)
        return Dtype::C64;
    ErrorBuilder("rfftn").not_implemented(
        "rfftn requires F16/F32 input, got " + std::string(dtype_name(in)));
    return Dtype::C64;
}

inline Dtype dtype_for_irfft(Dtype in) {
    if (in == Dtype::C64)
        return Dtype::F32;
    ErrorBuilder("irfftn").not_implemented(
        "irfftn requires C64 input, got " + std::string(dtype_name(in)));
    return Dtype::F32;
}

// Normalise a possibly-negative axis list to non-negative form, in-place.
// Validates each axis is in [-rank, rank).
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

// If `axes` is empty, populate it with [0, 1, ..., rank-1] (full transform).
// MLX fftn() accepts an empty axis list to mean the same thing, but we
// pre-fill to make subsequent shape arithmetic easier.
inline void default_axes_all(std::vector<int>& axes, int rank) {
    if (!axes.empty())
        return;
    axes.reserve(rank);
    for (int i = 0; i < rank; ++i)
        axes.push_back(i);
}

// Compute the output shape of a complex FFT (fftn / ifftn).
//   - If `n` is empty: output shape == input shape.
//   - Otherwise: out_shape[axes[i]] = n[i].
// Validates len(n) == len(axes).
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

// Compute the output shape of a real FFT (rfftn).
//   The last specified axis is reduced to n[-1] // 2 + 1.
//   Other specified axes follow `n` if provided.
//   If `n` is empty, the input shape is used and the last axis becomes
//   in_shape[axes.back()] // 2 + 1.
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

// Compute the output shape of an inverse real FFT (irfftn).
//   The last specified axis is expanded to n[-1] (default: 2*(in_shape[ax]-1)).
//   Other specified axes follow `n` if provided.
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

// Build an mlx::core::Shape vector from a Lucid Shape — only the indices
// listed in `axes` are taken, in the order given.  Used as MLX's `n` argument.
// Returns an empty mlx::core::Shape when `n` is empty (so MLX uses defaults).
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

// Convert an MLX array result into a Lucid Storage on the requested device.
//   GPU device: wrap the MLX array directly (no copy).
//   CPU device: download to a CpuStorage (one device-to-host copy).
inline Storage finalise_result(::mlx::core::array&& out,
                               Dtype out_dtype,
                               const Shape& out_shape,
                               Device dev) {
    if (dev == Device::GPU)
        return Storage{::lucid::gpu::wrap_mlx_array(::mlx::core::contiguous(out), out_dtype)};
    auto wrapped = ::lucid::gpu::wrap_mlx_array(::mlx::core::contiguous(out), out_dtype);
    return Storage{::lucid::gpu::download_gpu_to_cpu(wrapped, out_shape)};
}

// Materialise an input as an mlx::core::array regardless of device.
//   GPU input: extract the mlx array from GpuStorage.
//   CPU input: upload via MlxBridge.
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
