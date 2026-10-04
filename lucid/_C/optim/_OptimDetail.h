// lucid/_C/optim/_OptimDetail.h
//
// Internal helpers shared by all optimizer implementations. Provides
// thin, inline wrappers around Storage accessors and MLX scalar
// construction so that the optimizer .cpp files stay concise and
// avoid repeating the same casts and variant extractions. Nothing in
// this header is part of the public API.

#pragma once

#include <cstddef>
#include <utility>
#include <variant>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace lucid::optim_detail {

// Typed mutable pointer into a CPU storage buffer.
//
// Convenience wrapper around ``std::get<CpuStorage>(s).ptr.get()`` that
// reinterprets the raw byte pointer as ``T*``.  Used by every CPU-side
// optimizer update to walk the parameter / gradient / state buffers as
// a flat array of element-typed scalars.
//
// Parameters
// ----------
// s : Storage&
//     Storage whose ``CpuStorage`` variant should be accessed.
//
// Returns
// -------
// T*
//     Typed pointer to the first element.
//
// Raises
// ------
// std::bad_variant_access
//     If ``s`` does not hold a ``CpuStorage``.
//
// Notes
// -----
// The caller must ensure ``T`` matches ``s``'s ``Dtype`` — no runtime
// check is performed.  Mismatched ``T`` silently produces undefined
// behaviour.
template <typename T>
inline T* cpu_ptr(Storage& s) {
    return reinterpret_cast<T*>(std::get<CpuStorage>(s).ptr.get());
}

// Typed const pointer into a CPU storage buffer.
//
// Const-correct sibling of ``cpu_ptr`` for read-only access to gradient
// or state buffers.
//
// Parameters
// ----------
// s : const Storage&
//     Storage whose ``CpuStorage`` variant should be accessed.
//
// Returns
// -------
// const T*
//     Typed const pointer to the first element.
//
// Raises
// ------
// std::bad_variant_access
//     If ``s`` does not hold a ``CpuStorage``.
//
// See Also
// --------
// cpu_ptr : mutable variant.
template <typename T>
inline const T* cpu_cptr(const Storage& s) {
    return reinterpret_cast<const T*>(std::get<CpuStorage>(s).ptr.get());
}

// Total element count of a tensor.
//
// Parameters
// ----------
// t : const TensorImpl&
//     Tensor whose element count is requested.
//
// Returns
// -------
// std::size_t
//     ``t.numel()``.  Convenience alias kept here so optimizer .cpp
//     files can refer to a single namespace for all helpers.
inline std::size_t cpu_numel(const TensorImpl& t) {
    return t.numel();
}

// Mutable reference to a Storage's GPU variant.
//
// Parameters
// ----------
// s : Storage&
//     Storage whose ``GpuStorage`` variant should be accessed.
//
// Returns
// -------
// GpuStorage&
//     The variant payload.  Delegates to ``storage_gpu`` for the actual
//     extraction (which performs the variant check and error reporting).
inline GpuStorage& gpu_get(Storage& s) {
    return storage_gpu(s);
}

// Const reference to a Storage's GPU variant.
//
// Parameters
// ----------
// s : const Storage&
//     Storage whose ``GpuStorage`` variant should be accessed.
//
// Returns
// -------
// const GpuStorage&
//     The variant payload.
//
// See Also
// --------
// gpu_get : mutable variant.
inline const GpuStorage& gpu_get(const Storage& s) {
    return storage_gpu(s);
}

// Build a zero-dimensional MLX scalar array with the requested dtype.
//
// Used to form broadcast-compatible scalar operands in the GPU update
// rules — for example to express $v \leftarrow \mu\, v + g$ as a pair
// of MLX array ops where $\mu$ is a single scalar.
//
// Parameters
// ----------
// x : double
//     Scalar value to embed.
// dt : Dtype
//     Lucid dtype tag.  Translated to the corresponding MLX dtype via
//     ``gpu::to_mlx_dtype``.
//
// Returns
// -------
// mlx::core::array
//     Zero-dimensional MLX array holding ``x`` cast to ``dt``.
inline ::mlx::core::array mlx_scalar(double x, Dtype dt) {
    return gpu::mlx_scalar(x, gpu::to_mlx_dtype(dt));
}

// Replace the MLX array inside a GpuStorage with the result of an op.
//
// Used at the end of each GPU update branch to write the freshly
// computed parameter or state tensor back into the optimizer's storage
// without rebinding any of the surrounding metadata.
//
// Parameters
// ----------
// dst : GpuStorage&
//     Target storage whose ``arr`` pointer will be reassigned.
// arr : mlx::core::array&&
//     New MLX array to install; ownership is transferred.
// dt : Dtype
//     Lucid dtype tag to preserve on the wrapped array.
//
// Notes
// -----
// The previous MLX array is released when its shared_ptr drops to zero
// refs; the Lucid dtype tag is preserved via ``gpu::wrap_mlx_array``.
inline void gpu_replace(GpuStorage& dst, ::mlx::core::array&& arr, Dtype dt) {
    dst.arr = gpu::wrap_mlx_array(std::move(arr), dt).arr;
}

// Whether a state slot holds a buffer.
//
// A per-slot state vector is sized with ``resize``, which leaves every
// slot that has not been allocated as a default ``CpuStorage`` with no
// bytes.  A buffer that is allocated holds a GPU array or CPU bytes.
//
// Parameters
// ----------
// s : const Storage&
//     One entry of a per-slot state vector (``moment_``, ``vmax_`` ...).
//
// Returns
// -------
// bool
//     ``true`` when ``s`` holds a GPU array, CPU bytes, or shared memory.
//
// Notes
// -----
// An empty parameter's CPU buffer has no bytes either, so it reads as not
// held.  Allocating it again changes nothing, which is all the callers do
// with a slot that is not held.
inline bool holds_buffer(const Storage& s) {
    if (const auto* g = std::get_if<GpuStorage>(&s))
        return g->arr != nullptr;
    if (const auto* c = std::get_if<CpuStorage>(&s))
        return c->ptr != nullptr;
    return true;
}

// Dtype a per-parameter scalar of optimizer state is checkpointed in.
//
// F64 for an F64 parameter, F32 otherwise — the reference framework's
// default scalar dtype, and the widest the Metal device can hold, so a
// loader that rebuilds the scalar on the parameter's device can.
//
// Parameters
// ----------
// param_dtype : Dtype
//     Dtype of the parameter the scalar belongs to.
//
// Returns
// -------
// Dtype
//     ``Dtype::F64`` or ``Dtype::F32``.
inline Dtype state_scalar_dtype(Dtype param_dtype) {
    return param_dtype == Dtype::F64 ? Dtype::F64 : Dtype::F32;
}

// Round a scalar of optimizer state to the precision it is checkpointed at.
//
// An accumulator held in ``double`` but saved as F32 would come back from
// a checkpoint changed in its low bits, and the resumed run would part
// from the uninterrupted one.  Keeping the live value at the saved
// precision makes the round trip exact.
//
// Parameters
// ----------
// v : double
//     Value to round.
// param_dtype : Dtype
//     Dtype of the parameter the scalar belongs to.
//
// Returns
// -------
// double
//     ``v`` rounded to ``state_scalar_dtype(param_dtype)``.
inline double round_to_state_scalar(double v, Dtype param_dtype) {
    if (state_scalar_dtype(param_dtype) == Dtype::F64)
        return v;
    return static_cast<double>(static_cast<float>(v));
}

}  // namespace lucid::optim_detail
