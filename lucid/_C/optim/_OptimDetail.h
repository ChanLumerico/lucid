// lucid/_C/optim/_OptimDetail.h
//
// Internal helpers shared by all optimizer implementations. Provides
// thin, inline wrappers around Storage accessors and MLX scalar
// construction so that the optimizer .cpp files stay concise and
// avoid repeating the same casts and variant extractions. Nothing in
// this header is part of the public API.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "../core/ErrorBuilder.h"
#include "../core/Half.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace lucid::optim_detail {

// Reject a hyper-parameter that breaks one of an optimizer's rules.
//
// The form every ``check_hyperparams`` states its rules in, so a value
// set through ``set_hyperparams`` meets exactly the constructor's checks.
//
// Parameters
// ----------
// ok : bool
//     Whether the rule holds.  Write it so a NaN fails it where the
//     reference framework's ``not 0.0 <= x`` would, e.g. ``x >= 0.0``.
// op : const char*
//     Optimizer name, the prefix of the message.
// msg : const char*
//     The rule, e.g. ``"lr must be >= 0"``.
//
// Raises
// ------
// InvalidArgument
//     When ``ok`` is false.
inline void require(bool ok, const char* op, const char* msg) {
    if (!ok)
        ErrorBuilder(op).invalid_argument(msg);
}

// The reference framework's constructor checks shared by the Adam family
// (Adam, AdamW, NAdam, RAdam, Adamax).
//
// Written as ``x >= 0`` so a NaN fails, as the reference's ``not 0.0 <= x``
// does.
//
// Raises
// ------
// InvalidArgument
//     If ``lr``, ``eps`` or ``weight_decay`` is negative or NaN, or a beta
//     lies outside ``[0, 1)``.
inline void check_adam_family(
    const char* op, double lr, double beta1, double beta2, double eps, double weight_decay) {
    require(lr >= 0.0, op, "lr must be >= 0");
    require(eps >= 0.0, op, "eps must be >= 0");
    require(beta1 >= 0.0 && beta1 < 1.0, op, "beta1 must be in [0, 1)");
    require(beta2 >= 0.0 && beta2 < 1.0, op, "beta2 must be in [0, 1)");
    require(weight_decay >= 0.0, op, "weight_decay must be >= 0");
}

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

// Fill a CPU storage buffer of ``n`` elements with ``val`` converted to ``dt``.
inline void cpu_fill(Storage& s, Dtype dt, std::size_t n, double val) {
    if (dt == Dtype::F32) {
        auto* q = cpu_ptr<float>(s);
        const float v = static_cast<float>(val);
        for (std::size_t k = 0; k < n; ++k)
            q[k] = v;
    } else if (dt == Dtype::F64) {
        auto* q = cpu_ptr<double>(s);
        for (std::size_t k = 0; k < n; ++k)
            q[k] = val;
    } else if (dt == Dtype::F16 || dt == Dtype::BF16) {
        auto* q = cpu_ptr<std::uint16_t>(s);
        const float vf = static_cast<float>(val);
        const std::uint16_t bits = (dt == Dtype::BF16) ? backend::detail::float_to_bfloat16_bits(vf)
                                                       : backend::detail::float_to_half_bits(vf);
        for (std::size_t k = 0; k < n; ++k)
            q[k] = bits;
    }
}

// Dispatch a CPU optimizer step over ``F32``, ``F64``, ``F16``, or ``BF16``
// parameter / gradient / state buffers.  ``F16`` and ``BF16`` buffers are
// widened to ``float`` via ``Half.h`` bulk conversions, stepped in ``float``
// (op-math precision), and narrowed back to ``F16`` / ``BF16`` in place.
template <typename Fn, typename... StatePtrs>
inline void dispatch_cpu_step(const char* op,
                              Dtype dt,
                              std::size_t n,
                              Storage& param_s,
                              const Storage& grad_s,
                              Fn&& fn,
                              StatePtrs*... states) {
    if (dt == Dtype::F32) {
        fn(cpu_ptr<float>(param_s), cpu_cptr<float>(grad_s),
           (states ? cpu_ptr<float>(*states) : static_cast<float*>(nullptr))...);
    } else if (dt == Dtype::F64) {
        fn(cpu_ptr<double>(param_s), cpu_cptr<double>(grad_s),
           (states ? cpu_ptr<double>(*states) : static_cast<double*>(nullptr))...);
    } else if (dt == Dtype::F16 || dt == Dtype::BF16) {
        const bool is_bf16 = (dt == Dtype::BF16);
        std::vector<float> p_f32(n);
        std::vector<float> g_f32(n);
        backend::detail::half_or_bfloat16_to_float_bulk(cpu_cptr<std::uint16_t>(param_s),
                                                        p_f32.data(), n, is_bf16);
        backend::detail::half_or_bfloat16_to_float_bulk(cpu_cptr<std::uint16_t>(grad_s),
                                                        g_f32.data(), n, is_bf16);
        constexpr std::size_t kNumStates = sizeof...(StatePtrs);
        std::array<Storage*, kNumStates> state_arr{states...};
        std::array<std::vector<float>, kNumStates> state_f32{};
        for (std::size_t idx = 0; idx < kNumStates; ++idx) {
            if (state_arr[idx] != nullptr) {
                state_f32[idx].resize(n);
                backend::detail::half_or_bfloat16_to_float_bulk(
                    cpu_cptr<std::uint16_t>(*state_arr[idx]), state_f32[idx].data(), n, is_bf16);
            }
        }
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            fn(p_f32.data(), static_cast<const float*>(g_f32.data()),
               (state_arr[Is] ? state_f32[Is].data() : static_cast<float*>(nullptr))...);
        }(std::make_index_sequence<kNumStates>{});
        backend::detail::float_to_half_or_bfloat16_bulk(
            p_f32.data(), cpu_ptr<std::uint16_t>(param_s), n, is_bf16);
        for (std::size_t idx = 0; idx < kNumStates; ++idx) {
            if (state_arr[idx] != nullptr) {
                backend::detail::float_to_half_or_bfloat16_bulk(
                    state_f32[idx].data(), cpu_ptr<std::uint16_t>(*state_arr[idx]), n, is_bf16);
            }
        }
    } else {
        ErrorBuilder(op).not_implemented("dtype not supported (F16/BF16/F32/F64)");
    }
    storage_cpu(param_s).bump_version();
}

}  // namespace lucid::optim_detail
