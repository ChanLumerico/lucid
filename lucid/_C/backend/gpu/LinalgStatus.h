// lucid/_C/backend/gpu/LinalgStatus.h
//
// A Metal-stream factorisation's ``info``, read off the factor.
//
// MLX's linalg routines report no status: a factorisation that fails comes
// back as an ordinary-looking array.  What is left in it says where it
// failed — a zero pivot on the diagonal of ``U``, a diagonal entry of a
// Cholesky ``L`` that is not positive — so the caller builds one flag per
// step and this raises for the first one the way the CPU stream's LAPACK
// would (``backend/LapackInfo.h``): same class, same ``info``, same matrix
// of the batch.
//
// The factorisations that complete on a singular matrix (``lu_factor``,
// ``ldl_factor``) are not read off anything: the Metal stream runs the CPU
// backend's LAPACK and uploads its real ``info`` with the factor
// (``upload_factor``).

#pragma once

#include <cstddef>
#include <variant>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../core/MemoryStats.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../LapackInfo.h"
#include "MlxBridge.h"

namespace lucid::gpu {

// Raise for the first set flag of ``failed`` — shaped ``(..., n)``, one flag
// per factorisation step of each matrix — or return when none is set.  One
// host read of a single flag when nothing failed; the whole grid is read
// only to name the failure.
inline void raise_failed_step(const ::mlx::core::array& failed,
                              const char* op,
                              const ::mlx::core::StreamOrDevice& s) {
    auto any_failed = ::mlx::core::any(failed, false, s);
    any_failed.eval();
    MemoryTracker::track_host_sync(any_failed.nbytes());
    if (!any_failed.item<bool>())
        return;
    auto flags = ::mlx::core::contiguous(failed, false, s);
    flags.eval();
    MemoryTracker::track_host_sync(flags.nbytes());
    backend::check_step_flags(flags.data<bool>(), flags.size(),
                              static_cast<std::size_t>(failed.shape(-1)), op);
}

// A CPU factorisation's ``{factor, pivots, info}`` on the Metal stream:
// the factor shaped ``shape``, the pivots ``pivot_shape``, and ``info`` one
// code per matrix — the batch dims of ``shape``.
inline std::vector<Storage>
upload_factor(const std::vector<Storage>& cpu, const Shape& shape, const Shape& pivot_shape) {
    const Shape info_shape(shape.begin(), shape.end() - 2);
    return {Storage{upload_cpu_to_gpu(std::get<CpuStorage>(cpu[0]), shape)},
            Storage{upload_cpu_to_gpu(std::get<CpuStorage>(cpu[1]), pivot_shape)},
            Storage{upload_cpu_to_gpu(std::get<CpuStorage>(cpu[2]), info_shape)}};
}

}  // namespace lucid::gpu
