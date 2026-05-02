#pragma once

// =====================================================================
// linalg internal helpers — strict backend split.
// =====================================================================
//
// CPU stream → Apple Accelerate LAPACK (via lucid::backend::cpu::lapack_*)
// GPU stream → MLX (mlx::core::linalg::*)
//
// `as_mlx_array` / `wrap_result` are GPU-only helpers; CPU paths must use
// the Lapack wrappers directly. `ndim_check_2d_or_batched` and
// `batched_matrix_loop` factor out the boilerplate of iterating leading
// batch dims around a per-(M,N) matrix kernel.
//
// All routines support F32 and F64; integer / bool dtypes are rejected at
// the op surface.

#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::linalg_detail {

// Re-export of the canonical helper in `core/Helpers.h`.
using ::lucid::gpu::mlx_shape_to_lucid;
using ::lucid::helpers::fresh;

// ----- GPU-only helpers ------------------------------------------------ //
// `as_mlx_array_gpu` / `wrap_gpu_result` are *only* used inside the GPU
// branch (device == Device::GPU) — never call from CPU code. The CPU branch
// must dispatch through `lucid::backend::cpu::lapack_*` directly.
//
// MLX-internal note: `mlx::core::linalg::*` ops are implemented on the CPU
// backend only (they wrap LAPACK). The result must be requested on the CPU
// stream regardless of the input array's home device — the dispatch helper
// `kMlxLinalgStream` makes that explicit.

inline const ::mlx::core::Device kMlxLinalgStream{::mlx::core::Device::cpu};

inline ::mlx::core::array as_mlx_array_gpu(const TensorImplPtr& t) {
    if (t->device() != Device::GPU)
        ErrorBuilder("as_mlx_array_gpu").fail("not a GPU tensor");
    const auto& g = storage_gpu(t->storage());
    return *g.arr;
}

inline Storage wrap_gpu_result(::mlx::core::array&& out, Dtype dtype) {
    return Storage{gpu::wrap_mlx_array(std::move(out), dtype)};
}

// ----- CPU storage helpers --------------------------------------------- //

using ::lucid::helpers::allocate_cpu;

// ----- batched matrix loop --------------------------------------------- //
//
// Many linalg ops accept (..., M, N) shapes and apply the same per-matrix
// kernel to each leading-batch slice. This helper:
//   1. validates `shape.size() >= 2`
//   2. computes batch_count = product of leading dims
//   3. invokes `kernel(batch_index, M, N)` once per slice
//
// Caller is expected to compute slice pointers from batch_index using
// shape_numel of trailing dims. The helper itself is layout-agnostic.

inline std::int64_t leading_batch_count(const Shape& shape, std::size_t mat_dims) {
    if (shape.size() < mat_dims)
        ErrorBuilder("linalg").fail("input rank too small");
    std::int64_t b = 1;
    for (std::size_t i = 0; i + mat_dims < shape.size(); ++i)
        b *= shape[i];
    return b;
}

inline void require_float(Dtype dt, const char* op) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder(op).not_implemented("only F32/F64 supported (got" +
                                         std::string(dtype_name(dt)) + ")");
}

inline void require_square_2d(const Shape& sh, const char* op) {
    if (sh.size() < 2)
        ErrorBuilder(op).fail("input must be at least 2-D");
    if (sh[sh.size() - 1] != sh[sh.size() - 2])
        ErrorBuilder(op).fail("last two dims must be equal (square)");
}

inline void check_lapack_info(int info, const char* op) {
    if (info < 0)
        ErrorBuilder(op).fail("LAPACK invalid argument index" + std::to_string(-info));
    if (info > 0)
        ErrorBuilder(op).fail("LAPACK numerical failure (info=" + std::to_string(info) + ")");
}

}  // namespace lucid::linalg_detail
