// lucid/_C/ops/linalg/_Detail.h
//
// Internal helpers shared across all linalg ops.  Nothing in this header is
// part of the public API; it is included only by .cpp files inside ops/linalg/.
// Centralising these utilities avoids copy-paste across the dozen linalg ops
// and keeps the per-op files focused on their forward/backward logic.
//
// Design notes:
//   - All helpers are either inline free functions or type aliases so there is
//     no linkage cost; the header is pure header-only.
//   - The helpers deal with three recurring concerns:
//       (a) GPU/MLX interop: extracting mlx::core::array from GpuStorage and
//           wrapping the result back;
//       (b) CPU batch dispatch: computing the number of independent matrices in
//           a batched input before looping over LAPACK calls;
//       (c) Input validation: dtype and shape guards that must run before any
//           storage allocation.
//   - kMlxLinalgStream is the one architectural constant that appears in every
//     GPU linalg kernel; its rationale is documented at the definition site.

#pragma once

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

// Pull helper aliases into this namespace so callers do not need long prefixes.
// fresh() wraps a Storage + shape + dtype + device into a new TensorImpl.
// mlx_shape_to_lucid() converts an MLX shape vector to a Lucid Shape.
using ::lucid::gpu::mlx_shape_to_lucid;
using ::lucid::helpers::fresh;

// The MLX linalg stream runs on the CPU device even on GPU tensors because
// MLX routes its linear-algebra kernels (LAPACK wrappers) through the CPU
// execution queue, not the Metal GPU queue.  This means:
//   - mlx::core::linalg::inv / qr / svd / eig / eigh / solve / cholesky all
//     expect Device::cpu as their stream argument.
//   - The resulting mlx::core::array still lives in GPU-accessible memory and
//     is passed back to the Lucid GPU storage normally.
// All GPU linalg ops pass this constant as the stream argument to the MLX call.
inline const ::mlx::core::Device kMlxLinalgStream{::mlx::core::Device::cpu};

// Extract the underlying mlx::core::array from a GPU TensorImpl.
//
// Precondition: t->device() == Device::GPU.  The function aborts with a
// descriptive error if this precondition is violated, which would indicate
// an incorrect dispatch decision upstream (e.g. a CPU tensor reaching a GPU
// code path).  Calling storage_gpu() on a non-GPU storage would be UB, so
// the explicit check is load-bearing, not just defensive.
inline ::mlx::core::array as_mlx_array_gpu(const TensorImplPtr& t) {
    if (t->device() != Device::GPU)
        ErrorBuilder("as_mlx_array_gpu").fail("not a GPU tensor");
    const auto& g = storage_gpu(t->storage());
    return *g.arr;
}

// Wrap an mlx::core::array that was produced by a linalg kernel back into a
// GPU Storage object so it can be stored inside a TensorImpl.
//
// The array is moved (not copied) into the GpuStorage so no data is copied.
// The dtype parameter is the Lucid dtype, which must agree with the element
// type of the MLX array; the caller is responsible for that consistency.
inline Storage wrap_gpu_result(::mlx::core::array&& out, Dtype dtype) {
    return Storage{gpu::wrap_mlx_array(std::move(out), dtype)};
}

// Bring allocate_cpu into scope; it allocates a raw CPU buffer of the given
// shape and dtype and returns a CpuStorage with a managed unique_ptr.
using ::lucid::helpers::allocate_cpu;

// Return the total element count of all batch dimensions of shape, where the
// last mat_dims dimensions are considered the matrix axes (typically 2).
//
// Example: shape = [4, 3, 8, 8], mat_dims = 2 → returns 4*3 = 12, meaning
// there are 12 independent 8×8 matrices packed into this tensor.
//
// Used by batched linalg CPU kernels that loop over leading batch dimensions
// before calling LAPACK on each [m×n] slice.  The loop stride between
// consecutive matrices is shape[-2] * shape[-1] elements.
inline std::int64_t leading_batch_count(const Shape& shape, std::size_t mat_dims) {
    if (shape.size() < mat_dims)
        ErrorBuilder("linalg").fail("input rank too small");
    std::int64_t b = 1;
    for (std::size_t i = 0; i + mat_dims < shape.size(); ++i)
        b *= shape[i];
    return b;
}

// Raise a not-implemented error if dt is not F32 or F64.
//
// Apple Accelerate LAPACK routines (sgetrf, dgetrf, spotrf, dsyev, etc.) only
// accept single- and double-precision real inputs.  Integer and half-precision
// dtypes are rejected here before any allocation or dispatch, so callers get
// a clear message rather than a silent miscompute or a crash inside LAPACK.
// All linalg ops call this guard before dispatching to the CPU backend.
inline void require_float(Dtype dt, const char* op) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder(op).not_implemented("only F32/F64 supported (got" +
                                         std::string(dtype_name(dt)) + ")");
}

// Raise an error if sh does not describe at least a 2-D square matrix.
//
// "Square" means sh[rank-1] == sh[rank-2].  Batched inputs (rank > 2) are
// accepted as long as the trailing two dimensions satisfy this constraint.
// This check is required by inv, det, solve, cholesky, eig, eigh, and
// matrix_power, all of which are only defined on square matrices.
inline void require_square_2d(const Shape& sh, const char* op) {
    if (sh.size() < 2)
        ErrorBuilder(op).fail("input must be at least 2-D");
    if (sh[sh.size() - 1] != sh[sh.size() - 2])
        ErrorBuilder(op).fail("last two dims must be equal (square)");
}

// Translate a LAPACK integer info return code into a Lucid error.
//
// LAPACK convention:
//   info == 0 : success
//   info <  0 : the (-info)-th argument had an illegal value
//   info >  0 : a numerical failure occurred (e.g. matrix is singular,
//               decomposition failed to converge, etc.)
//
// Both error cases are fatal at the engine level.  In practice callers should
// pre-validate inputs (require_float, require_square_2d) so that the only
// reachable error is info > 0, which signals a genuinely ill-conditioned
// matrix at runtime.
inline void check_lapack_info(int info, const char* op) {
    if (info < 0)
        ErrorBuilder(op).fail("LAPACK invalid argument index" + std::to_string(-info));
    if (info > 0)
        ErrorBuilder(op).fail("LAPACK numerical failure (info=" + std::to_string(info) + ")");
}

}  // namespace lucid::linalg_detail
