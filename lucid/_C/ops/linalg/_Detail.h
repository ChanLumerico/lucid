// lucid/_C/ops/linalg/_Detail.h
//
// Shared internal helpers for the engine-side linalg ops.
//
// Nothing in this header is part of the public API; it is included only by
// ``.cpp`` files under ``lucid/_C/ops/linalg/``.  Centralising these
// utilities avoids copy-paste across the dozen linalg ops (cholesky, lu,
// qr, svd, eig, eigh, solve, …) and keeps each per-op file focused on its
// forward / backward logic.
//
// Design notes
// ------------
// - All helpers are either ``inline`` free functions or type aliases, so the
//   header is pure header-only with zero linkage cost.
// - The helpers address three recurring concerns:
//     (a) GPU/MLX interop — extracting ``mlx::core::array`` from a
//         ``GpuStorage`` and wrapping the result back into a ``Storage``.
//     (b) CPU batch dispatch — computing the number of independent matrices
//         in a batched input before looping over LAPACK calls.
//     (c) Input validation — dtype and shape guards that must run *before*
//         any storage allocation so failures are cheap and obvious.
// - ``kMlxLinalgStream`` is the one architectural constant that appears in
//   every GPU linalg kernel; its rationale is documented at the definition.

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

// Re-export common helper aliases into this namespace so callers do not have
// to spell out long ``::lucid::helpers::`` / ``::lucid::gpu::`` prefixes:
// - ``fresh()`` wraps a ``Storage`` + ``Shape`` + ``Dtype`` + ``Device`` into
//   a freshly allocated ``TensorImpl``.
// - ``mlx_shape_to_lucid()`` converts an MLX shape vector to a Lucid
//   ``Shape``.
using ::lucid::gpu::mlx_shape_to_lucid;
using ::lucid::helpers::fresh;

// MLX linalg stream — pinned to the CPU device.
//
// MLX routes its linear-algebra kernels (which are LAPACK wrappers under the
// hood) through the CPU execution queue, not the Metal GPU queue.  This means
// the helpers ``mlx::core::linalg::inv``, ``qr``, ``svd``, ``eig``, ``eigh``,
// ``solve``, ``cholesky`` all expect ``Device::cpu`` as their stream argument
// — even when their inputs live in unified GPU-accessible memory.  Passing
// the GPU device would either silently fall back or trip an assertion in
// recent MLX builds.
//
// The resulting ``mlx::core::array`` still resides in GPU-accessible memory
// and is forwarded back into Lucid's GPU storage normally; only the *stream*
// distinction is on the CPU side.
//
// All engine-side GPU linalg ops pass this constant as the ``StreamOrDevice``
// argument to the MLX call.
inline const ::mlx::core::Device kMlxLinalgStream{::mlx::core::Device::cpu};

// Extract the underlying ``mlx::core::array`` from a GPU ``TensorImpl``.
//
// Used by every GPU linalg dispatch to obtain the MLX handle backing the
// input.  The explicit device check is load-bearing: ``storage_gpu()`` on a
// non-GPU ``Storage`` would be undefined behaviour, so this guard converts
// an upstream dispatch bug into a clear engine-level error.
//
// Parameters
// ----------
// t : const TensorImplPtr&
//     A GPU tensor.
//
// Returns
// -------
// mlx::core::array
//     A handle copy referring to the same backing memory as ``t``.
//
// Raises
// ------
// LucidError
//     If ``t->device() != Device::GPU``.
inline ::mlx::core::array as_mlx_array_gpu(const TensorImplPtr& t) {
    if (t->device() != Device::GPU)
        ErrorBuilder("as_mlx_array_gpu").fail("not a GPU tensor");
    const auto& g = storage_gpu(t->storage());
    return *g.arr;
}

// Wrap an ``mlx::core::array`` produced by a linalg kernel back into a
// ``Storage`` so it can be stored inside a ``TensorImpl``.
//
// The MLX array is *moved* (not copied) into the new ``GpuStorage`` so there
// is no data duplication; the underlying buffer is shared.
//
// Parameters
// ----------
// out : mlx::core::array&&
//     Array returned by an MLX kernel.  Consumed.
// dtype : Dtype
//     The Lucid dtype the caller will tag the new ``TensorImpl`` with; must
//     agree with the element type of ``out``.  The helper does not enforce
//     this — responsibility lies with the caller.
//
// Returns
// -------
// Storage
//     A GPU ``Storage`` referring to ``out``.
inline Storage wrap_gpu_result(::mlx::core::array&& out, Dtype dtype) {
    return Storage{gpu::wrap_mlx_array(std::move(out), dtype)};
}

// Re-export ``allocate_cpu`` into this namespace.  It allocates a raw CPU
// buffer of the given shape and dtype and returns a ``CpuStorage`` backed
// by a managed ``unique_ptr``.
using ::lucid::helpers::allocate_cpu;

// Return the number of leading-batch matrices packed into a tensor of shape
// ``shape`` whose trailing ``mat_dims`` axes constitute the matrix part.
//
// Used by batched CPU linalg kernels: they loop over independent matrices,
// calling LAPACK once per slice.  The stride between consecutive matrices
// is ``shape[-2] * shape[-1]`` elements (assuming the standard contiguous
// row-major layout enforced by the dispatcher).
//
// Parameters
// ----------
// shape : const Shape&
//     The full tensor shape, including matrix axes.
// mat_dims : std::size_t
//     Number of trailing matrix dimensions — almost always ``2``.
//
// Returns
// -------
// std::int64_t
//     The product of all leading (non-matrix) dimensions.  Equals ``1`` for
//     a non-batched ``mat_dims``-dimensional input.
//
// Examples
// --------
// ``leading_batch_count({4, 3, 8, 8}, 2)`` returns ``12`` — twelve
// independent $8 \times 8$ matrices packed into a single tensor.
//
// Raises
// ------
// LucidError
//     If ``shape.size() < mat_dims``.
inline std::int64_t leading_batch_count(const Shape& shape, std::size_t mat_dims) {
    if (shape.size() < mat_dims)
        ErrorBuilder("linalg").fail("input rank too small");
    std::int64_t b = 1;
    for (std::size_t i = 0; i + mat_dims < shape.size(); ++i)
        b *= shape[i];
    return b;
}

// Reject non-float dtypes with a clear "not implemented" error.
//
// Apple Accelerate LAPACK (``sgetrf``, ``dgetrf``, ``spotrf``, ``dsyev``,
// ``ssytrf``, ``strtrs``, …) only accepts single- and double-precision real
// inputs.  Integer and half-precision dtypes are rejected here *before* any
// allocation or dispatch, so callers see an actionable error rather than a
// silent miscompute or a crash inside LAPACK.
//
// All linalg ops call this guard before dispatching to the CPU backend; the
// equivalent check on the GPU path is handled inside the MLX wrappers.
//
// Parameters
// ----------
// dt : Dtype
//     The dtype to validate.
// op : const char*
//     Symbolic op name used in the error message (e.g. ``"cholesky"``).
//
// Raises
// ------
// LucidError
//     With a not-implemented status if ``dt`` is neither ``F32`` nor
//     ``F64``.
inline void require_float(Dtype dt, const char* op) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder(op).not_implemented("only F32/F64 supported (got" +
                                         std::string(dtype_name(dt)) + ")");
}

// Validate that ``sh`` describes at least a 2-D square matrix.
//
// "Square" here means ``sh[rank-1] == sh[rank-2]``.  Batched inputs
// (``rank > 2``) are accepted as long as the trailing two dimensions
// satisfy this constraint — leading dimensions are interpreted as batch.
// Used by ``inv``, ``det``, ``solve``, ``cholesky``, ``eig``, ``eigh``,
// ``matrix_power``, ``ldl_factor`` and any other op only defined on square
// matrices.
//
// Parameters
// ----------
// sh : const Shape&
//     Shape to validate.
// op : const char*
//     Symbolic op name used in the error message.
//
// Raises
// ------
// LucidError
//     If ``sh.size() < 2``.
// LucidError
//     If the last two dimensions of ``sh`` are not equal.
inline void require_square_2d(const Shape& sh, const char* op) {
    if (sh.size() < 2)
        ErrorBuilder(op).fail("input must be at least 2-D");
    if (sh[sh.size() - 1] != sh[sh.size() - 2])
        ErrorBuilder(op).fail("last two dims must be equal (square)");
}

// Translate a LAPACK ``info`` return code into a Lucid error.
//
// LAPACK convention
// -----------------
// - ``info == 0`` — success.
// - ``info < 0``  — the ``(-info)``-th argument had an illegal value.  This
//   is always a Lucid bug: it means our argument-marshalling routed an
//   inconsistent shape, leading dimension, or work-size to LAPACK.
// - ``info > 0``  — a numerical failure occurred (singular factor,
//   non-positive Cholesky pivot, failure to converge, …).  This may be a
//   user-data issue — e.g. passing a non-SPD matrix to Cholesky.
//
// Both cases are fatal at the engine level.  Because callers pre-validate
// with ``require_float`` / ``require_square_2d``, in practice the only
// reachable error in production is ``info > 0`` (ill-conditioned input).
//
// Parameters
// ----------
// info : int
//     The LAPACK status return.
// op : const char*
//     Symbolic op name used in the error message.
//
// Raises
// ------
// LucidError
//     If ``info != 0``.
inline void check_lapack_info(int info, const char* op) {
    if (info < 0)
        ErrorBuilder(op).fail("LAPACK invalid argument index" + std::to_string(-info));
    if (info > 0)
        ErrorBuilder(op).fail("LAPACK numerical failure (info=" + std::to_string(info) + ")");
}

}  // namespace lucid::linalg_detail
