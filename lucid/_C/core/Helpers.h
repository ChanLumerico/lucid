#pragma once

// =====================================================================
// Lucid C++ engine — common low-level helpers (single source of truth).
// =====================================================================
//
// This header consolidates primitive helpers that previously lived as
// duplicate inline definitions in every op-family `_Detail.h` (bfunc,
// ufunc, utils, linalg, plus scattered nn ops).
//
// Op code should call `lucid::helpers::*` directly. The legacy
// `bfunc_detail::allocate_cpu`, `utils_detail::allocate_cpu`, etc. names
// are being kept temporarily as namespace re-exports until the migration
// finishes (see the per-family `_Detail.h` files).
//
// Layer: core/. Depends on Allocator.h, Storage.h, TensorImpl.h.

#include <cstring>
#include <memory>
#include <utility>

#include "../api.h"
#include "Allocator.h"
#include "Shape.h"
#include "Storage.h"
#include "TensorImpl.h"
#include "fwd.h"

namespace lucid::helpers {

/// Allocate a contiguous, zero-filled CPU buffer for the given shape/dtype.
/// Returned `CpuStorage` owns its bytes via `unique_ptr` aligned for SIMD.
inline CpuStorage allocate_cpu(const Shape& shape, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = shape_numel(shape) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    if (s.nbytes > 0)
        std::memset(s.ptr.get(), 0, s.nbytes);
    return s;
}

/// Construct a fresh `TensorImpl` (no autograd metadata) wrapping the given
/// storage. Replaces the four copies of this helper that used to live in
/// each op-family `_Detail.h`.
inline TensorImplPtr fresh(Storage&& s, Shape shape, Dtype dt, Device device) {
    return std::make_shared<TensorImpl>(std::move(s), std::move(shape), dt, device,
                                        /*requires_grad=*/false);
}

}  // namespace lucid::helpers
//
// MLX-side helpers (e.g. `mlx_scalar`) live in
// `backend/gpu/MlxBridge.h` to keep this header free of MLX dependencies
// and preserve the core ← backend layer rule.
