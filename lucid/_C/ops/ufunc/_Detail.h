#pragma once

// =====================================================================
// ufunc internal helpers — shared by Var/Trace/Scan and any future ufunc
// additions. Header-only inline functions. Not user-facing.
// =====================================================================

#include <cstring>

#include "../../core/Allocator.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::ufunc_detail {

inline CpuStorage allocate_cpu(const Shape& shape, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = shape_numel(shape) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    if (s.nbytes > 0)
        std::memset(s.ptr.get(), 0, s.nbytes);
    return s;
}

inline TensorImplPtr fresh(Storage&& s, Shape shape, Dtype dt, Device device) {
    return std::make_shared<TensorImpl>(std::move(s), std::move(shape), dt, device,
                                        /*requires_grad=*/false);
}

}  // namespace lucid::ufunc_detail
