#pragma once

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
    return std::make_shared<TensorImpl>(std::move(s), std::move(shape), dt, device, false);
}

}  // namespace lucid::helpers
