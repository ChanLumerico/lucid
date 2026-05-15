// lucid/_C/core/Helpers.h
//
// Lightweight inline factory functions used throughout the op and backend
// layers.  These are not part of the public API surface — they exist solely
// to reduce boilerplate in op kernel implementations that frequently need to
// allocate a zeroed CPU buffer and wrap it into a TensorImpl.
//
// The helpers live in the lucid::helpers sub-namespace to make call sites
// self-documenting (helpers::fresh(...), helpers::allocate_cpu(...)).

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

// Allocates a zero-initialised CPU buffer large enough to hold shape_numel(shape)
// elements of type dt and returns it as a CpuStorage.
//
// The buffer is 64-byte aligned (kCpuAlignment) and zeroed via memset.
// Allocating zero bytes (empty shape or a dimension of 0) returns a CpuStorage
// with a null ptr and nbytes == 0.
inline CpuStorage allocate_cpu(const Shape& shape, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = shape_numel(shape) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    if (s.nbytes > 0)
        std::memset(s.ptr.get(), 0, s.nbytes);
    return s;
}

// Wraps an already-constructed Storage (CPU, GPU, or Shared) and its geometry
// into a new TensorImpl with requires_grad == false.  The Storage is moved in,
// so no copy occurs.  The stride is initialised to the C-contiguous default
// by the TensorImpl constructor.
inline TensorImplPtr fresh(Storage&& s, Shape shape, Dtype dt, Device device) {
    return std::make_shared<TensorImpl>(std::move(s), std::move(shape), dt, device, false);
}

}  // namespace lucid::helpers
