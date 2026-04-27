#pragma once

// =====================================================================
// bfunc internal helpers — shared by Compare/Bitwise/Dot/Inner/Outer/Tensordot.
// Not included by user code. Header-only inline functions to avoid linker
// duplicates.
// =====================================================================

#include <cstring>
#include <stdexcept>
#include <variant>

#include "../../api.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::bfunc_detail {

inline CpuStorage allocate_cpu(const Shape& shape, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = shape_numel(shape) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    if (s.nbytes > 0) std::memset(s.ptr.get(), 0, s.nbytes);
    return s;
}

inline TensorImplPtr fresh(Storage&& s, Shape shape, Dtype dt, Device device) {
    return std::make_shared<TensorImpl>(std::move(s), std::move(shape),
                                        dt, device, /*requires_grad=*/false);
}

inline void validate_pair(const TensorImplPtr& a, const TensorImplPtr& b,
                          const char* op) {
    if (!a || !b)
        throw LucidError(std::string(op) + ": null input");
    if (a->dtype_ != b->dtype_)
        throw DtypeMismatch(std::string(dtype_name(a->dtype_)),
                            std::string(dtype_name(b->dtype_)),
                            std::string(op));
    if (a->device_ != b->device_)
        throw DeviceMismatch(std::string(device_name(a->device_)),
                             std::string(device_name(b->device_)),
                             std::string(op));
}

inline void validate_pair_eq_shape(const TensorImplPtr& a,
                                   const TensorImplPtr& b, const char* op) {
    validate_pair(a, b, op);
    if (a->shape_ != b->shape_)
        throw ShapeMismatch(a->shape_, b->shape_, std::string(op));
}

}  // namespace lucid::bfunc_detail
