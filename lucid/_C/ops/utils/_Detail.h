#pragma once

// =====================================================================
// utils internal helpers — shared by Concat/Repeat/Pad/Layout/Tri/Select/
// Sort/Meshgrid. Header-only inline functions.
// =====================================================================

#include <cstring>
#include <vector>

#include <mlx/array.h>

#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Exceptions.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../core/fwd.h"

namespace lucid::utils_detail {

// Re-exports of the canonical helpers in `core/Helpers.h`.
using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

inline Shape mlx_shape_to_lucid(const ::mlx::core::Shape& s) {
    Shape out;
    out.reserve(s.size());
    for (auto d : s)
        out.push_back(static_cast<std::int64_t>(d));
    return out;
}

inline std::size_t numel(const Shape& s) {
    return shape_numel(s);
}

inline void check_dtype_device_match(const std::vector<TensorImplPtr>& xs, const char* op) {
    if (xs.empty())
        ErrorBuilder(op).fail("empty input");
    Dtype dt = xs[0]->dtype_;
    Device device = xs[0]->device_;
    for (auto& t : xs) {
        Validator::input(t, std::string(op) + ".t").non_null();
        if (t->dtype_ != dt)
            throw DtypeMismatch(std::string(dtype_name(dt)), std::string(dtype_name(t->dtype_)),
                                std::string(op));
        if (t->device_ != device)
            throw DeviceMismatch(std::string(device_name(device)),
                                 std::string(device_name(t->device_)), std::string(op));
    }
}

inline int wrap_axis(int axis, int ndim) {
    int a = axis;
    if (a < 0)
        a += ndim;
    if (a < 0 || a >= ndim)
        ErrorBuilder("axis").fail("out of range");
    return a;
}

}  // namespace lucid::utils_detail
