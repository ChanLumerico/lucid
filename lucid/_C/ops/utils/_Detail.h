#pragma once

#include <cstring>
#include <vector>

#include <mlx/array.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../core/fwd.h"

namespace lucid::utils_detail {

using ::lucid::gpu::mlx_shape_to_lucid;
using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

inline std::size_t numel(const Shape& s) {
    return shape_numel(s);
}

inline void check_dtype_device_match(const std::vector<TensorImplPtr>& xs, const char* op) {
    if (xs.empty())
        ErrorBuilder(op).fail("empty input");
    Dtype dt = xs[0]->dtype();
    Device device = xs[0]->device();
    for (auto& t : xs) {
        Validator::input(t, std::string(op) + ".t").non_null();
        if (t->dtype() != dt)
            throw DtypeMismatch(std::string(dtype_name(dt)), std::string(dtype_name(t->dtype())),
                                std::string(op));
        if (t->device() != device)
            throw DeviceMismatch(std::string(device_name(device)),
                                 std::string(device_name(t->device())), std::string(op));
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
