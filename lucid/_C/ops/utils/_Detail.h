// lucid/_C/ops/utils/_Detail.h
//
// Internal helpers shared across the ops/utils subsystem.  Nothing in this
// header is part of the public API; it is included only by the .cpp files in
// this directory.  Keeping these utilities in one place avoids repeating the
// same boilerplate (numel wrapper, batch-validation, axis normalisation) in
// every translation unit.

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

// All symbols in this namespace are intended for consumption only inside
// ops/utils/*.cpp.  External callers should use the public op free functions
// declared in each module's header.
namespace lucid::utils_detail {

// Bring frequently-used helpers into this namespace so callers can write
// unqualified names after a single `using` declaration.
using ::lucid::gpu::mlx_shape_to_lucid;
using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

// Thin wrapper around shape_numel for brevity in ops that need element counts.
// Returns the product of all dimension sizes, or 1 for a 0-D tensor.
inline std::size_t numel(const Shape& s) {
    return shape_numel(s);
}

// Verify that every tensor in xs shares the same dtype and device, raising the
// appropriate typed error on the first mismatch.  Also rejects an empty list
// because all multi-tensor ops require at least one input.
//
// `op` is a C-string label used to prefix error messages (e.g. "concatenate").
// Throws DtypeMismatch or DeviceMismatch on the first pair that disagrees.
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

// Normalise axis to a non-negative index in [0, ndim).  Negative values are
// wrapped by adding ndim (Python semantics).  Raises an index error if the
// result falls outside the valid range.
//
// `axis` may be any signed integer; `ndim` must be positive.
// Returns a value in [0, ndim).  Throws via ErrorBuilder on out-of-range input.
inline int wrap_axis(int axis, int ndim) {
    int a = axis;
    if (a < 0)
        a += ndim;
    if (a < 0 || a >= ndim)
        ErrorBuilder("axis").fail("out of range");
    return a;
}

}  // namespace lucid::utils_detail
