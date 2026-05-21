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

// Compute the total element count of a shape.
//
// Thin wrapper around :func:`shape_numel` provided so call sites in
// ops/utils/*.cpp can use the shorter ``numel(s)`` form.  Returns ``1`` for
// 0-D (scalar) shapes by convention.
//
// Parameters
// ----------
// s : Shape
//     Shape vector (length == rank).
//
// Returns
// -------
// size_t
//     Product of ``s[d]`` for all dimensions; ``1`` if ``s`` is empty.
inline std::size_t numel(const Shape& s) {
    return shape_numel(s);
}

// Validate that a batch of tensors share dtype and device.
//
// Iterates ``xs`` once and raises on the first mismatching dtype or device.
// Also rejects empty input (multi-tensor utility ops require at least one
// argument).  Each tensor is null-checked via ``Validator::input``.
//
// Parameters
// ----------
// xs : vector<TensorImplPtr>
//     Non-empty list of input tensors.
// op : const char*
//     Operation label used to prefix error messages (e.g. ``"concatenate"``).
//
// Raises
// ------
// ErrorBuilder
//     If ``xs`` is empty.
// DtypeMismatch
//     If any tensor's dtype differs from ``xs[0]``'s.
// DeviceMismatch
//     If any tensor's device differs from ``xs[0]``'s.
//
// Notes
// -----
// Shape compatibility is *not* checked here — each caller has its own
// axis-specific shape contract (see e.g. :func:`concatenate_op`).
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

// Normalise a possibly negative axis to a non-negative index.
//
// Wraps negative ``axis`` by adding ``ndim`` (Python semantics) and verifies
// the result lies in ``[0, ndim)``.  Used by every ops/utils function that
// accepts an axis parameter.
//
// Parameters
// ----------
// axis : int
//     Signed axis index; negative values count from the end.
// ndim : int
//     Number of dimensions of the target tensor.  Must be > 0.
//
// Returns
// -------
// int
//     A non-negative axis in ``[0, ndim)``.
//
// Raises
// ------
// ErrorBuilder
//     If the wrapped axis falls outside ``[0, ndim)``.
inline int wrap_axis(int axis, int ndim) {
    int a = axis;
    if (a < 0)
        a += ndim;
    if (a < 0 || a >= ndim)
        ErrorBuilder("axis").fail("out of range");
    return a;
}

}  // namespace lucid::utils_detail
