// lucid/_C/ops/ufunc/CubeRoot.cpp
//
// Gradient formula and entry point for the cube-root op.
// ForceFP32 is used to keep the backend numerically stable; cbrt in
// float16 can lose precision for large or small exponents.

#include "CubeRoot.h"

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

// ForceFP32: cbrt is not numerically safe in half precision.
const OpSchema CubeRootBackward::schema_v1{"cube_root", 1, AmpPolicy::ForceFP32, true};

// dL/dx = dL/dy / (3 * y^2), where y = saved_output_ = cbrt(x).
// Squaring the saved output is cheaper than recomputing cbrt(x)^2 from
// scratch and avoids a second cbrt call into the backend.
Storage CubeRootBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage y_sq = square_storage(saved_output_, n, dtype_, device_);
    Storage denom = mul_scalar_storage(y_sq, 3.0, n, dtype_, device_);
    return divide_storages(g, denom, n, dtype_, device_);
}

TensorImplPtr cube_root_op(const TensorImplPtr& a) {
    return CubeRootBackward::forward(a);
}
LUCID_REGISTER_OP(CubeRootBackward)

}  // namespace lucid
