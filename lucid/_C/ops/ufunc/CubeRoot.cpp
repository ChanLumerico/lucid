#include "CubeRoot.h"

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema CubeRootBackward::schema_v1{"cube_root", 1, AmpPolicy::ForceFP32, true};

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
