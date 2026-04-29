#include "CubeRoot.h"

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema CubeRootBackward::schema_v1{
    "cube_root", 1, AmpPolicy::ForceFP32, /*deterministic=*/true};

Storage CubeRootBackward::grad_formula(const Storage& g) {
    // TODO: implement backward for cube_root
    (void)g;
    ErrorBuilder("cube_root").not_implemented("grad_formula not yet implemented");
}

TensorImplPtr cube_root_op(const TensorImplPtr& a) {
    return CubeRootBackward::forward(a);
}
LUCID_REGISTER_OP(CubeRootBackward)

}  // namespace lucid
