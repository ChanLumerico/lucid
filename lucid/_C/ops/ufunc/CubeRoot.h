// lucid/_C/ops/ufunc/CubeRoot.h
//
// Autograd backward node and entry point for the cube-root operation: y = x^(1/3).

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Backward node for element-wise cube root: y = cbrt(x) = x^(1/3).
//
// Gradient rule: dL/dx = dL/dy / (3 * y^2).
// Saves the *output* y rather than the input to avoid re-running cbrt in the
// backward pass; squaring y is cheaper and numerically equivalent.
// kSavesInput = false opts out of the default input-save in UnaryKernel.
// ForceFP32 is used because cbrt is numerically unreliable in half precision.
class LUCID_API CubeRootBackward : public UnaryOp<CubeRootBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cube_root(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr cube_root_op(const TensorImplPtr& a);

}  // namespace lucid
