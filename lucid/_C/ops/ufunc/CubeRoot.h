#pragma once

// Cube root: x^(1/3), grad = g/(3 * cbrt(x)^2)

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

/// Autograd backward node for CubeRoot.
class LUCID_API CubeRootBackward : public UnaryOp<CubeRootBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a,
                            const Shape& s, Dtype dt) {
        // TODO: call be.<method>(a, s, dt)
        (void)be; (void)a; (void)s; (void)dt;
        ErrorBuilder(schema_v1.name).not_implemented("dispatch not yet implemented");
    }
    Storage grad_formula(const Storage& g);
};

/// Cube root.
LUCID_API TensorImplPtr cube_root_op(const TensorImplPtr& a);

}  // namespace lucid
