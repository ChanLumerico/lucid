#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

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
