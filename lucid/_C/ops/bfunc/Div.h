#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

class LUCID_API DivBackward : public BinaryOp<DivBackward> {
public:
    static const OpSchema schema_v1;

    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.div(a, b, shape, dt);
    }

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

LUCID_API TensorImplPtr div_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
