#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

class LUCID_API PowBackward : public BinaryOp<PowBackward> {
public:
    static const OpSchema schema_v1;

    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.pow(a, b, shape, dt);
    }

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
};

LUCID_API TensorImplPtr pow_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
