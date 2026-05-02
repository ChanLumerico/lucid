#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

class LUCID_API SinhBackward : public UnaryOp<SinhBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sinh(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API CoshBackward : public UnaryOp<CoshBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cosh(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API TanhBackward : public UnaryOp<TanhBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.tanh(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr sinh_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr cosh_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr tanh_op(const TensorImplPtr& a);

}  // namespace lucid
