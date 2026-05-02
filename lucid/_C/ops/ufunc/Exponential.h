#pragma once

#include <utility>

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

class LUCID_API ExpBackward : public UnaryOp<ExpBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.exp(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API LogBackward : public UnaryOp<LogBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.log(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API Log2Backward : public UnaryOp<Log2Backward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.log2(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API SqrtBackward : public UnaryOp<SqrtBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.sqrt(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr exp_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr log_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr log2_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr sqrt_op(const TensorImplPtr& a);

}  // namespace lucid
