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

class LUCID_API NegBackward : public UnaryOp<NegBackward> {
public:
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.neg(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API AbsBackward : public UnaryOp<AbsBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.abs(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API SignBackward : public UnaryOp<SignBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sign(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API ReciprocalBackward : public UnaryOp<ReciprocalBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.reciprocal(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API SquareBackward : public UnaryOp<SquareBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.square(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API CubeBackward : public UnaryOp<CubeBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cube(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr neg_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr abs_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr sign_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr reciprocal_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr square_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr cube_op(const TensorImplPtr& a);

}  // namespace lucid
