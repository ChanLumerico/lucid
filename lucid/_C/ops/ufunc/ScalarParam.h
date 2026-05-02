#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

class LUCID_API PowScalarBackward : public UnaryOp<PowScalarBackward> {
public:
    double exp_ = 0.0;
    static const OpSchema schema_v1;
    static TensorImplPtr forward(const TensorImplPtr& a, double exp);
    Storage grad_formula(const Storage& g);
};

class LUCID_API RPowScalarBackward : public UnaryOp<RPowScalarBackward> {
public:
    double base_ = 0.0;

    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static TensorImplPtr forward(double base, const TensorImplPtr& a);
    Storage grad_formula(const Storage& g);
};

class LUCID_API ClipBackward : public UnaryOp<ClipBackward> {
public:
    double min_ = 0.0;
    double max_ = 0.0;
    static const OpSchema schema_v1;
    static TensorImplPtr forward(const TensorImplPtr& a, double min_v, double max_v);
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr pow_scalar_op(const TensorImplPtr& a, double exp);

LUCID_API TensorImplPtr rpow_scalar_op(double base, const TensorImplPtr& a);

LUCID_API TensorImplPtr clip_op(const TensorImplPtr& a, double min_v, double max_v);

}  // namespace lucid
