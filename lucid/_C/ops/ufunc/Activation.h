#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

class LUCID_API ReluBackward : public UnaryOp<ReluBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.relu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API SigmoidBackward : public UnaryOp<SigmoidBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sigmoid(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API SiluBackward : public UnaryOp<SiluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.silu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API GeluBackward : public UnaryOp<GeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.gelu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API LeakyReluBackward : public UnaryOp<LeakyReluBackward> {
public:
    static constexpr bool kSavesInput = true;
    double slope_ = 0.01;
    static const OpSchema schema_v1;
    static TensorImplPtr forward(const TensorImplPtr& a, double slope);
    static CpuStorage
    cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt, double slope);
    Storage grad_formula(const Storage& g);
};

class LUCID_API SoftplusBackward : public UnaryOp<SoftplusBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.softplus(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API EluBackward : public UnaryOp<EluBackward> {
public:
    static constexpr bool kSavesInput = true;
    double alpha_ = 1.0;
    static const OpSchema schema_v1;
    static TensorImplPtr forward(const TensorImplPtr& a, double alpha);
    static CpuStorage
    cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt, double alpha);
    Storage grad_formula(const Storage& g);
};

class LUCID_API SeluBackward : public UnaryOp<SeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.selu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API MishBackward : public UnaryOp<MishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.mish(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API HardSigmoidBackward : public UnaryOp<HardSigmoidBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.hard_sigmoid(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API HardSwishBackward : public UnaryOp<HardSwishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.hard_swish(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

class LUCID_API Relu6Backward : public UnaryOp<Relu6Backward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.relu6(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr relu_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr sigmoid_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr silu_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr gelu_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr leaky_relu_op(const TensorImplPtr& a, double slope);

LUCID_API TensorImplPtr softplus_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr elu_op(const TensorImplPtr& a, double alpha);

LUCID_API TensorImplPtr selu_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr mish_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr hard_sigmoid_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr hard_swish_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr relu6_op(const TensorImplPtr& a);

}  // namespace lucid
