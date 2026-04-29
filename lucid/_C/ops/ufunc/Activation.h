#pragma once

// =====================================================================
// Lucid C++ engine — activation unary ops.
// =====================================================================
//
//   relu(x)         = max(x, 0)
//   sigmoid(x)      = 1 / (1 + exp(-x))     [saves output]
//   silu(x)         = x * sigmoid(x)
//   gelu(x)         = 0.5x(1+tanh(c1(x+c2 x³)))
//   leaky_relu(x;s) = x if x>=0 else s*x     [scalar param — no dispatch]
//   softplus(x)     = log(1 + exp(x))
//   elu(x;α)        = x if x>=0 else α(eˣ-1) [scalar param — no dispatch]
//   selu(x)         = scale·(x if x>=0 else α(eˣ-1))
//   mish(x)         = x·tanh(softplus(x))
//   hard_sigmoid(x) = clip((x+3)/6, 0, 1)
//   hard_swish(x)   = x·hard_sigmoid(x)
//   relu6(x)        = clip(x, 0, 6)

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

/// Autograd backward node for Relu.
class LUCID_API ReluBackward : public UnaryOp<ReluBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.relu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

/// Autograd backward node for Sigmoid.
class LUCID_API SigmoidBackward : public UnaryOp<SigmoidBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;  // grad uses z (1-z)
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sigmoid(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

/// Autograd backward node for Silu.
class LUCID_API SiluBackward : public UnaryOp<SiluBackward> {
public:
    static constexpr bool kSavesInput = true;  // need x for grad = σ(1 + x(1-σ))
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.silu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

/// Autograd backward node for Gelu.
class LUCID_API GeluBackward : public UnaryOp<GeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.gelu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// LeakyRelu has a scalar param — custom forward/cpu_kernel, no dispatch().
/// Autograd backward node for LeakyRelu.
class LUCID_API LeakyReluBackward : public UnaryOp<LeakyReluBackward> {
public:
    static constexpr bool kSavesInput = true;
    double slope_ = 0.01;
    static const OpSchema schema_v1;
    static TensorImplPtr forward(const TensorImplPtr& a, double slope);
    static CpuStorage cpu_kernel(const CpuStorage& a,
                                 const Shape& out_shape,
                                 Dtype dt,
                                 double slope);
    Storage grad_formula(const Storage& g);
};

/// Autograd backward node for Softplus.
class LUCID_API SoftplusBackward : public UnaryOp<SoftplusBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.softplus(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// ELU(x; α) = x if x >= 0 else α (exp(x) - 1)  — scalar param, no dispatch().
/// Autograd backward node for Elu.
class LUCID_API EluBackward : public UnaryOp<EluBackward> {
public:
    static constexpr bool kSavesInput = true;
    double alpha_ = 1.0;
    static const OpSchema schema_v1;
    static TensorImplPtr forward(const TensorImplPtr& a, double alpha);
    static CpuStorage cpu_kernel(const CpuStorage& a,
                                 const Shape& out_shape,
                                 Dtype dt,
                                 double alpha);
    Storage grad_formula(const Storage& g);
};

// SELU(x) = scale * (x if x>=0 else α(exp(x)-1))
// scale=1.0507009873554805, α=1.6732632423543772 (fixed)
/// Autograd backward node for Selu.
class LUCID_API SeluBackward : public UnaryOp<SeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.selu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Mish(x) = x * tanh(softplus(x))
/// Autograd backward node for Mish.
class LUCID_API MishBackward : public UnaryOp<MishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.mish(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// HardSigmoid(x) = clip((x + 3) / 6, 0, 1)
/// Autograd backward node for HardSigmoid.
class LUCID_API HardSigmoidBackward : public UnaryOp<HardSigmoidBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.hard_sigmoid(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// HardSwish(x) = x * HardSigmoid(x)
/// Autograd backward node for HardSwish.
class LUCID_API HardSwishBackward : public UnaryOp<HardSwishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.hard_swish(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// ReLU6(x) = clip(x, 0, 6)
/// Autograd backward node for Relu6.
class LUCID_API Relu6Backward : public UnaryOp<Relu6Backward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.relu6(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

/// Relu.
LUCID_API TensorImplPtr relu_op(const TensorImplPtr& a);
/// Sigmoid.
LUCID_API TensorImplPtr sigmoid_op(const TensorImplPtr& a);
/// Silu.
LUCID_API TensorImplPtr silu_op(const TensorImplPtr& a);
/// Gelu.
LUCID_API TensorImplPtr gelu_op(const TensorImplPtr& a);
/// Leaky relu.
LUCID_API TensorImplPtr leaky_relu_op(const TensorImplPtr& a, double slope);
/// Softplus.
LUCID_API TensorImplPtr softplus_op(const TensorImplPtr& a);
/// Elu.
LUCID_API TensorImplPtr elu_op(const TensorImplPtr& a, double alpha);
/// Selu.
LUCID_API TensorImplPtr selu_op(const TensorImplPtr& a);
/// Mish.
LUCID_API TensorImplPtr mish_op(const TensorImplPtr& a);
/// Hard sigmoid.
LUCID_API TensorImplPtr hard_sigmoid_op(const TensorImplPtr& a);
/// Hard swish.
LUCID_API TensorImplPtr hard_swish_op(const TensorImplPtr& a);
/// Relu6.
LUCID_API TensorImplPtr relu6_op(const TensorImplPtr& a);

}  // namespace lucid
