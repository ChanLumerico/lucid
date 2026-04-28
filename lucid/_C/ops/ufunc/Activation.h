#pragma once

// =====================================================================
// Lucid C++ engine — activation unary ops.
// =====================================================================
//
//   relu(x)   = max(x, 0)        grad: g * (x > 0)
//
// More activations (sigmoid, gelu, silu, leaky_relu, elu, softplus, softmax)
// land in Phase 3.5 alongside the NN op family.

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

class LUCID_API ReluBackward : public UnaryOp<ReluBackward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API SigmoidBackward : public UnaryOp<SigmoidBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;  // grad uses z (1-z)
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API SiluBackward : public UnaryOp<SiluBackward> {
public:
    static constexpr bool kSavesInput = true;  // need x for grad = σ(1 + x(1-σ))
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API GeluBackward : public UnaryOp<GeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

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

class LUCID_API SoftplusBackward : public UnaryOp<SoftplusBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

// ELU(x; α) = x if x >= 0 else α (exp(x) - 1)
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
class LUCID_API SeluBackward : public UnaryOp<SeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

// Mish(x) = x * tanh(softplus(x))
class LUCID_API MishBackward : public UnaryOp<MishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

// HardSigmoid(x) = clip((x + 3) / 6, 0, 1)
class LUCID_API HardSigmoidBackward : public UnaryOp<HardSigmoidBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

// HardSwish(x) = x * HardSigmoid(x)
class LUCID_API HardSwishBackward : public UnaryOp<HardSwishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

// ReLU6(x) = clip(x, 0, 6)
class LUCID_API Relu6Backward : public UnaryOp<Relu6Backward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
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
