// lucid/_C/ops/ufunc/Activation.h
//
// Autograd backward nodes and entry points for neural-network activation
// functions: relu, sigmoid, silu/swish, gelu, leaky_relu, softplus, elu, selu,
// mish, hard_sigmoid, hard_swish, relu6.
//
// Most activations delegate the analytically complex backward computation to
// the backend dispatcher (IBackend::{activation}_backward) so that CPU can use
// Apple Accelerate and GPU can use MLX.  Simpler activations (relu, sigmoid,
// relu6) implement grad_formula directly using storage primitives for clarity.
//
// Ops with a scalar hyper-parameter (leaky_relu slope, elu alpha) override the
// standard static forward() from UnaryKernel rather than using dispatch(), so
// they can capture and persist the parameter in the backward node.

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Backward node for ReLU: y = max(0, x).
//
// Gradient rule: dL/dx = (x > 0) * dL/dy  (a binary mask multiplied element-
// wise with the upstream gradient).
// Saves the input to build the positive-mask in grad_formula.
class LUCID_API ReluBackward : public UnaryOp<ReluBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.relu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
    // dx = (x > 0) * g — uses relu_op to get the 0/1 mask
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Backward node for logistic sigmoid: y = 1 / (1 + e^{-x}).
//
// Gradient rule: dL/dx = y*(1-y) * dL/dy.
// Saves the *output* y because the backward formula only needs y, not x.
class LUCID_API SigmoidBackward : public UnaryOp<SigmoidBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sigmoid(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
    // dx = out*(1-out)*g  where out = sigmoid(x)
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr& out);
};

// Backward node for SiLU/Swish: y = x * sigmoid(x).
//
// Gradient rule: dL/dx = sigmoid(x) * (1 + x*(1 - sigmoid(x))) * dL/dy.
// Saves the input; grad_formula recomputes sigmoid(x) from the saved value.
class LUCID_API SiluBackward : public UnaryOp<SiluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.silu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for GeLU (Gaussian Error Linear Unit).
//
// The exact GeLU backward is non-trivial (involves the Gaussian CDF and PDF);
// it is delegated entirely to the backend dispatcher for CPU/GPU portability.
// Saves the input for the backward call.
class LUCID_API GeluBackward : public UnaryOp<GeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.gelu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for Leaky ReLU: y = x if x > 0, else slope * x.
//
// Gradient rule: dL/dx = (x > 0 ? 1 : slope) * dL/dy  (a leaky mask).
// slope_ is persisted on the node so that grad_formula can build the mask.
// Because slope requires a custom forward(), the standard dispatch() path is
// not used; instead forward() and cpu_kernel() are overridden explicitly.
class LUCID_API LeakyReluBackward : public UnaryOp<LeakyReluBackward> {
public:
    static constexpr bool kSavesInput = true;
    double slope_ = 0.01;
    static const OpSchema schema_v1;
    // Override: captures slope and wires the backward node manually.
    static TensorImplPtr forward(const TensorImplPtr& a, double slope);
    static CpuStorage
    cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt, double slope);
    Storage grad_formula(const Storage& g);
};

// Backward node for Softplus: y = log(1 + e^x).
//
// Gradient rule: dL/dx = sigmoid(x) * dL/dy  (derivative of softplus is
// sigmoid).  Saves the input to evaluate sigmoid(x) in grad_formula.
class LUCID_API SoftplusBackward : public UnaryOp<SoftplusBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.softplus(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for ELU: y = x if x >= 0, else alpha*(e^x - 1).
//
// The piecewise backward depends on both x and alpha; it is delegated to the
// backend.  alpha_ is saved on the node for the backward call.
// Like LeakyReluBackward, uses an explicit forward() that captures alpha_.
class LUCID_API EluBackward : public UnaryOp<EluBackward> {
public:
    static constexpr bool kSavesInput = true;
    double alpha_ = 1.0;
    static const OpSchema schema_v1;
    // Override: captures alpha and wires the backward node manually.
    static TensorImplPtr forward(const TensorImplPtr& a, double alpha);
    static CpuStorage
    cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt, double alpha);
    Storage grad_formula(const Storage& g);
};

// Backward node for SELU (Scaled ELU).
//
// SELU uses fixed α ≈ 1.6733 and λ ≈ 1.0507; the backward is delegated to the
// backend since the piecewise formula involves these constants.
class LUCID_API SeluBackward : public UnaryOp<SeluBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.selu(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for Mish: y = x * tanh(softplus(x)).
//
// The backward involves both tanh and its derivative through a composed
// function; it is delegated to the backend.
class LUCID_API MishBackward : public UnaryOp<MishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.mish(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for Hard Sigmoid: y = clamp((x + 3) / 6, 0, 1).
//
// Gradient rule: dL/dx = 1/6 if -3 < x < 3, else 0.  The backend computes
// this mask via hard_sigmoid_backward.
class LUCID_API HardSigmoidBackward : public UnaryOp<HardSigmoidBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.hard_sigmoid(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for Hard Swish: y = x * hard_sigmoid(x).
//
// Gradient rule: dL/dx = hard_sigmoid(x) + x * d/dx[hard_sigmoid(x)]; the
// backend handles the piecewise linear derivative.
class LUCID_API HardSwishBackward : public UnaryOp<HardSwishBackward> {
public:
    static constexpr bool kSavesInput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.hard_swish(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for ReLU6: y = clamp(x, 0, 6).
//
// Gradient rule: dL/dx = 1 if 0 < x < 6, else 0  (a boolean range mask
// multiplied element-wise with the upstream gradient).
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
