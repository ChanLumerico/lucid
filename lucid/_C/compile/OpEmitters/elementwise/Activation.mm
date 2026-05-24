// lucid/_C/compile/OpEmitters/Activation.mm
//
// Single-tensor activation emitters: ReLU / Sigmoid / Tanh.
//
// Op names match the engine schemas defined in:
//   - lucid/_C/ops/ufunc/Activation.cpp ("relu", "sigmoid")
//   - lucid/_C/ops/ufunc/Hyperbolic.cpp ("tanh")
//
// All three are 1-input → 1-output ops; MPSGraph exposes them
// directly as single-call builders.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

// Helper: shared "resolve single input + dispatch one-call MPSGraph
// builder" boilerplate.  Each concrete emitter just supplies the
// builder block.
template <class BuilderBlock>
inline void* emit_unary(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.size() != 1)
        return nullptr;
    TensorId x_id = node.inputs[0];
    if (x_id < 0)
        return nullptr;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
    if (x_t == nil || graph == nil)
        return nullptr;
    MPSGraphTensor* y = builder(graph, x_t);
    return (__bridge void*)y;
}

class ReluEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "relu"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g reLUWithTensor:x name:@"relu"];
        });
    }
};

class SigmoidEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sigmoid"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g sigmoidWithTensor:x name:@"sigmoid"];
        });
    }
};

class TanhEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "tanh"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g tanhWithTensor:x name:@"tanh"];
        });
    }
};

// SiLU: y = x * sigmoid(x).  MPSGraph composes from primitives.
class SiluEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "silu"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            MPSGraphTensor* s = [g sigmoidWithTensor:x name:nil];
            return [g multiplicationWithPrimaryTensor:x secondaryTensor:s name:@"silu"];
        });
    }
};

// GELU tanh-approximation:
//   y = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
// Matches the engine's "gelu" schema (tanh approx).
class GeluEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "gelu"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            MPSDataType dt = x.dataType;
            MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:dt];
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor* a044 = [g constantWithScalar:0.044715 dataType:dt];
            MPSGraphTensor* k = [g constantWithScalar:0.7978845608028654 dataType:dt];  // sqrt(2/π)
            MPSGraphTensor* x2 =
                [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
            MPSGraphTensor* x3 =
                [g multiplicationWithPrimaryTensor:x2 secondaryTensor:x name:nil];
            MPSGraphTensor* ax3 =
                [g multiplicationWithPrimaryTensor:a044 secondaryTensor:x3 name:nil];
            MPSGraphTensor* inner_add =
                [g additionWithPrimaryTensor:x secondaryTensor:ax3 name:nil];
            MPSGraphTensor* inner =
                [g multiplicationWithPrimaryTensor:k secondaryTensor:inner_add name:nil];
            MPSGraphTensor* th = [g tanhWithTensor:inner name:nil];
            MPSGraphTensor* one_plus =
                [g additionWithPrimaryTensor:one secondaryTensor:th name:nil];
            MPSGraphTensor* half_x =
                [g multiplicationWithPrimaryTensor:half secondaryTensor:x name:nil];
            return
                [g multiplicationWithPrimaryTensor:half_x secondaryTensor:one_plus name:@"gelu"];
        });
    }
};

// GELU exact: y = 0.5 * x * (1 + erf(x / sqrt(2))).  Engine schema name "gelu_exact".
class GeluExactEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "gelu_exact"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            MPSDataType dt = x.dataType;
            MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:dt];
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor* inv_sqrt2 =
                [g constantWithScalar:0.7071067811865475 dataType:dt];  // 1/sqrt(2)
            MPSGraphTensor* x_scaled =
                [g multiplicationWithPrimaryTensor:x secondaryTensor:inv_sqrt2 name:nil];
            MPSGraphTensor* erf_t = [g erfWithTensor:x_scaled name:nil];
            MPSGraphTensor* one_plus =
                [g additionWithPrimaryTensor:one secondaryTensor:erf_t name:nil];
            MPSGraphTensor* half_x =
                [g multiplicationWithPrimaryTensor:half secondaryTensor:x name:nil];
            return [g multiplicationWithPrimaryTensor:half_x secondaryTensor:one_plus
                                                  name:@"gelu_exact"];
        });
    }
};

// R1 — additional activations: ELU / LeakyReLU / SELU / Mish /
// Softplus / HardSigmoid / HardSwish / ReLU6.
// ``alpha`` (elu) and ``slope`` (leaky_relu) come from the forward's
// ``set_attr`` payload; defaults match the Lucid scalar defaults.

class EluEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "elu"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [&](MPSGraph* g, MPSGraphTensor* x) {
            MPSDataType dt = x.dataType;
            double alpha = 1.0;
            auto it = node.attrs.find("alpha");
            if (it != node.attrs.end()) {
                if (const auto* p = std::get_if<double>(&it->second)) alpha = *p;
            }
            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor* a_const = [g constantWithScalar:alpha dataType:dt];
            MPSGraphTensor* ex = [g exponentWithTensor:x name:nil];
            MPSGraphTensor* exm1 =
                [g subtractionWithPrimaryTensor:ex secondaryTensor:one name:nil];
            MPSGraphTensor* neg_branch =
                [g multiplicationWithPrimaryTensor:a_const secondaryTensor:exm1 name:nil];
            MPSGraphTensor* mask =
                [g greaterThanOrEqualToWithPrimaryTensor:x secondaryTensor:zero name:nil];
            return [g selectWithPredicateTensor:mask
                            truePredicateTensor:x
                           falsePredicateTensor:neg_branch
                                           name:@"elu"];
        });
    }
};

class LeakyReluEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "leaky_relu"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [&](MPSGraph* g, MPSGraphTensor* x) {
            MPSDataType dt = x.dataType;
            double slope = 0.01;
            auto it = node.attrs.find("slope");
            if (it != node.attrs.end()) {
                if (const auto* p = std::get_if<double>(&it->second)) slope = *p;
            }
            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
            MPSGraphTensor* s = [g constantWithScalar:slope dataType:dt];
            MPSGraphTensor* sx =
                [g multiplicationWithPrimaryTensor:s secondaryTensor:x name:nil];
            MPSGraphTensor* mask =
                [g greaterThanOrEqualToWithPrimaryTensor:x secondaryTensor:zero name:nil];
            return [g selectWithPredicateTensor:mask
                            truePredicateTensor:x
                           falsePredicateTensor:sx
                                           name:@"leaky_relu"];
        });
    }
};

class SeluEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "selu"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            // SELU: λ · (x > 0 ? x : α(exp(x) - 1))   with α≈1.6733, λ≈1.0507.
            MPSDataType dt = x.dataType;
            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor* alpha =
                [g constantWithScalar:1.6732632423543772 dataType:dt];
            MPSGraphTensor* lambda =
                [g constantWithScalar:1.0507009873554805 dataType:dt];
            MPSGraphTensor* ex = [g exponentWithTensor:x name:nil];
            MPSGraphTensor* exm1 =
                [g subtractionWithPrimaryTensor:ex secondaryTensor:one name:nil];
            MPSGraphTensor* neg_branch =
                [g multiplicationWithPrimaryTensor:alpha secondaryTensor:exm1 name:nil];
            MPSGraphTensor* mask =
                [g greaterThanOrEqualToWithPrimaryTensor:x secondaryTensor:zero name:nil];
            MPSGraphTensor* sel =
                [g selectWithPredicateTensor:mask
                         truePredicateTensor:x
                        falsePredicateTensor:neg_branch
                                        name:nil];
            return [g multiplicationWithPrimaryTensor:lambda
                                       secondaryTensor:sel
                                                  name:@"selu"];
        });
    }
};

class MishEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "mish"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            // mish(x) = x · tanh(softplus(x)) = x · tanh(log(1 + e^x)).
            MPSDataType dt = x.dataType;
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor* ex = [g exponentWithTensor:x name:nil];
            MPSGraphTensor* ex1 =
                [g additionWithPrimaryTensor:one secondaryTensor:ex name:nil];
            MPSGraphTensor* sp = [g logarithmWithTensor:ex1 name:nil];
            MPSGraphTensor* th = [g tanhWithTensor:sp name:nil];
            return [g multiplicationWithPrimaryTensor:x
                                       secondaryTensor:th
                                                  name:@"mish"];
        });
    }
};

class SoftplusEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "softplus"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            // softplus(x) = log(1 + e^x) — beta=1 (Lucid only ships
            // the single-parameter form so no attr lookup needed).
            MPSDataType dt = x.dataType;
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor* ex = [g exponentWithTensor:x name:nil];
            MPSGraphTensor* ex1 =
                [g additionWithPrimaryTensor:one secondaryTensor:ex name:nil];
            return [g logarithmWithTensor:ex1 name:@"softplus"];
        });
    }
};

class HardSigmoidEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "hard_sigmoid"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            // hard_sigmoid(x) = clip((x + 3) / 6, 0, 1).
            MPSDataType dt = x.dataType;
            MPSGraphTensor* three = [g constantWithScalar:3.0 dataType:dt];
            MPSGraphTensor* six = [g constantWithScalar:6.0 dataType:dt];
            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor* shifted =
                [g additionWithPrimaryTensor:x secondaryTensor:three name:nil];
            MPSGraphTensor* scaled =
                [g divisionWithPrimaryTensor:shifted secondaryTensor:six name:nil];
            MPSGraphTensor* clipped_lo =
                [g maximumWithPrimaryTensor:scaled secondaryTensor:zero name:nil];
            return [g minimumWithPrimaryTensor:clipped_lo
                                secondaryTensor:one
                                           name:@"hard_sigmoid"];
        });
    }
};

class HardSwishEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "hard_swish"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            // hard_swish(x) = x · clip((x + 3) / 6, 0, 1).
            MPSDataType dt = x.dataType;
            MPSGraphTensor* three = [g constantWithScalar:3.0 dataType:dt];
            MPSGraphTensor* six = [g constantWithScalar:6.0 dataType:dt];
            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor* shifted =
                [g additionWithPrimaryTensor:x secondaryTensor:three name:nil];
            MPSGraphTensor* scaled =
                [g divisionWithPrimaryTensor:shifted secondaryTensor:six name:nil];
            MPSGraphTensor* clipped_lo =
                [g maximumWithPrimaryTensor:scaled secondaryTensor:zero name:nil];
            MPSGraphTensor* hs =
                [g minimumWithPrimaryTensor:clipped_lo secondaryTensor:one name:nil];
            return [g multiplicationWithPrimaryTensor:x
                                       secondaryTensor:hs
                                                  name:@"hard_swish"];
        });
    }
};

class Relu6Emitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "relu6"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            MPSDataType dt = x.dataType;
            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
            MPSGraphTensor* six = [g constantWithScalar:6.0 dataType:dt];
            MPSGraphTensor* lo =
                [g maximumWithPrimaryTensor:x secondaryTensor:zero name:nil];
            return [g minimumWithPrimaryTensor:lo
                                secondaryTensor:six
                                           name:@"relu6"];
        });
    }
};

// One file-scope registrar for the whole family — keeps the .o
// reachable from `--gc-sections` via a single anchor (the binding
// layer touches it under that name).
struct ActivationEmitterRegistrar {
    ActivationEmitterRegistrar() {
        register_emitter(std::make_unique<ReluEmitter>());
        register_emitter(std::make_unique<SigmoidEmitter>());
        register_emitter(std::make_unique<TanhEmitter>());
        register_emitter(std::make_unique<SiluEmitter>());
        register_emitter(std::make_unique<GeluEmitter>());
        register_emitter(std::make_unique<GeluExactEmitter>());
        // R1 additions.
        register_emitter(std::make_unique<EluEmitter>());
        register_emitter(std::make_unique<LeakyReluEmitter>());
        register_emitter(std::make_unique<SeluEmitter>());
        register_emitter(std::make_unique<MishEmitter>());
        register_emitter(std::make_unique<SoftplusEmitter>());
        register_emitter(std::make_unique<HardSigmoidEmitter>());
        register_emitter(std::make_unique<HardSwishEmitter>());
        register_emitter(std::make_unique<Relu6Emitter>());
    }
};

[[maybe_unused]] static const ActivationEmitterRegistrar g_activation_registrar;

}  // namespace

}  // namespace lucid::compile
