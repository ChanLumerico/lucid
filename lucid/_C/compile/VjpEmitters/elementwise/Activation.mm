// lucid/_C/compile/VjpEmitters/elementwise/Activation.mm
//
// Activation VJPs.  All single-input ops — straight elementwise
// gradient formulas, no broadcast unreduce needed.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// All activation VJPs use the unary skeleton from _VjpHelpers.h:
//   emit_unary_vjp(bctx, node, grad_outs, body) → bool
// where ``body(graph, x, grad_out) → MPSGraphTensor*`` returns dx.

// ────────────────────────────────────────────────────────────────────
// ReLU: dA = grad * (x > 0).
// ────────────────────────────────────────────────────────────────────
class ReluVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "relu"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = x.dataType;
                MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
                MPSGraphTensor* mask =
                    [g greaterThanWithPrimaryTensor:x secondaryTensor:zero name:nil];
                MPSGraphTensor* mask_f = [g castTensor:mask toType:dt name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:mask_f
                                                     name:@"relu_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// Sigmoid: dA = grad * y * (1 - y) where y = sigmoid(x).
// We recompute y from x; same cost as saving it but avoids a cache
// slot on the bctx side.
// ────────────────────────────────────────────────────────────────────
class SigmoidVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sigmoid"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* y = [g sigmoidWithTensor:x name:nil];
                MPSDataType dt = y.dataType;
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* one_m_y =
                    [g subtractionWithPrimaryTensor:one secondaryTensor:y name:nil];
                MPSGraphTensor* y_omy =
                    [g multiplicationWithPrimaryTensor:y secondaryTensor:one_m_y name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:y_omy
                                                     name:@"sigmoid_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// Tanh: dA = grad * (1 - y²) where y = tanh(x).
// ────────────────────────────────────────────────────────────────────
class TanhVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "tanh"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* y = [g tanhWithTensor:x name:nil];
                MPSDataType dt = y.dataType;
                MPSGraphTensor* y_sq =
                    [g multiplicationWithPrimaryTensor:y secondaryTensor:y name:nil];
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* one_m_y_sq =
                    [g subtractionWithPrimaryTensor:one secondaryTensor:y_sq name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:one_m_y_sq
                                                     name:@"tanh_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// SiLU: y = x * sigmoid(x).  dy/dx = sigmoid(x) * (1 + x*(1-sigmoid(x))).
// ────────────────────────────────────────────────────────────────────
class SiluVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "silu"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = x.dataType;
                MPSGraphTensor* sig = [g sigmoidWithTensor:x name:nil];
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* one_m_sig =
                    [g subtractionWithPrimaryTensor:one secondaryTensor:sig name:nil];
                MPSGraphTensor* x_omsig =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:one_m_sig name:nil];
                MPSGraphTensor* one_plus =
                    [g additionWithPrimaryTensor:one secondaryTensor:x_omsig name:nil];
                MPSGraphTensor* deriv =
                    [g multiplicationWithPrimaryTensor:sig secondaryTensor:one_plus name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:deriv
                                                     name:@"silu_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// GELU (tanh-approx):
//   y = 0.5 * x * (1 + tanh( k * (x + 0.044715 * x^3) ))   k = sqrt(2/π).
// Derivative (Hendrycks-Gimpel form):
//   let u = k*(x + 0.044715*x^3),  t = tanh(u),  s = sech²(u) = 1 - t².
//   dy/dx = 0.5 * (1 + t) + 0.5 * x * s * k * (1 + 3 * 0.044715 * x²)
//          = 0.5 * (1 + t) + 0.5 * x * (1 - t²) * k * (1 + 0.134145 * x²).
// ────────────────────────────────────────────────────────────────────
class GeluVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "gelu"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = x.dataType;
                MPSGraphTensor* k =
                    [g constantWithScalar:0.7978845608028654 dataType:dt];  // sqrt(2/π)
                MPSGraphTensor* c044 =
                    [g constantWithScalar:0.044715 dataType:dt];
                MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:dt];
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* c134 =
                    [g constantWithScalar:0.134145 dataType:dt];  // 3 * 0.044715
                // x²
                MPSGraphTensor* x2 =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
                // x³
                MPSGraphTensor* x3 =
                    [g multiplicationWithPrimaryTensor:x2 secondaryTensor:x name:nil];
                // 0.044715 * x³
                MPSGraphTensor* cx3 =
                    [g multiplicationWithPrimaryTensor:c044 secondaryTensor:x3 name:nil];
                // x + 0.044715 * x³
                MPSGraphTensor* inner_add =
                    [g additionWithPrimaryTensor:x secondaryTensor:cx3 name:nil];
                // u = k * (x + 0.044715 * x³)
                MPSGraphTensor* u =
                    [g multiplicationWithPrimaryTensor:k secondaryTensor:inner_add name:nil];
                MPSGraphTensor* t = [g tanhWithTensor:u name:nil];
                MPSGraphTensor* t2 =
                    [g multiplicationWithPrimaryTensor:t secondaryTensor:t name:nil];
                MPSGraphTensor* sech2 =
                    [g subtractionWithPrimaryTensor:one secondaryTensor:t2 name:nil];
                // first half: 0.5 * (1 + t)
                MPSGraphTensor* one_plus_t =
                    [g additionWithPrimaryTensor:one secondaryTensor:t name:nil];
                MPSGraphTensor* term1 =
                    [g multiplicationWithPrimaryTensor:half secondaryTensor:one_plus_t name:nil];
                // second half: 0.5 * x * (1-t²) * k * (1 + 0.134145 * x²)
                MPSGraphTensor* c134_x2 =
                    [g multiplicationWithPrimaryTensor:c134 secondaryTensor:x2 name:nil];
                MPSGraphTensor* one_plus_c134x2 =
                    [g additionWithPrimaryTensor:one secondaryTensor:c134_x2 name:nil];
                MPSGraphTensor* k_term =
                    [g multiplicationWithPrimaryTensor:k secondaryTensor:one_plus_c134x2 name:nil];
                MPSGraphTensor* x_k_term =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:k_term name:nil];
                MPSGraphTensor* x_k_sech2 =
                    [g multiplicationWithPrimaryTensor:x_k_term secondaryTensor:sech2 name:nil];
                MPSGraphTensor* term2 =
                    [g multiplicationWithPrimaryTensor:half secondaryTensor:x_k_sech2 name:nil];
                // sum + multiply by grad
                MPSGraphTensor* deriv =
                    [g additionWithPrimaryTensor:term1 secondaryTensor:term2 name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:deriv
                                                     name:@"gelu_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// astype (dtype cast): dx = cast(grad_out, x.dtype).  The forward
// changes precision; the backward casts the grad back to the input's
// precision so the chain stays self-consistent.  Identity-shape so
// no broadcast unreduce is needed.
// ────────────────────────────────────────────────────────────────────
class AstypeVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "astype"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType x_dt = x.dataType;
                if (go.dataType == x_dt) return go;
                return [g castTensor:go toType:x_dt name:@"astype_vjp"];
            });
    }
};

struct ActivationVjpRegistrar {
    ActivationVjpRegistrar() {
        register_vjp_emitter(std::make_unique<ReluVjp>());
        register_vjp_emitter(std::make_unique<SigmoidVjp>());
        register_vjp_emitter(std::make_unique<TanhVjp>());
        register_vjp_emitter(std::make_unique<SiluVjp>());
        register_vjp_emitter(std::make_unique<GeluVjp>());
        register_vjp_emitter(std::make_unique<AstypeVjp>());
    }
};

[[maybe_unused]] static const ActivationVjpRegistrar g_activation_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
