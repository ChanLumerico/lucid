// lucid/_C/compile/VjpEmitters/elementwise/Math.mm
//
// VJPs for the unary math ops (log / exp / sqrt / abs / square /
// reciprocal / sin / cos).  All single-input, all closed-form
// gradients with no broadcast unreduce.
//
// Forward emitters live in :file:`OpEmitters/elementwise/Math.mm`,
// :file:`OpEmitters/elementwise/Trig.mm`, etc.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string>
#include <string_view>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// d(log(x))/dx = 1/x
class LogVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "log"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                return [g divisionWithPrimaryTensor:go
                                     secondaryTensor:x
                                                name:@"log_vjp"];
            });
    }
};

// d(exp(x))/dx = exp(x).  Recomputed (fast) rather than saved.
class ExpVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "exp"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* y = [g exponentWithTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:y
                                                     name:@"exp_vjp"];
            });
    }
};

// d(sqrt(x))/dx = 1 / (2 sqrt(x))
class SqrtVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sqrt"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* y = [g squareRootWithTensor:x name:nil];
                MPSGraphTensor* two = [g constantWithScalar:2.0
                                                    dataType:x.dataType];
                MPSGraphTensor* two_y =
                    [g multiplicationWithPrimaryTensor:two
                                       secondaryTensor:y name:nil];
                return [g divisionWithPrimaryTensor:go
                                     secondaryTensor:two_y
                                                name:@"sqrt_vjp"];
            });
    }
};

// d(rsqrt(x))/dx = -1 / (2 * x^(3/2)) = -0.5 * rsqrt(x)^3
class RsqrtVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "rsqrt"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* r = [g reciprocalSquareRootWithTensor:x name:nil];
                MPSGraphTensor* r3a =
                    [g multiplicationWithPrimaryTensor:r secondaryTensor:r name:nil];
                MPSGraphTensor* r3 =
                    [g multiplicationWithPrimaryTensor:r3a secondaryTensor:r name:nil];
                MPSGraphTensor* nhalf = [g constantWithScalar:-0.5 dataType:x.dataType];
                MPSGraphTensor* d =
                    [g multiplicationWithPrimaryTensor:nhalf secondaryTensor:r3 name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:d
                                                     name:@"rsqrt_vjp"];
            });
    }
};

// d(square(x))/dx = 2x
class SquareVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "square"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* two = [g constantWithScalar:2.0 dataType:x.dataType];
                MPSGraphTensor* two_x =
                    [g multiplicationWithPrimaryTensor:two secondaryTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:two_x
                                                     name:@"square_vjp"];
            });
    }
};

// d(abs(x))/dx = sign(x)
class AbsVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "abs"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* s = [g signWithTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:s
                                                     name:@"abs_vjp"];
            });
    }
};

// d(reciprocal(x))/dx = -1/x²
class ReciprocalVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "reciprocal"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* x_sq =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
                MPSGraphTensor* neg_go = [g negativeWithTensor:go name:nil];
                return [g divisionWithPrimaryTensor:neg_go
                                     secondaryTensor:x_sq
                                                name:@"reciprocal_vjp"];
            });
    }
};

// d(sin(x))/dx = cos(x); d(cos(x))/dx = -sin(x)
class SinVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sin"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* c = [g cosWithTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:c
                                                     name:@"sin_vjp"];
            });
    }
};

class CosVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "cos"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* s = [g sinWithTensor:x name:nil];
                MPSGraphTensor* neg_s = [g negativeWithTensor:s name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:neg_s
                                                     name:@"cos_vjp"];
            });
    }
};

struct MathVjpRegistrar {
    MathVjpRegistrar() {
        register_vjp_emitter(std::make_unique<LogVjp>());
        register_vjp_emitter(std::make_unique<ExpVjp>());
        register_vjp_emitter(std::make_unique<SqrtVjp>());
        register_vjp_emitter(std::make_unique<RsqrtVjp>());
        register_vjp_emitter(std::make_unique<SquareVjp>());
        register_vjp_emitter(std::make_unique<AbsVjp>());
        register_vjp_emitter(std::make_unique<ReciprocalVjp>());
        register_vjp_emitter(std::make_unique<SinVjp>());
        register_vjp_emitter(std::make_unique<CosVjp>());
    }
};

[[maybe_unused]] static const MathVjpRegistrar g_math_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
