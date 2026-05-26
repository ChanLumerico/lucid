// lucid/_C/compile/VjpEmitters/elementwise/Softmax.mm
//
// Softmax / LogSoftmax VJPs.
//
// Both have closed-form gradients that use the *output* y:
//
//   softmax:      dx = y * (grad - sum(grad * y, axis=dim, keepdim=true))
//   log_softmax:  dx = grad - exp(y) * sum(grad, axis=dim, keepdim=true)
//
// We recompute y from x (cheap, fp32) rather than save it on the
// BackwardContext side — keeps the API symmetric with the
// activation VJPs.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

inline std::int64_t softmax_dim(const OpNode& node) {
    auto it = node.attrs.find("dim");
    if (it == node.attrs.end()) return -1;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : -1;
}

class SoftmaxVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "softmax"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        const std::int64_t dim = softmax_dim(node);
        if (dim < 0) return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [dim](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* grad) {
                MPSGraphTensor* y =
                    [g softMaxWithTensor:x axis:(NSInteger)dim name:nil];
                MPSGraphTensor* s =
                    [g multiplicationWithPrimaryTensor:grad secondaryTensor:y name:nil];
                NSArray<NSNumber*>* axes = @[ [NSNumber numberWithLongLong:dim] ];
                MPSGraphTensor* s_sum =
                    [g reductionSumWithTensor:s axes:axes name:nil];
                MPSGraphTensor* diff =
                    [g subtractionWithPrimaryTensor:grad secondaryTensor:s_sum name:nil];
                return [g multiplicationWithPrimaryTensor:y
                                            secondaryTensor:diff
                                                       name:@"softmax_vjp"];
            });
    }
};

class LogSoftmaxVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "log_softmax"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        const std::int64_t dim = softmax_dim(node);
        if (dim < 0) return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [dim](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* grad) {
                MPSGraphTensor* y =
                    [g softMaxWithTensor:x axis:(NSInteger)dim name:nil];
                NSArray<NSNumber*>* axes = @[ [NSNumber numberWithLongLong:dim] ];
                MPSGraphTensor* g_sum =
                    [g reductionSumWithTensor:grad axes:axes name:nil];
                MPSGraphTensor* y_gsum =
                    [g multiplicationWithPrimaryTensor:y secondaryTensor:g_sum name:nil];
                return [g subtractionWithPrimaryTensor:grad
                                         secondaryTensor:y_gsum
                                                    name:@"log_softmax_vjp"];
            });
    }
};

struct SoftmaxVjpRegistrar {
    SoftmaxVjpRegistrar() {
        register_vjp_emitter(std::make_unique<SoftmaxVjp>());
        register_vjp_emitter(std::make_unique<LogSoftmaxVjp>());
    }
};

[[maybe_unused]] static const SoftmaxVjpRegistrar g_softmax_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
