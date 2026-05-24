// lucid/_C/compile/OpEmitters/Softmax.mm
//
// Softmax + LogSoftmax emitters.
//
// Op names match the engine schemas in lucid/_C/ops/ufunc/Softmax.cpp.
// Both ops carry a single ``dim`` (int64) attribute reported by the
// forward via :func:`OpScopeFull::set_attr`.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

// Pull the ``dim`` (int64) attribute; returns -1 on miss / wrong variant.
inline std::int64_t dim_attr(const OpNode& node) {
    auto it = node.attrs.find("dim");
    if (it == node.attrs.end())
        return -1;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : -1;
}

class SoftmaxEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "softmax"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        const std::int64_t dim = dim_attr(node);
        if (dim < 0)
            return nullptr;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return nullptr;
        MPSGraphTensor* y =
            [graph softMaxWithTensor:x_t axis:static_cast<NSInteger>(dim) name:@"softmax"];
        return (__bridge void*)y;
    }
};

class LogSoftmaxEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "log_softmax"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        const std::int64_t dim = dim_attr(node);
        if (dim < 0)
            return nullptr;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return nullptr;
        // log_softmax = log(softmax(x, axis)).  MPSGraph has no direct
        // builder, so compose; numerical-stability is handled inside
        // softMaxWithTensor.
        MPSGraphTensor* sm =
            [graph softMaxWithTensor:x_t axis:static_cast<NSInteger>(dim) name:@"log_softmax_sm"];
        MPSGraphTensor* y = [graph logarithmWithTensor:sm name:@"log_softmax"];
        return (__bridge void*)y;
    }
};

struct SoftmaxEmitterRegistrar {
    SoftmaxEmitterRegistrar() {
        register_emitter(std::make_unique<SoftmaxEmitter>());
        register_emitter(std::make_unique<LogSoftmaxEmitter>());
    }
};

[[maybe_unused]] static const SoftmaxEmitterRegistrar g_softmax_registrar;

}  // namespace

}  // namespace lucid::compile
