// lucid/_C/compile/OpEmitters/Cumulative.mm
//
// R3 — cumulative reductions: cumsum / cumprod / cummax / cummin.
// All four are 1-input → 1-output with an ``axis`` (int64) attribute
// set on the engine forward.  MPSGraph exposes each as a single
// builder (``cumulativeSumWithTensor:axis:...``).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline void* emit_scan(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.empty() || node.outputs.empty())
        return nullptr;
    TensorId x_id = node.inputs[0];
    if (x_id < 0)
        return nullptr;
    auto it = node.attrs.find("axis");
    if (it == node.attrs.end())
        return nullptr;
    const auto* axp = std::get_if<std::int64_t>(&it->second);
    if (axp == nullptr)
        return nullptr;

    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
    if (graph == nil || x_t == nil)
        return nullptr;
    return (__bridge void*)builder(graph, x_t, (NSInteger)*axp);
}

class CumsumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cumsum"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_scan(ctx, node, [](MPSGraph* g, MPSGraphTensor* x, NSInteger ax) {
            return [g cumulativeSumWithTensor:x axis:ax name:@"cumsum"];
        });
    }
};

class CumprodEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cumprod"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_scan(ctx, node, [](MPSGraph* g, MPSGraphTensor* x, NSInteger ax) {
            return [g cumulativeProductWithTensor:x axis:ax name:@"cumprod"];
        });
    }
};

class CummaxEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cummax"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_scan(ctx, node, [](MPSGraph* g, MPSGraphTensor* x, NSInteger ax) {
            return [g cumulativeMaximumWithTensor:x axis:ax name:@"cummax"];
        });
    }
};

class CumminEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cummin"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_scan(ctx, node, [](MPSGraph* g, MPSGraphTensor* x, NSInteger ax) {
            return [g cumulativeMinimumWithTensor:x axis:ax name:@"cummin"];
        });
    }
};

struct CumulativeEmitterRegistrar {
    CumulativeEmitterRegistrar() {
        register_emitter(std::make_unique<CumsumEmitter>());
        register_emitter(std::make_unique<CumprodEmitter>());
        register_emitter(std::make_unique<CummaxEmitter>());
        register_emitter(std::make_unique<CumminEmitter>());
    }
};

[[maybe_unused]] static const CumulativeEmitterRegistrar g_cum_registrar;

}  // namespace

}  // namespace lucid::compile
