// lucid/_C/compile/OpEmitters/TrigHyperbolic.mm
//
// Five single-tensor unary trig / hyperbolic emitters.
//
// Op names match the engine schemas in:
//   - lucid/_C/ops/ufunc/Trig.cpp       ("sin", "cos", "tan")
//   - lucid/_C/ops/ufunc/Hyperbolic.cpp ("sinh", "cosh")
//
// All five are 1-input → 1-output UnaryKernel ops with direct
// MPSGraph 1-builders, no attribute payload required.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

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
    return (__bridge void*)builder(graph, x_t);
}

class SinEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sin"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g sinWithTensor:x name:@"sin"];
        });
    }
};

class CosEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cos"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g cosWithTensor:x name:@"cos"];
        });
    }
};

class TanEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "tan"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g tanWithTensor:x name:@"tan"];
        });
    }
};

class SinhEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sinh"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g sinhWithTensor:x name:@"sinh"];
        });
    }
};

class CoshEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cosh"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g coshWithTensor:x name:@"cosh"];
        });
    }
};

struct TrigHyperbolicEmitterRegistrar {
    TrigHyperbolicEmitterRegistrar() {
        register_emitter(std::make_unique<SinEmitter>());
        register_emitter(std::make_unique<CosEmitter>());
        register_emitter(std::make_unique<TanEmitter>());
        register_emitter(std::make_unique<SinhEmitter>());
        register_emitter(std::make_unique<CoshEmitter>());
    }
};

[[maybe_unused]] static const TrigHyperbolicEmitterRegistrar g_trig_hyperbolic_registrar;

}  // namespace

}  // namespace lucid::compile
