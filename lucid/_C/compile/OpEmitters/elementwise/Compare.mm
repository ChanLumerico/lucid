// lucid/_C/compile/OpEmitters/elementwise/Compare.mm
//
// Element-wise comparison emitters: equal / not_equal / greater /
// greater_equal / less / less_equal + bitwise invert.  All six
// comparison forms produce a Bool output and share a single
// MPSGraph builder per op.
//
// Op names match the engine schemas:
//   - equal / not_equal / greater / greater_equal / less / less_equal
//     (lucid/_C/ops/bfunc/Compare.cpp) — share ``cmp_dispatch``
//   - invert (lucid/_C/ops/bfunc/Predicate.cpp) — bitwise NOT, also
//     acts as logical NOT for Bool dtype.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline void* emit_cmp(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.size() != 2 || node.outputs.empty())
        return nullptr;
    TensorId a_id = node.inputs[0];
    TensorId b_id = node.inputs[1];
    if (a_id < 0 || b_id < 0)
        return nullptr;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* a_t = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
    MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
    if (graph == nil || a_t == nil || b_t == nil)
        return nullptr;
    return (__bridge void*)builder(graph, a_t, b_t);
}

class EqualEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "equal"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g equalWithPrimaryTensor:a secondaryTensor:b name:@"equal"];
        });
    }
};

class NotEqualEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "not_equal"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g notEqualWithPrimaryTensor:a secondaryTensor:b name:@"not_equal"];
        });
    }
};

class GreaterEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "greater"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g greaterThanWithPrimaryTensor:a secondaryTensor:b name:@"greater"];
        });
    }
};

class GreaterEqualEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "greater_equal"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g greaterThanOrEqualToWithPrimaryTensor:a
                                            secondaryTensor:b
                                                       name:@"greater_equal"];
        });
    }
};

class LessEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "less"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g lessThanWithPrimaryTensor:a secondaryTensor:b name:@"less"];
        });
    }
};

class LessEqualEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "less_equal"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g lessThanOrEqualToWithPrimaryTensor:a
                                         secondaryTensor:b
                                                    name:@"less_equal"];
        });
    }
};

// invert — bitwise NOT (also acts as logical NOT for Bool dtype).
class InvertEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "invert"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return nullptr;
        return (__bridge void*)[graph bitwiseNOTWithTensor:x_t name:@"invert"];
    }
};

struct CompareEmitterRegistrar {
    CompareEmitterRegistrar() {
        register_emitter(std::make_unique<EqualEmitter>());
        register_emitter(std::make_unique<NotEqualEmitter>());
        register_emitter(std::make_unique<GreaterEmitter>());
        register_emitter(std::make_unique<GreaterEqualEmitter>());
        register_emitter(std::make_unique<LessEmitter>());
        register_emitter(std::make_unique<LessEqualEmitter>());
        register_emitter(std::make_unique<InvertEmitter>());
    }
};

[[maybe_unused]] static const CompareEmitterRegistrar g_compare_registrar;

}  // namespace

}  // namespace lucid::compile
