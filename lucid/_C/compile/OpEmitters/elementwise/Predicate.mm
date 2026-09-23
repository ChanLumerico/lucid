// lucid/_C/compile/OpEmitters/elementwise/Predicate.mm
//
// Element-wise predicate emitters: isfinite / isnan / isinf.  All
// three are single-input ops producing a Bool tensor with the same
// shape as the input — used most prominently by
// :class:`lucid.amp.GradScaler`'s fused-step overflow detection
// path, where ``isfinite(unscaled_grad)`` is reduced to a single
// found_inf scalar fully inside the executable.
//
// MPSGraph exposes the matching primitives directly:
//   - ``isFiniteWithTensor:name:``
//   - ``isNaNWithTensor:name:``
//   - ``isInfiniteWithTensor:name:``
//
// No autograd path here — these ops are grad-sinks (the C++ VJP
// walker lists them under :func:`no_grad_ops` so no manual VJP is
// required).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline bool emit_predicate(BuilderContext& ctx, const OpNode& node,
                            BuilderBlock builder) {
    if (node.inputs.size() != 1 || node.outputs.empty())
        return false;
    TensorId x_id = node.inputs[0];
    if (x_id < 0)
        return false;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
    if (graph == nil || x_t == nil)
        return false;
    // Eager answers these for integer and bool tensors too (always
    // finite, never NaN or infinite).  MPSGraph's predicates take floating
    // point only and throw an Objective-C exception at compile time on
    // anything else, which reached Python as "Caught an unknown
    // exception!".  Casting an integer or bool to float32 cannot produce a
    // NaN or an infinity — int64's range sits far inside float32's — so
    // the predicate on the cast gives eager's answer.
    if (!(x_t.dataType & MPSDataTypeFloatBit))
        x_t = [graph castTensor:x_t toType:MPSDataTypeFloat32 name:nil];
    ctx.bind(node.outputs[0].id, (__bridge void*)(builder(graph, x_t)));
    return true;
}

class IsFiniteEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "isfinite"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_predicate(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g isFiniteWithTensor:x name:@"isfinite"];
        });
    }
};

class IsNanEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "isnan"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_predicate(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g isNaNWithTensor:x name:@"isnan"];
        });
    }
};

class IsInfEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "isinf"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_predicate(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g isInfiniteWithTensor:x name:@"isinf"];
        });
    }
};

struct PredicateEmitterRegistrar {
    PredicateEmitterRegistrar() {
        register_emitter(std::make_unique<IsFiniteEmitter>());
        register_emitter(std::make_unique<IsNanEmitter>());
        register_emitter(std::make_unique<IsInfEmitter>());
    }
};

[[maybe_unused]] static const PredicateEmitterRegistrar g_predicate_registrar;

}  // namespace

}  // namespace lucid::compile
