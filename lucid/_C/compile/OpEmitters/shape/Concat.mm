// lucid/_C/compile/OpEmitters/Concat.mm
//
// Concatenate / Stack emitters.  Variadic input arity — inputs[i] for
// i ∈ [0, N).  ``dim`` int64 attribute reported by the forward via
// ``OpScopeFull::set_attr``.
//
// Engine schemas (lucid/_C/ops/utils/Concat.cpp):
//   - "concatenate" — concatTensors:withDimension:name:
//   - "stack"       — concatTensors over a new axis (handled by Lucid
//                     engine; MPSGraph has no direct stack builder so
//                     we emit reshape+concat).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

inline std::int64_t dim_attr(const OpNode& node) {
    auto it = node.attrs.find("dim");
    if (it == node.attrs.end())
        return -1;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : -1;
}

class ConcatEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "concatenate"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty())
            return false;
        const std::int64_t dim = dim_attr(node);
        if (dim < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        if (graph == nil)
            return false;

        NSMutableArray<MPSGraphTensor*>* tensors =
            [NSMutableArray arrayWithCapacity:node.inputs.size()];
        for (TensorId iid : node.inputs) {
            if (iid < 0)
                return false;
            MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(iid);
            if (t == nil)
                return false;
            [tensors addObject:t];
        }
        MPSGraphTensor* y = [graph concatTensors:tensors
                                       dimension:(NSInteger)dim
                                            name:@"concatenate"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// stack(xs, axis) = expand_dims(xs[i], axis) then concat along axis.
// MPSGraph has no native stack, so we unsqueeze each input at ``axis``
// and concat.  The output shape carries the new size-N axis already
// (set by stack_op).
class StackEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "stack"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return false;
        auto ax_it = node.attrs.find("axis");
        if (ax_it == node.attrs.end())
            return false;
        const auto* axp = std::get_if<std::int64_t>(&ax_it->second);
        if (axp == nullptr)
            return false;
        const std::int64_t ax = *axp;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        if (graph == nil)
            return false;

        // Unsqueeze each input at ``ax`` so the new axis can be the
        // concat dimension.
        NSMutableArray<MPSGraphTensor*>* expanded =
            [NSMutableArray arrayWithCapacity:node.inputs.size()];
        for (TensorId iid : node.inputs) {
            if (iid < 0)
                return false;
            MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(iid);
            if (t == nil)
                return false;
            NSArray<NSNumber*>* src_shape = t.shape;
            NSMutableArray<NSNumber*>* new_shape =
                [NSMutableArray arrayWithCapacity:src_shape.count + 1];
            for (NSUInteger d = 0; d < src_shape.count; ++d) {
                if ((std::int64_t)d == ax)
                    [new_shape addObject:[NSNumber numberWithLongLong:1]];
                [new_shape addObject:src_shape[d]];
            }
            if ((std::int64_t)src_shape.count == ax)
                [new_shape addObject:[NSNumber numberWithLongLong:1]];
            MPSGraphTensor* expanded_t =
                [graph reshapeTensor:t withShape:new_shape name:nil];
            [expanded addObject:expanded_t];
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)([graph concatTensors:expanded
                                          dimension:(NSInteger)ax
                                               name:@"stack"]));
        return true;
    }
};

struct ConcatEmitterRegistrar {
    ConcatEmitterRegistrar() {
        register_emitter(std::make_unique<ConcatEmitter>());
        register_emitter(std::make_unique<StackEmitter>());
    }
};

[[maybe_unused]] static const ConcatEmitterRegistrar g_concat_registrar;

}  // namespace

}  // namespace lucid::compile
