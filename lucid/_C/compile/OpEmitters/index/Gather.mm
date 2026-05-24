// lucid/_C/compile/OpEmitters/index/Gather.mm
//
// Data-dependent gather emitters:
//
//   - ``gather``    (lucid/_C/ops/utils/Select.cpp) — narrow gather
//                   matching index shape to the source rank, selects
//                   along ``axis`` only.  MPSGraph's
//                   ``gatherAlongAxis:`` mirrors the contract exactly.
//   - ``embedding`` (lucid/_C/ops/utils/Select.cpp) — gather along
//                   axis=0 with arbitrary-rank indices; equivalent to
//                   ``gather(weight, indices, axis=0)``.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

class GatherEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "gather"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return nullptr;
        TensorId data_id = node.inputs[0];
        TensorId idx_id = node.inputs[1];
        if (data_id < 0 || idx_id < 0)
            return nullptr;

        auto it = node.attrs.find("axis");
        if (it == node.attrs.end())
            return nullptr;
        const auto* axp = std::get_if<std::int64_t>(&it->second);
        if (axp == nullptr)
            return nullptr;
        const NSUInteger axis = static_cast<NSUInteger>(*axp);

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* data_t = (__bridge MPSGraphTensor*)ctx.resolve(data_id);
        MPSGraphTensor* idx_t = (__bridge MPSGraphTensor*)ctx.resolve(idx_id);
        if (graph == nil || data_t == nil || idx_t == nil)
            return nullptr;

        return (__bridge void*)[graph gatherAlongAxis:static_cast<NSInteger>(axis)
                                   withUpdatesTensor:data_t
                                       indicesTensor:idx_t
                                                name:@"gather"];
    }
};

// embedding — gather along axis=0 with arbitrary-rank indices.
class EmbeddingEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "embedding"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty())
            return nullptr;
        TensorId w_id = node.inputs[0];
        TensorId i_id = node.inputs[1];
        if (w_id < 0 || i_id < 0)
            return nullptr;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* w_t = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* i_t = (__bridge MPSGraphTensor*)ctx.resolve(i_id);
        if (graph == nil || w_t == nil || i_t == nil)
            return nullptr;
        return (__bridge void*)[graph gatherWithUpdatesTensor:w_t
                                                indicesTensor:i_t
                                                         axis:0
                                              batchDimensions:0
                                                         name:@"embedding"];
    }
};

struct GatherEmitterRegistrar {
    GatherEmitterRegistrar() {
        register_emitter(std::make_unique<GatherEmitter>());
        register_emitter(std::make_unique<EmbeddingEmitter>());
    }
};

[[maybe_unused]] static const GatherEmitterRegistrar g_gather_registrar;

}  // namespace

}  // namespace lucid::compile
