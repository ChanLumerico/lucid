// lucid/_C/compile/OpEmitters/index/Gather.mm
//
// Data-dependent gather emitters:
//
//   - ``gather``    (lucid/_C/ops/utils/Select.cpp) — narrow gather
//                   matching index shape to the source rank, selects
//                   along ``axis`` only.  MPSGraph's
//                   ``gatherAlongAxis:`` mirrors the contract exactly.
//   - ``embedding`` (lucid/_C/ops/utils/Select.cpp) — gather along
//                   axis=0 with arbitrary-rank indices, then zero every
//                   row selected at ``padding_idx``.

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
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return false;
        TensorId data_id = node.inputs[0];
        TensorId idx_id = node.inputs[1];
        if (data_id < 0 || idx_id < 0)
            return false;

        auto it = node.attrs.find("axis");
        if (it == node.attrs.end())
            return false;
        const auto* axp = std::get_if<std::int64_t>(&it->second);
        if (axp == nullptr)
            return false;
        const NSUInteger axis = static_cast<NSUInteger>(*axp);

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* data_t = (__bridge MPSGraphTensor*)ctx.resolve(data_id);
        MPSGraphTensor* idx_t = (__bridge MPSGraphTensor*)ctx.resolve(idx_id);
        if (graph == nil || data_t == nil || idx_t == nil)
            return false;

        ctx.bind(node.outputs[0].id, (__bridge void*)([graph gatherAlongAxis:static_cast<NSInteger>(axis)
                                   withUpdatesTensor:data_t
                                       indicesTensor:idx_t
                                                name:@"gather"]));
        return true;
    }
};

// embedding — gather along axis=0 with arbitrary-rank indices.
class EmbeddingEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "embedding"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty())
            return false;
        TensorId w_id = node.inputs[0];
        TensorId i_id = node.inputs[1];
        if (w_id < 0 || i_id < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* w_t = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* i_t = (__bridge MPSGraphTensor*)ctx.resolve(i_id);
        if (graph == nil || w_t == nil || i_t == nil)
            return false;
        MPSGraphTensor* out = [graph gatherWithUpdatesTensor:w_t
                                              indicesTensor:i_t
                                                       axis:0
                                            batchDimensions:0
                                                     name:@"embedding"];

        // ``padding_idx`` is part of the op's contract, not a hint: the eager
        // kernel zeroes every row gathered at that index, so a compiled graph
        // that only gathers silently hands the pad token a real embedding.
        // The mismatch is invisible whenever the weight's pad row happens to
        // be zero, which is why it survived — a freshly initialised table
        // usually has it zeroed already.
        auto pad_it = node.attrs.find("padding_idx");
        if (pad_it != node.attrs.end()) {
            const auto* pad_p = std::get_if<std::int64_t>(&pad_it->second);
            if (pad_p != nullptr && *pad_p >= 0) {
                MPSGraphTensor* pad_c =
                    [graph constantWithScalar:static_cast<double>(*pad_p)
                                     dataType:i_t.dataType];
                // (indices != pad) -> 1/0, cast to the output dtype and
                // broadcast over the trailing embedding dimension.
                MPSGraphTensor* keep = [graph notEqualWithPrimaryTensor:i_t
                                                       secondaryTensor:pad_c
                                                                  name:@"emb_pad_ne"];
                keep = [graph castTensor:keep
                                  toType:out.dataType
                                    name:@"emb_pad_mask"];
                keep = [graph expandDimsOfTensor:keep
                                            axis:-1
                                            name:@"emb_pad_mask_bc"];
                out = [graph multiplicationWithPrimaryTensor:out
                                             secondaryTensor:keep
                                                        name:@"emb_pad_apply"];
            }
        }

        ctx.bind(node.outputs[0].id, (__bridge void*)out);
        return true;
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
