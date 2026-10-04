// lucid/_C/compile/OpEmitters/index/Gather.mm
//
// Data-dependent gather emitters:
//
//   - ``gather``    (lucid/_C/ops/utils/Select.cpp) — narrow gather
//                   matching index shape to the source rank, selects
//                   along ``axis`` only.  MPSGraph's
//                   ``gatherAlongAxis:`` mirrors the contract exactly.
//   - ``embedding`` (lucid/_C/nn/Embedding.cpp) — gather along axis=0
//                   with arbitrary-rank indices.  ``padding_idx`` rides
//                   on the node and is deliberately not applied; see the
//                   note in ``emit``.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"
#include "../_IndexBounds.h"

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
        if (!shapes_agree_off_axis(data_t, idx_t, static_cast<NSInteger>(axis)))
            return false;
        const GraphAxisIndex ix =
            graph_axis_index(graph, idx_t, static_extent(data_t, static_cast<NSInteger>(axis)),
                             NegativeIndex::Wrap);
        if (ix.safe == nil)
            return false;
        MPSGraphTensor* out = graph_fill_out_of_range(
            graph,
            [graph gatherAlongAxis:static_cast<NSInteger>(axis)
                 withUpdatesTensor:data_t
                     indicesTensor:ix.safe
                              name:@"gather"],
            ix.in_range);
        if (out == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)out);
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
        // A row outside the table, negative ones included, reads NaN
        // (policy B, ``_IndexBounds.h``): eager's ``gpu_take`` with
        // ``NegativeIndex::OutOfRange``.
        const GraphAxisIndex ix =
            graph_axis_index(graph, i_t, static_extent(w_t, 0), NegativeIndex::OutOfRange);
        if (ix.safe == nil)
            return false;
        MPSGraphTensor* rows = [graph gatherWithUpdatesTensor:w_t
                                                indicesTensor:ix.safe
                                                         axis:0
                                              batchDimensions:0
                                                         name:@"embedding"];
        // ``in_range`` has the index's shape; a row is the table's
        // remaining axes, so the mask gains one unit axis per one of them.
        NSMutableArray<NSNumber*>* mask_shape = [ix.in_range.shape mutableCopy];
        for (NSUInteger d = 1; d < w_t.shape.count; ++d)
            [mask_shape addObject:@1];
        MPSGraphTensor* out = graph_fill_out_of_range(
            graph, rows, [graph reshapeTensor:ix.in_range withShape:mask_shape name:nil]);
        if (out == nil)
            return false;

        // ``padding_idx`` does not mask the gather, here or in eager.
        //
        // This used to, and the comment that stood here said the eager
        // kernel zeroed the pad row so a compiled graph had to match.
        // That was true and both were wrong: the convention zeroes the
        // row at initialisation and stops its gradient, leaving the
        // lookup alone, so a trained checkpoint returns whatever it
        // learned there. BERT's [PAD] carries a real vector, and zeroing
        // it put bert_base 4.14 away from the implementation its weights
        // came from.
        //
        // Left as a note rather than deleted because the pairing is the
        // point: eager and compiled have to agree, and the way to keep
        // them agreeing is to fix the contract in one place and follow
        // it in the other — which is what happened, twice, in opposite
        // directions.

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
