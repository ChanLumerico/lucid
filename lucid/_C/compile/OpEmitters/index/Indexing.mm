// lucid/_C/compile/OpEmitters/index/Indexing.mm
//
// Data-dependent value-selection emitters:
//
//   - ``where``       — element-wise select via predicate
//   - ``masked_fill`` — where(mask, full(v), x) shortcut

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

class WhereEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "where"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3 || node.outputs.empty())
            return false;
        TensorId c_id = node.inputs[0];
        TensorId x_id = node.inputs[1];
        TensorId y_id = node.inputs[2];
        if (c_id < 0 || x_id < 0 || y_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* c_t = (__bridge MPSGraphTensor*)ctx.resolve(c_id);
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* y_t = (__bridge MPSGraphTensor*)ctx.resolve(y_id);
        if (graph == nil || c_t == nil || x_t == nil || y_t == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)([graph selectWithPredicateTensor:c_t
                                            truePredicateTensor:x_t
                                           falsePredicateTensor:y_t
                                                           name:@"where"]));
        return true;
    }
};

class MaskedFillEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "masked_fill"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return false;
        TensorId a_id = node.inputs[0];
        TensorId m_id = node.inputs[1];
        if (a_id < 0 || m_id < 0)
            return false;
        auto it = node.attrs.find("fill_value");
        if (it == node.attrs.end())
            return false;
        const auto* vp = std::get_if<double>(&it->second);
        if (vp == nullptr)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a_t = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
        MPSGraphTensor* m_t = (__bridge MPSGraphTensor*)ctx.resolve(m_id);
        if (graph == nil || a_t == nil || m_t == nil)
            return false;
        MPSGraphTensor* fill =
            [graph constantWithScalar:*vp dataType:a_t.dataType];
        ctx.bind(node.outputs[0].id, (__bridge void*)([graph selectWithPredicateTensor:m_t
                                            truePredicateTensor:fill
                                           falsePredicateTensor:a_t
                                                           name:@"masked_fill"]));
        return true;
    }
};

struct IndexingEmitterRegistrar {
    IndexingEmitterRegistrar() {
        register_emitter(std::make_unique<WhereEmitter>());
        register_emitter(std::make_unique<MaskedFillEmitter>());
    }
};

[[maybe_unused]] static const IndexingEmitterRegistrar g_indexing_registrar;

}  // namespace

}  // namespace lucid::compile
