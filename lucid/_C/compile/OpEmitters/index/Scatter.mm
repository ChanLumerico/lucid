// lucid/_C/compile/OpEmitters/index/Scatter.mm
//
// Axis-style scatter family — ``scatter_add`` / ``scatter_amax`` /
// ``scatter_amin`` / ``scatter_prod``.  All four route through
// MPSGraph's ``scatterAlongAxisTensor:`` (SDK 13+) which matches
// Lucid / reference-framework semantics
// ``base[..., indices[i], ...] op= src[..., i, ...]``.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

// MODE: 0=Add 1=Max 2=Min 3=Mul.
template <int MODE>
class ScatterEmitterT final : public OpEmitter {
public:
    explicit ScatterEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 3 || node.outputs.empty()) return false;
        TensorId b_id = node.inputs[0];
        TensorId i_id = node.inputs[1];
        TensorId s_id = node.inputs[2];
        if (b_id < 0 || i_id < 0 || s_id < 0) return false;
        std::int64_t dim = int_attr(node, "dim", 0);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* base = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        MPSGraphTensor* idx = (__bridge MPSGraphTensor*)ctx.resolve(i_id);
        MPSGraphTensor* src = (__bridge MPSGraphTensor*)ctx.resolve(s_id);
        if (g == nil || base == nil || idx == nil || src == nil) return false;
        MPSGraphScatterMode mode;
        switch (MODE) {
            case 0: mode = MPSGraphScatterModeAdd; break;
            case 1: mode = MPSGraphScatterModeMax; break;
            case 2: mode = MPSGraphScatterModeMin; break;
            case 3: mode = MPSGraphScatterModeMul; break;
            default: return false;
        }
        if (![g respondsToSelector:@selector(scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:)]) {
            return false;
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)([g scatterAlongAxis:(NSInteger)dim
                                    withDataTensor:base
                                     updatesTensor:src
                                     indicesTensor:idx
                                              mode:mode
                                              name:@"scatter_axis"]));
        return true;
    }

private:
    std::string name_;
};

struct ScatterEmitterRegistrar {
    ScatterEmitterRegistrar() {
        register_emitter(std::make_unique<ScatterEmitterT<0>>("scatter_add"));
        register_emitter(std::make_unique<ScatterEmitterT<1>>("scatter_amax"));
        register_emitter(std::make_unique<ScatterEmitterT<2>>("scatter_amin"));
        register_emitter(std::make_unique<ScatterEmitterT<3>>("scatter_prod"));
    }
};

[[maybe_unused]] static const ScatterEmitterRegistrar g_scatter_registrar;

}  // namespace

}  // namespace lucid::compile
