// lucid/_C/compile/OpEmitters/index/Scatter.mm
//
// Axis-style scatter family — ``scatter_add`` / ``scatter_amax`` /
// ``scatter_amin`` / ``scatter_prod`` / ``scatter_set``.  All route through
// MPSGraph's ``scatterAlongAxisTensor:`` (SDK 13+) which matches
// Lucid / reference-framework semantics
// ``base[..., indices[i], ...] op= src[..., i, ...]`` (``=`` for set).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

// MPSGraph's along-axis gather and scatter want the index to match the data
// on every axis but ``axis`` (and the updates to match the index).  The
// reference — and Lucid's eager kernels — allow an index shorter there, and
// MPSGraph handed one aborted the process: "updates shape and indices shape
// must match except at axis".  Such a call is declined and runs eagerly.
inline bool shapes_agree(MPSGraphTensor* a, MPSGraphTensor* b, NSInteger skip_axis) {
    NSArray<NSNumber*>* sa = a.shape;
    NSArray<NSNumber*>* sb = b.shape;
    if (sa == nil || sb == nil || sa.count != sb.count)
        return false;
    for (NSUInteger i = 0; i < sa.count; ++i) {
        if (static_cast<NSInteger>(i) == skip_axis)
            continue;
        if (sa[i].longLongValue != sb[i].longLongValue)
            return false;
    }
    return true;
}

// MODE: 0=Add 1=Max 2=Min 3=Mul 4=Set.
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
        const NSInteger axis = static_cast<NSInteger>(dim < 0 ? dim + (std::int64_t)base.shape.count : dim);
        if (!shapes_agree(base, idx, axis) || !shapes_agree(src, idx, -1)) return false;
        // MPSGraph scatters int64 data in 32 bits — -7 came back as
        // 4294967289 — and bool data as all false.  Neither narrows safely,
        // so both stay eager; float and int32 compile.
        for (MPSGraphTensor* t : {base, src})
            if (t.dataType == MPSDataTypeInt64 || t.dataType == MPSDataTypeBool) return false;
        MPSGraphScatterMode mode;
        switch (MODE) {
            case 0: mode = MPSGraphScatterModeAdd; break;
            case 1: mode = MPSGraphScatterModeMax; break;
            case 2: mode = MPSGraphScatterModeMin; break;
            case 3: mode = MPSGraphScatterModeMul; break;
            case 4: mode = MPSGraphScatterModeSet; break;
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
        register_emitter(std::make_unique<ScatterEmitterT<4>>("scatter_set"));
    }
};

[[maybe_unused]] static const ScatterEmitterRegistrar g_scatter_registrar;

}  // namespace

}  // namespace lucid::compile
