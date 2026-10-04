// lucid/_C/compile/OpEmitters/index/Scatter.mm
//
// Axis-style scatter family — ``scatter_add`` / ``scatter_amax`` /
// ``scatter_amin`` / ``scatter_prod`` / ``scatter_set``.  All route through
// MPSGraph's ``scatterAlongAxisTensor:`` (SDK 13+) which matches
// Lucid / reference-framework semantics
// ``base[..., indices[i], ...] op= src[..., i, ...]`` (``=`` for set).
// An out-of-range index drops its update (policy B, ``../_IndexBounds.h``).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "../_AttrHelpers.h"
#include "../_IndexBounds.h"

namespace lucid::compile {

namespace {

// A reducing scatter under policy B: each out-of-range update becomes the
// reduction's identity, aimed at the safe (in-range) index.
inline MPSGraphTensor* scatter_reduce_dropping(MPSGraph* g,
                                               MPSGraphTensor* base,
                                               const GraphAxisIndex& ix,
                                               MPSGraphTensor* src,
                                               NSInteger axis,
                                               GraphScatter op) {
    MPSGraphTensor* updates = graph_drop_out_of_range(g, src, ix.in_range, op);
    if (updates == nil)
        return nil;
    MPSGraphScatterMode mode = MPSGraphScatterModeAdd;
    switch (op) {
    case GraphScatter::Add:
        mode = MPSGraphScatterModeAdd;
        break;
    case GraphScatter::Max:
        mode = MPSGraphScatterModeMax;
        break;
    case GraphScatter::Min:
        mode = MPSGraphScatterModeMin;
        break;
    case GraphScatter::Mul:
        mode = MPSGraphScatterModeMul;
        break;
    }
    return [g scatterAlongAxis:axis
                withDataTensor:base
                 updatesTensor:updates
                 indicesTensor:ix.safe
                          mode:mode
                          name:@"scatter_axis"];
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
        if (!shapes_agree_off_axis(base, idx, axis) || !shapes_agree_off_axis(src, idx, -1))
            return false;
        // MPSGraph scatters int64 data in 32 bits — -7 came back as
        // 4294967289 — and bool data as all false.  Neither narrows safely,
        // so both stay eager; float and int32 compile.
        for (MPSGraphTensor* t : {base, src})
            if (t.dataType == MPSDataTypeInt64 || t.dataType == MPSDataTypeBool) return false;
        if (![g respondsToSelector:@selector(scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:)]) {
            return false;
        }
        const std::int64_t extent = static_extent(base, axis);
        const GraphAxisIndex ix = graph_axis_index(g, idx, extent, NegativeIndex::Wrap);
        if (ix.safe == nil) return false;
        MPSGraphTensor* out = nil;
        switch (MODE) {
            case 0: out = scatter_reduce_dropping(g, base, ix, src, axis, GraphScatter::Add); break;
            case 1: out = scatter_reduce_dropping(g, base, ix, src, axis, GraphScatter::Max); break;
            case 2: out = scatter_reduce_dropping(g, base, ix, src, axis, GraphScatter::Min); break;
            case 3: out = scatter_reduce_dropping(g, base, ix, src, axis, GraphScatter::Mul); break;
            case 4: out = graph_scatter_set_dropping(g, base, ix, src, axis, extent); break;
            default: return false;
        }
        if (out == nil) return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)out);
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
