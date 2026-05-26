// lucid/_C/compile/OpEmitters/Factory.mm
//
// Zero-input factory emitters that produce constant tensors directly
// inside the graph.  Currently:
//   - "full"  : MPSGraph constantWithScalar:shape:dataType: using the
//               ``fill_value`` (double) attribute reported by
//               ``ops/gfunc/Gfunc.cpp::full_op``.
//
// These emit at compile time, not at run time — the constant is baked
// into the executable rather than fed in.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

inline MPSDataType to_mps_dtype_local(Dtype dt) {
    switch (dt) {
        case Dtype::F16:
            return MPSDataTypeFloat16;
        case Dtype::I32:
            return MPSDataTypeInt32;
        case Dtype::I64:
            return MPSDataTypeInt64;
        case Dtype::Bool:
            return MPSDataTypeBool;
        case Dtype::F32:
        default:
            return MPSDataTypeFloat32;
    }
}

inline NSArray<NSNumber*>* shape_to_nsarray(const Shape& shape) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:shape.size()];
    for (std::int64_t d : shape)
        [out addObject:[NSNumber numberWithLongLong:d]];
    return out;
}

class FullEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "full"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        // full takes no traced inputs — the trace records 0 input ids.
        if (!node.inputs.empty())
            return false;
        if (node.outputs.empty())
            return false;

        auto it = node.attrs.find("fill_value");
        if (it == node.attrs.end())
            return false;
        const auto* v = std::get_if<double>(&it->second);
        if (v == nullptr)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        if (graph == nil)
            return false;

        const TensorMeta& meta = node.outputs[0];
        NSArray<NSNumber*>* ns_shape = shape_to_nsarray(meta.shape);
        MPSDataType ns_dt = to_mps_dtype_local(meta.dtype);
        MPSGraphTensor* y = [graph constantWithScalar:*v
                                                 shape:ns_shape
                                              dataType:ns_dt];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// ── zeros / ones — same constantWithScalar treatment as full but with
// a fixed scalar.  Previously these were stubbed (Stubs.mm) because the
// builder factory-skip pathway handled them when their output became an
// external feed — but when a `zeros()` output is consumed by another op
// within the same trace (e.g. RNN's default `hx = zeros(...)` then used
// as the recurrent state), the factory-skip never fires and we'd hit a
// nullptr emit fallback.  Real-emit covers both cases.
template <int FILL_VALUE>
class FixedFillEmitterT final : public OpEmitter {
public:
    explicit FixedFillEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (!node.inputs.empty())
            return false;
        if (node.outputs.empty())
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        if (graph == nil)
            return false;
        const TensorMeta& meta = node.outputs[0];
        NSArray<NSNumber*>* ns_shape = shape_to_nsarray(meta.shape);
        MPSDataType ns_dt = to_mps_dtype_local(meta.dtype);
        MPSGraphTensor* y = [graph constantWithScalar:(double)FILL_VALUE
                                                 shape:ns_shape
                                              dataType:ns_dt];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }

private:
    std::string name_;
};

struct FactoryEmitterRegistrar {
    FactoryEmitterRegistrar() {
        register_emitter(std::make_unique<FullEmitter>());
        register_emitter(std::make_unique<FixedFillEmitterT<0>>("zeros"));
        register_emitter(std::make_unique<FixedFillEmitterT<1>>("ones"));
    }
};

[[maybe_unused]] static const FactoryEmitterRegistrar g_factory_registrar;

}  // namespace

}  // namespace lucid::compile
