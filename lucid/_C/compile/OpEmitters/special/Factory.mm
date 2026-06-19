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

#include <cstdint>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>

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
        MPSGraphTensor* y = [graph constantWithScalar:*v shape:ns_shape dataType:ns_dt];
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

// ── arange — bake the sequence ``start + i*step`` as a constant.  The
// generator parameters are static (recorded as ``start`` / ``step`` attrs by
// ``ops/gfunc/Gfunc.cpp::arange_op``; the length is the output dim), so the
// whole 1-D tensor is known at compile time.  This is what unblocks strided
// slicing (``x[..., ::2]`` lowers to ``gather`` over an ``arange`` index), so
// e.g. RoFormer's interleaved RoPE compiles instead of falling back to eager.
class ArangeEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "arange"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (!node.inputs.empty())
            return false;
        if (node.outputs.empty())
            return false;
        const TensorMeta& meta = node.outputs[0];
        if (meta.shape.size() != 1)
            return false;
        const std::int64_t n = meta.shape[0];
        if (n <= 0)
            return false;  // empty arange — let the rare case fall back

        auto its = node.attrs.find("start");
        auto itp = node.attrs.find("step");
        if (its == node.attrs.end() || itp == node.attrs.end())
            return false;
        const double* startp = std::get_if<double>(&its->second);
        const double* stepp = std::get_if<double>(&itp->second);
        if (startp == nullptr || stepp == nullptr)
            return false;
        const double start = *startp;
        const double step = *stepp;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        if (graph == nil)
            return false;
        NSArray<NSNumber*>* ns_shape = shape_to_nsarray(meta.shape);

        MPSGraphTensor* y = nil;
        switch (meta.dtype) {
        case Dtype::I32: {
            std::vector<std::int32_t> buf(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i)
                buf[static_cast<std::size_t>(i)] =
                    static_cast<std::int32_t>(start + static_cast<double>(i) * step);
            NSData* d = [NSData dataWithBytes:buf.data() length:buf.size() * sizeof(std::int32_t)];
            y = [graph constantWithData:d shape:ns_shape dataType:MPSDataTypeInt32];
            break;
        }
        case Dtype::I64: {
            std::vector<std::int64_t> buf(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i)
                buf[static_cast<std::size_t>(i)] =
                    static_cast<std::int64_t>(start + static_cast<double>(i) * step);
            NSData* d = [NSData dataWithBytes:buf.data() length:buf.size() * sizeof(std::int64_t)];
            y = [graph constantWithData:d shape:ns_shape dataType:MPSDataTypeInt64];
            break;
        }
        default: {
            // F16 / F32 / F64 — bake float32 and cast to the target type
            // (Metal has no f64; F32 is the highest-precision graph type).
            std::vector<float> buf(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i)
                buf[static_cast<std::size_t>(i)] =
                    static_cast<float>(start + static_cast<double>(i) * step);
            NSData* d = [NSData dataWithBytes:buf.data() length:buf.size() * sizeof(float)];
            y = [graph constantWithData:d shape:ns_shape dataType:MPSDataTypeFloat32];
            MPSDataType mdt = to_mps_dtype_local(meta.dtype);
            if (mdt != MPSDataTypeFloat32)
                y = [graph castTensor:y toType:mdt name:nil];
            break;
        }
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

struct FactoryEmitterRegistrar {
    FactoryEmitterRegistrar() {
        register_emitter(std::make_unique<FullEmitter>());
        register_emitter(std::make_unique<FixedFillEmitterT<0>>("zeros"));
        register_emitter(std::make_unique<FixedFillEmitterT<1>>("ones"));
        register_emitter(std::make_unique<ArangeEmitter>());
    }
};

[[maybe_unused]] static const FactoryEmitterRegistrar g_factory_registrar;

}  // namespace

}  // namespace lucid::compile
