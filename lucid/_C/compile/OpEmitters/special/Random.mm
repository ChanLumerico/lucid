// lucid/_C/compile/OpEmitters/special/Random.mm
//
// Stateless real-emit for the RNG family: rand / uniform / randn /
// normal / randint / bernoulli.
//
// Design trade-off
// ----------------
// MPSGraph offers two RNG paths:
//
//   1. ``randomTensorWithShape:descriptor:`` — seed baked into the
//      descriptor.  Every call to the compiled executable produces
//      the same sequence.  Deterministic per executable.
//
//   2. ``randomTensorWithShape:descriptor:stateTensor:`` — takes a
//      Philox state input and returns (tensor, new_state).  Proper
//      stateful RNG, but the state buffer must be plumbed as an
//      additional executable input *and* output, with lifecycle
//      managed Python-side per call.  That requires changes to the
//      compile pipeline's I/O schema that are cross-cutting.
//
// This file uses path (1) — deterministic-per-executable.  The seed
// is the eager Generator's counter at trace time (see the
// ``scope.set_attr("seed", ...)`` calls in
// :file:`lucid/_C/random/Random.cpp`), so distinct RNG calls within
// a single trace get distinct seeds (the eager Generator advances
// between draws).  Across executable invocations the same trace
// produces identical random values — useful for:
//
//   * deterministic noise injection (adversarial-robustness probes,
//     unit-test smoke checks);
//   * inference-mode dropout with ``p == 0`` (already passthrough,
//     no RNG needed);
//   * any pattern where seeded reproducibility is wanted.
//
// For training loops where stochasticity is required step-to-step
// (dropout regularisation, data augmentation, MC sampling) callers
// should keep using eager — the cache check infrastructure will
// detect the RNG signature and route to eager automatically.  Future
// stateful path is tracked in [[engine-rng-stateful-future]].

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "../../../core/Dtype.h"
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

inline std::int64_t int_attr(const OpNode& node, const char* key, std::int64_t def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end()) return def;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : def;
}

inline double double_attr(const OpNode& node, const char* key, double def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end()) return def;
    const auto* p = std::get_if<double>(&it->second);
    return p ? *p : def;
}

// Shared head: pull shape/dtype from the output meta, look up the
// MPSGraph + seed.  Returns nullptr on any missing piece.
struct RngContext {
    MPSGraph* graph = nil;
    NSArray<NSNumber*>* ns_shape = nil;
    MPSDataType ns_dt = MPSDataTypeFloat32;
    NSUInteger seed = 0;
};

inline bool open_rng(BuilderContext& ctx, const OpNode& node, RngContext& rc) {
    if (!node.inputs.empty()) return false;  // RNG has no traced inputs
    if (node.outputs.empty()) return false;
    rc.graph = (__bridge MPSGraph*)ctx.graph();
    if (rc.graph == nil) return false;
    const TensorMeta& meta = node.outputs[0];
    rc.ns_shape = shape_to_nsarray(meta.shape);
    rc.ns_dt = to_mps_dtype_local(meta.dtype);
    rc.seed = static_cast<NSUInteger>(int_attr(node, "seed", 0));
    return true;
}

class RandEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "rand"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        RngContext rc;
        if (!open_rng(ctx, node, rc)) return false;
        MPSGraphRandomOpDescriptor* d =
            [MPSGraphRandomOpDescriptor descriptorWithDistribution:
                                            MPSGraphRandomDistributionUniform
                                                          dataType:rc.ns_dt];
        d.min = 0.0f;
        d.max = 1.0f;
        ctx.bind(node.outputs[0].id, (__bridge void*)([rc.graph randomTensorWithShape:rc.ns_shape
                                                     descriptor:d
                                                           seed:rc.seed
                                                           name:@"rand"]));
        return true;
    }
};

class UniformEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "uniform"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        RngContext rc;
        if (!open_rng(ctx, node, rc)) return false;
        const double low = double_attr(node, "low", 0.0);
        const double high = double_attr(node, "high", 1.0);
        MPSGraphRandomOpDescriptor* d =
            [MPSGraphRandomOpDescriptor descriptorWithDistribution:
                                            MPSGraphRandomDistributionUniform
                                                          dataType:rc.ns_dt];
        d.min = static_cast<float>(low);
        d.max = static_cast<float>(high);
        ctx.bind(node.outputs[0].id, (__bridge void*)([rc.graph randomTensorWithShape:rc.ns_shape
                                                     descriptor:d
                                                           seed:rc.seed
                                                           name:@"uniform"]));
        return true;
    }
};

class RandnEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "randn"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        RngContext rc;
        if (!open_rng(ctx, node, rc)) return false;
        MPSGraphRandomOpDescriptor* d =
            [MPSGraphRandomOpDescriptor descriptorWithDistribution:
                                            MPSGraphRandomDistributionNormal
                                                          dataType:rc.ns_dt];
        d.mean = 0.0f;
        d.standardDeviation = 1.0f;
        ctx.bind(node.outputs[0].id, (__bridge void*)([rc.graph randomTensorWithShape:rc.ns_shape
                                                     descriptor:d
                                                           seed:rc.seed
                                                           name:@"randn"]));
        return true;
    }
};

class NormalEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "normal"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        RngContext rc;
        if (!open_rng(ctx, node, rc)) return false;
        const double mean = double_attr(node, "mean", 0.0);
        const double std_ = double_attr(node, "std", 1.0);
        MPSGraphRandomOpDescriptor* d =
            [MPSGraphRandomOpDescriptor descriptorWithDistribution:
                                            MPSGraphRandomDistributionNormal
                                                          dataType:rc.ns_dt];
        d.mean = static_cast<float>(mean);
        d.standardDeviation = static_cast<float>(std_);
        ctx.bind(node.outputs[0].id, (__bridge void*)([rc.graph randomTensorWithShape:rc.ns_shape
                                                     descriptor:d
                                                           seed:rc.seed
                                                           name:@"normal"]));
        return true;
    }
};

class RandintEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "randint"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        RngContext rc;
        if (!open_rng(ctx, node, rc)) return false;
        const std::int64_t low = int_attr(node, "low", 0);
        const std::int64_t high = int_attr(node, "high", 0);
        // MPSGraph's integer uniform path requires an int dtype.  If
        // Lucid materialised the result as a float (rare but
        // possible), fall back to eager.
        if (rc.ns_dt != MPSDataTypeInt32 && rc.ns_dt != MPSDataTypeInt64)
            return false;
        MPSGraphRandomOpDescriptor* d =
            [MPSGraphRandomOpDescriptor descriptorWithDistribution:
                                            MPSGraphRandomDistributionUniform
                                                          dataType:rc.ns_dt];
        d.minInteger = static_cast<NSInteger>(low);
        // MPSGraph integer uniform is inclusive of both ends; Lucid /
        // reference framework convention is [low, high).  Subtract 1
        // to match.
        d.maxInteger = static_cast<NSInteger>(high - 1);
        ctx.bind(node.outputs[0].id, (__bridge void*)([rc.graph randomTensorWithShape:rc.ns_shape
                                                     descriptor:d
                                                           seed:rc.seed
                                                           name:@"randint"]));
        return true;
    }
};

class BernoulliEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "bernoulli"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        RngContext rc;
        if (!open_rng(ctx, node, rc)) return false;
        const double p = double_attr(node, "p", 0.5);
        // Bernoulli(p): draw u ~ U(0, 1), return 1.0 if u < p else 0.0.
        // MPSGraph has no native Bernoulli, so compose: sample uniform
        // then compare < p.
        MPSGraphRandomOpDescriptor* d =
            [MPSGraphRandomOpDescriptor descriptorWithDistribution:
                                            MPSGraphRandomDistributionUniform
                                                          dataType:MPSDataTypeFloat32];
        d.min = 0.0f;
        d.max = 1.0f;
        MPSGraphTensor* u =
            [rc.graph randomTensorWithShape:rc.ns_shape
                                  descriptor:d
                                        seed:rc.seed
                                        name:@"bernoulli_u"];
        MPSGraphTensor* p_const =
            [rc.graph constantWithScalar:p dataType:MPSDataTypeFloat32];
        MPSGraphTensor* mask_bool =
            [rc.graph lessThanWithPrimaryTensor:u
                                 secondaryTensor:p_const
                                            name:nil];
        // Cast to the requested dtype (typically F32 or Bool).
        if (rc.ns_dt == MPSDataTypeBool) {
            ctx.bind(node.outputs[0].id, (__bridge void*)(mask_bool));
        return true;
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)([rc.graph castTensor:mask_bool
                                              toType:rc.ns_dt
                                                name:@"bernoulli"]));
        return true;
    }
};

struct RandomEmitterRegistrar {
    RandomEmitterRegistrar() {
        register_emitter(std::make_unique<RandEmitter>());
        register_emitter(std::make_unique<UniformEmitter>());
        register_emitter(std::make_unique<RandnEmitter>());
        register_emitter(std::make_unique<NormalEmitter>());
        register_emitter(std::make_unique<RandintEmitter>());
        register_emitter(std::make_unique<BernoulliEmitter>());
    }
};

[[maybe_unused]] static const RandomEmitterRegistrar g_random_registrar;

}  // namespace

}  // namespace lucid::compile
