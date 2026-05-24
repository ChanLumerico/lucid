// lucid/_C/compile/OpEmitters/UnaryMath.mm
//
// Single-tensor mathematical emitters: neg / abs / exp / log / sqrt /
// square / reciprocal / rsqrt.
//
// Op names match the engine schemas in:
//   - lucid/_C/ops/ufunc/Arith.cpp        ("neg", "abs", "reciprocal", "square")
//   - lucid/_C/ops/ufunc/Exponential.cpp  ("exp", "log", "sqrt", "rsqrt")
//
// All eight ops are 1-input → 1-output, no attribute payload required —
// MPSGraph exposes them directly as one-call builders (or, for
// ``square`` / ``rsqrt``, a single derived builder over the primitive).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <optional>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline void* emit_unary_math(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.size() != 1)
        return nullptr;
    TensorId x_id = node.inputs[0];
    if (x_id < 0)
        return nullptr;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
    if (x_t == nil || graph == nil)
        return nullptr;
    return (__bridge void*)builder(graph, x_t);
}

class NegEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "neg"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g negativeWithTensor:x name:@"neg"];
        });
    }
};

class AbsEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "abs"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g absoluteWithTensor:x name:@"abs"];
        });
    }
};

class ExpEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "exp"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g exponentWithTensor:x name:@"exp"];
        });
    }
};

class LogEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "log"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g logarithmWithTensor:x name:@"log"];
        });
    }
};

class SqrtEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sqrt"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g squareRootWithTensor:x name:@"sqrt"];
        });
    }
};

class SquareEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "square"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g squareWithTensor:x name:@"square"];
        });
    }
};

class ReciprocalEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "reciprocal"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g reciprocalWithTensor:x name:@"reciprocal"];
        });
    }
};

class RsqrtEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "rsqrt"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g reciprocalSquareRootWithTensor:x name:@"rsqrt"];
        });
    }
};

// Phase 2 (R1) — additional unary math: trig inverses + rounding +
// cube + log2 + sign + erf + clip.  Each is a direct MPSGraph
// 1-builder so the per-op emitter body stays a single lambda.

class ArccosEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "arccos"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g acosWithTensor:x name:@"arccos"];
        });
    }
};

class ArcsinEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "arcsin"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g asinWithTensor:x name:@"arcsin"];
        });
    }
};

class ArctanEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "arctan"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g atanWithTensor:x name:@"arctan"];
        });
    }
};

class CeilEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "ceil"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g ceilWithTensor:x name:@"ceil"];
        });
    }
};

class FloorEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "floor"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g floorWithTensor:x name:@"floor"];
        });
    }
};

class RoundEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "round"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g roundWithTensor:x name:@"round"];
        });
    }
};

class CubeEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cube"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            MPSGraphTensor* x2 = [g squareWithTensor:x name:@"cube_sq"];
            return [g multiplicationWithPrimaryTensor:x2
                                       secondaryTensor:x
                                                  name:@"cube"];
        });
    }
};

class CubeRootEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cube_root"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            // Sign-preserving cbrt(x) = sign(x) · |x|^(1/3).
            MPSGraphTensor* a = [g absoluteWithTensor:x name:@"cbrt_abs"];
            MPSGraphTensor* third =
                [g constantWithScalar:(1.0 / 3.0) dataType:a.dataType];
            MPSGraphTensor* mag = [g powerWithPrimaryTensor:a
                                            secondaryTensor:third
                                                       name:@"cbrt_pow"];
            MPSGraphTensor* s = [g signWithTensor:x name:@"cbrt_sign"];
            return [g multiplicationWithPrimaryTensor:s
                                       secondaryTensor:mag
                                                  name:@"cube_root"];
        });
    }
};

class Log2Emitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "log2"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g logarithmBase2WithTensor:x name:@"log2"];
        });
    }
};

class SignEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sign"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g signWithTensor:x name:@"sign"];
        });
    }
};

class ErfEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "erf"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g erfWithTensor:x name:@"erf"];
        });
    }
};

// ``clip(x, min, max)`` — Lucid records ``clip_min`` / ``clip_max``
// double attrs on the OpScopeFull (the underlying op may pass either
// or both bounds; missing bounds default to ±∞ via NaN/inf sentinels
// already filtered out of the attrs).  When both bounds are present
// we use ``clampWithTensor:min:max:``; when only one is present we
// compose `maximum` / `minimum` against a constant.
class ClipEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "clip"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1 || node.outputs.empty())
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return nullptr;

        auto have = [&](const char* key) -> std::optional<double> {
            auto it = node.attrs.find(key);
            if (it == node.attrs.end())
                return std::nullopt;
            const auto* p = std::get_if<double>(&it->second);
            return p ? std::optional<double>(*p) : std::nullopt;
        };
        std::optional<double> lo = have("min");
        std::optional<double> hi = have("max");
        if (!lo && !hi)
            return (__bridge void*)x_t;
        MPSGraphTensor* out = x_t;
        if (lo) {
            MPSGraphTensor* c =
                [graph constantWithScalar:*lo dataType:x_t.dataType];
            out = [graph maximumWithPrimaryTensor:out
                                   secondaryTensor:c
                                              name:@"clip_lo"];
        }
        if (hi) {
            MPSGraphTensor* c =
                [graph constantWithScalar:*hi dataType:x_t.dataType];
            out = [graph minimumWithPrimaryTensor:out
                                   secondaryTensor:c
                                              name:@"clip_hi"];
        }
        return (__bridge void*)out;
    }
};

struct UnaryMathEmitterRegistrar {
    UnaryMathEmitterRegistrar() {
        register_emitter(std::make_unique<NegEmitter>());
        register_emitter(std::make_unique<AbsEmitter>());
        register_emitter(std::make_unique<ExpEmitter>());
        register_emitter(std::make_unique<LogEmitter>());
        register_emitter(std::make_unique<SqrtEmitter>());
        register_emitter(std::make_unique<SquareEmitter>());
        register_emitter(std::make_unique<ReciprocalEmitter>());
        register_emitter(std::make_unique<RsqrtEmitter>());
        // R1 additions.
        register_emitter(std::make_unique<ArccosEmitter>());
        register_emitter(std::make_unique<ArcsinEmitter>());
        register_emitter(std::make_unique<ArctanEmitter>());
        register_emitter(std::make_unique<CeilEmitter>());
        register_emitter(std::make_unique<FloorEmitter>());
        register_emitter(std::make_unique<RoundEmitter>());
        register_emitter(std::make_unique<CubeEmitter>());
        register_emitter(std::make_unique<CubeRootEmitter>());
        register_emitter(std::make_unique<Log2Emitter>());
        register_emitter(std::make_unique<SignEmitter>());
        register_emitter(std::make_unique<ErfEmitter>());
        register_emitter(std::make_unique<ClipEmitter>());
    }
};

[[maybe_unused]] static const UnaryMathEmitterRegistrar g_unary_math_registrar;

}  // namespace

}  // namespace lucid::compile
