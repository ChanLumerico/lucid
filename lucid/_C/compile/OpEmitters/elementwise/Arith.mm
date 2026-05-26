// lucid/_C/compile/OpEmitters/elementwise/Arith.mm
//
// Two-tensor element-wise arithmetic emitters: add / sub / mul / div /
// pow / maximum / minimum / floordiv / nextafter + scalar-form
// pow_scalar / rpow_scalar.
//
// Op names match the engine schemas in:
//   - lucid/_C/ops/bfunc/{Add,Sub,Mul,Div,Pow,...}.cpp
//
// MPSGraph's primary/secondary binary builders broadcast natively, so
// the emitters don't need to thread broadcast-shape logic — the
// builder ensures the input MPSGraphTensors carry the correct shapes
// from their producing ops (Lucid's broadcast happens before
// dispatch, so each side has already been reshaped/copied as needed).
//
// ``matmul`` used to live in this file because the engine ships it
// under ``bfunc/``, but conceptually it's a matrix-algebra primitive
// — it now lives in ``linalg/Matmul.mm`` alongside Linear / Inner /
// Outer / Tensordot.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline bool emit_binary(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.size() != 2 || node.outputs.empty())
        return false;
    TensorId a_id = node.inputs[0];
    TensorId b_id = node.inputs[1];
    if (a_id < 0 || b_id < 0)
        return false;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* a_t = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
    MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
    if (a_t == nil || b_t == nil || graph == nil)
        return false;
    MPSGraphTensor* y = builder(graph, a_t, b_t);
    if (y == nil) return false;
    ctx.bind(node.outputs[0].id, (__bridge void*)y);
    return true;
}

class AddEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "add"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g additionWithPrimaryTensor:a secondaryTensor:b name:@"add"];
        });
    }
};

class SubEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sub"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:@"sub"];
        });
    }
};

class MulEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "mul"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:@"mul"];
        });
    }
};

class DivEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "div"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g divisionWithPrimaryTensor:a secondaryTensor:b name:@"div"];
        });
    }
};

class PowEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "pow"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g powerWithPrimaryTensor:a secondaryTensor:b name:@"pow"];
        });
    }
};

// ``matmul`` lives in ``linalg/Matmul.mm`` — see file header.

// R1 — additional binary ops.

class MaximumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "maximum"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g maximumWithPrimaryTensor:a secondaryTensor:b name:@"maximum"];
        });
    }
};

class MinimumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "minimum"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g minimumWithPrimaryTensor:a secondaryTensor:b name:@"minimum"];
        });
    }
};

class FloordivEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "floordiv"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            // floor(a/b) — uses native floor builder on the quotient.
            MPSGraphTensor* q =
                [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
            return [g floorWithTensor:q name:@"floordiv"];
        });
    }
};

class NextafterEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "nextafter"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        // MPSGraph doesn't expose nextafter directly; the eager backend
        // implements it via a 1-ULP step.  Emit a 1-ULP approximation
        // using the spacing helper: nextafter(a, b) ≈ a + sign(b-a) * eps,
        // good enough for fp32 / fp16 chain backward but not exact for
        // edge cases (zero / inf).  Document that this emitter is an
        // approximation; the eager fallback path handles exact semantics.
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            MPSDataType dt = a.dataType;
            MPSGraphTensor* diff =
                [g subtractionWithPrimaryTensor:b secondaryTensor:a name:nil];
            MPSGraphTensor* s = [g signWithTensor:diff name:nil];
            const double eps = (dt == MPSDataTypeFloat16) ? 9.7656e-4 : 1.1921e-7;
            MPSGraphTensor* e = [g constantWithScalar:eps dataType:dt];
            MPSGraphTensor* step =
                [g multiplicationWithPrimaryTensor:s secondaryTensor:e name:nil];
            return [g additionWithPrimaryTensor:a secondaryTensor:step name:@"nextafter"];
        });
    }
};

// pow_scalar — single-tensor input + ``exp`` (double) attr.
class PowScalarEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "pow_scalar"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1 || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;
        double exp_v = 1.0;
        auto it = node.attrs.find("exp");
        if (it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) exp_v = *p;
        }
        MPSGraphTensor* e = [graph constantWithScalar:exp_v dataType:x_t.dataType];
        MPSGraphTensor* y = [graph powerWithPrimaryTensor:x_t
                                          secondaryTensor:e
                                                     name:@"pow_scalar"];
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

// rpow_scalar — base^x with ``base`` (double) attr.
class RPowScalarEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "rpow_scalar"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1 || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;
        double base_v = std::exp(1.0);
        auto it = node.attrs.find("base");
        if (it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) base_v = *p;
        }
        MPSGraphTensor* b = [graph constantWithScalar:base_v dataType:x_t.dataType];
        MPSGraphTensor* y = [graph powerWithPrimaryTensor:b
                                          secondaryTensor:x_t
                                                     name:@"rpow_scalar"];
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

struct ElementwiseEmitterRegistrar {
    ElementwiseEmitterRegistrar() {
        register_emitter(std::make_unique<AddEmitter>());
        register_emitter(std::make_unique<SubEmitter>());
        register_emitter(std::make_unique<MulEmitter>());
        register_emitter(std::make_unique<DivEmitter>());
        register_emitter(std::make_unique<PowEmitter>());
        // ``matmul`` registered in linalg/Matmul.mm.
        // R1 additions.
        register_emitter(std::make_unique<MaximumEmitter>());
        register_emitter(std::make_unique<MinimumEmitter>());
        register_emitter(std::make_unique<FloordivEmitter>());
        register_emitter(std::make_unique<NextafterEmitter>());
        register_emitter(std::make_unique<PowScalarEmitter>());
        register_emitter(std::make_unique<RPowScalarEmitter>());
    }
};

[[maybe_unused]] static const ElementwiseEmitterRegistrar g_elementwise_registrar;

}  // namespace

}  // namespace lucid::compile
