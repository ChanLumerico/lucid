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

// Map a Lucid Dtype to an MPSDataType (local helper to avoid a sister
// header pull-in for the single dtype enum mapping we need below).
inline MPSDataType lucid_dtype_to_mps_local(Dtype dt) {
    switch (dt) {
    case Dtype::F16:
        return MPSDataTypeFloat16;
    case Dtype::F32:
        return MPSDataTypeFloat32;
    case Dtype::F64:
        return MPSDataTypeFloat32;  // MPS has no F64
    case Dtype::I8:
        return MPSDataTypeInt8;
    case Dtype::I16:
        return MPSDataTypeInt16;
    case Dtype::I32:
        return MPSDataTypeInt32;
    case Dtype::I64:
        return MPSDataTypeInt64;
    case Dtype::Bool:
        return MPSDataTypeBool;
    default:
        return MPSDataTypeFloat32;
    }
}

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

    // **AMP/mixed-dtype reconciliation.**  Under autocast, Lucid's
    // eager dispatch auto-downcasts F32+F16 binary ops to the lower
    // (autocast-target) dtype, so the trace's recorded output dtype
    // can disagree with what MPSGraph's binary builders produce
    // (MPSGraph silently *upcasts* mixed-dtype inputs to the higher
    // precision).  Reconcile by casting any operand whose dtype
    // doesn't match the recorded output dtype before the binary op.
    //
    // When all dtypes match the output's, both casts are no-ops and
    // this branch is free.
    const MPSDataType target_dt = lucid_dtype_to_mps_local(node.outputs[0].dtype);
    if (a_t.dataType != target_dt) {
        a_t = [graph castTensor:a_t toType:target_dt name:@"binop_cast_a"];
    }
    if (b_t.dataType != target_dt) {
        b_t = [graph castTensor:b_t toType:target_dt name:@"binop_cast_b"];
    }

    MPSGraphTensor* y = builder(graph, a_t, b_t);
    if (y == nil)
        return false;
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
            // ``floor(a/b)`` alone is right for floats and wrong for
            // integers: MPSGraph's integer division truncates toward zero
            // and flooring an integer is a no-op, so ``-7 // 2`` came back
            // as -3 where eager and Python both give -4.  This emitter was
            // unreachable until floordiv started recording its trace I/O,
            // so nothing had ever measured it.
            //
            // Correct it the way C's truncating division is corrected:
            // subtract one when the remainder is non-zero and disagrees in
            // sign with the divisor.  The correction is written to be a
            // no-op on the cases that are already right — on floats the
            // remainder after flooring already carries the divisor's sign,
            // and so does an integer division that floors on its own — so
            // this stays correct whichever semantics the backend has.
            MPSGraphTensor* q = [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
            q = [g floorWithTensor:q name:nil];

            MPSGraphTensor* prod = [g multiplicationWithPrimaryTensor:q secondaryTensor:b name:nil];
            MPSGraphTensor* rem = [g subtractionWithPrimaryTensor:a secondaryTensor:prod name:nil];

            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:a.dataType];
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:a.dataType];
            MPSGraphTensor* rem_nonzero = [g notEqualWithPrimaryTensor:rem
                                                       secondaryTensor:zero
                                                                  name:nil];
            MPSGraphTensor* rem_neg = [g lessThanWithPrimaryTensor:rem
                                                   secondaryTensor:zero
                                                              name:nil];
            MPSGraphTensor* div_neg = [g lessThanWithPrimaryTensor:b secondaryTensor:zero name:nil];
            // XOR, not ``notEqual``: comparing two booleans with the
            // latter is the same answer, but MPSGraph's
            // ConvertBinaryCompareToZero pass then warns on every run
            // that the second operand is not zero.
            MPSGraphTensor* signs_differ = [g logicalXORWithPrimaryTensor:rem_neg
                                                          secondaryTensor:div_neg
                                                                     name:nil];
            MPSGraphTensor* needs_adjust = [g logicalANDWithPrimaryTensor:rem_nonzero
                                                          secondaryTensor:signs_differ
                                                                     name:nil];
            MPSGraphTensor* adjust = [g selectWithPredicateTensor:needs_adjust
                                              truePredicateTensor:one
                                             falsePredicateTensor:zero
                                                             name:nil];
            return [g subtractionWithPrimaryTensor:q secondaryTensor:adjust name:@"floordiv"];
        });
    }
};

class NextafterEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "nextafter"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        // The same bit step as the eager Metal kernel (``nextafter_gpu_f32``):
        // one ULP is one unit of the float's bit pattern, not a fixed
        // epsilon.  This emitter used to add ``sign(b - a) * 1.19e-7``,
        // which leaves 100.0 unchanged and jumps 1e-10 by three orders of
        // magnitude — it was unreachable only because the op never
        // recorded its trace inputs.  Eager accepts F32 / F64 and F64 never
        // reaches Metal, so anything but F32 declines.
        if (node.outputs.empty() || node.outputs[0].dtype != Dtype::F32)
            return false;
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            auto i32 = [g](double v) { return [g constantWithScalar:v dataType:MPSDataTypeInt32]; };
            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
            MPSGraphTensor* a_bits = [g reinterpretCastTensor:a toType:MPSDataTypeInt32 name:nil];
            MPSGraphTensor* b_bits = [g reinterpretCastTensor:b toType:MPSDataTypeInt32 name:nil];

            // Toward b is +1 in value; a negative float's bits order the
            // other way, so the bit step flips sign with a.
            MPSGraphTensor* direction =
                [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:b
                                                             secondaryTensor:a
                                                                        name:nil]
                         truePredicateTensor:i32(1)
                        falsePredicateTensor:i32(-1)
                                        name:nil];
            MPSGraphTensor* delta =
                [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:a
                                                             secondaryTensor:zero
                                                                        name:nil]
                         truePredicateTensor:direction
                        falsePredicateTensor:[g negativeWithTensor:direction name:nil]
                                        name:nil];
            MPSGraphTensor* step = [g additionWithPrimaryTensor:a_bits
                                                secondaryTensor:delta
                                                           name:nil];

            // ±0 steps to the smallest subnormal carrying b's sign.
            MPSGraphTensor* from_zero =
                [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:b
                                                             secondaryTensor:zero
                                                                        name:nil]
                         truePredicateTensor:i32(1)
                        falsePredicateTensor:i32(static_cast<std::int32_t>(0x80000001u))
                                        name:nil];
            step = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:a
                                                          secondaryTensor:zero
                                                                     name:nil]
                            truePredicateTensor:from_zero
                           falsePredicateTensor:step
                                           name:nil];

            // a == b returns b itself, which keeps the sign of a zero.
            step = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:a
                                                          secondaryTensor:b
                                                                     name:nil]
                            truePredicateTensor:b_bits
                           falsePredicateTensor:step
                                           name:nil];

            MPSGraphTensor* any_nan = [g logicalORWithPrimaryTensor:[g isNaNWithTensor:a name:nil]
                                                    secondaryTensor:[g isNaNWithTensor:b name:nil]
                                                               name:nil];
            step = [g selectWithPredicateTensor:any_nan
                            truePredicateTensor:i32(static_cast<std::int32_t>(0x7FC00000u))
                           falsePredicateTensor:step
                                           name:nil];
            return [g reinterpretCastTensor:step toType:MPSDataTypeFloat32 name:@"nextafter"];
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
            if (const auto* p = std::get_if<double>(&it->second))
                exp_v = *p;
        }
        MPSGraphTensor* e = [graph constantWithScalar:exp_v dataType:x_t.dataType];
        MPSGraphTensor* y = [graph powerWithPrimaryTensor:x_t secondaryTensor:e name:@"pow_scalar"];
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
            if (const auto* p = std::get_if<double>(&it->second))
                base_v = *p;
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
