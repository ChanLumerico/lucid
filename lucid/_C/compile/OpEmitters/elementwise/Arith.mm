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

#include "../../MpsDtype.h"
#include "../OpEmitter.h"
#include "../_SignedZero.h"

namespace lucid::compile {

namespace {

inline bool is_0d_f32_tensor(MPSGraphTensor* t) {
    return t.dataType == MPSDataTypeFloat32 && t.shape != nil && t.shape.count == 0;
}

inline bool is_half_f32_0d_pair(const OpNode& node, MPSDataType target_dt, MPSGraphTensor* a_t,
                                MPSGraphTensor* b_t) {
    const Dtype out_dt = node.outputs[0].dtype;
    if (out_dt != Dtype::F16 && out_dt != Dtype::BF16)
        return false;
    if (node.name != "mul" && node.name != "div" && node.name != "floordiv")
        return false;
    return (a_t.dataType == target_dt && is_0d_f32_tensor(b_t)) ||
           (b_t.dataType == target_dt && is_0d_f32_tensor(a_t));
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
    // **Half op-math (F16/BF16 + 0-d F32 scalar/tensor).**  Eager keeps a
    // half tensor paired with a 0-d float32 operand at the half output dtype
    // while computing ``mul`` / ``div`` / ``floordiv`` in float32 before
    // rounding back to half (``_Opmath.h``).
    const auto held = mps_dtype_of(node.outputs[0].dtype);
    if (!held)
        return false;
    const MPSDataType target_dt = *held;
    const MPSDataType op_dt =
        is_half_f32_0d_pair(node, target_dt, a_t, b_t) ? MPSDataTypeFloat32 : target_dt;

    if (a_t.dataType != op_dt) {
        a_t = [graph castTensor:a_t toType:op_dt name:@"binop_cast_a"];
    }
    if (b_t.dataType != op_dt) {
        b_t = [graph castTensor:b_t toType:op_dt name:@"binop_cast_b"];
    }

    MPSGraphTensor* y = builder(graph, a_t, b_t);
    if (y == nil)
        return false;
    if (y.dataType != target_dt) {
        y = [graph castTensor:y toType:target_dt name:@"binop_cast_out"];
    }
    ctx.bind(node.outputs[0].id, (__bridge void*)y);
    return true;
}

// ``add`` and ``sub`` route an operand that is a +0 (add) or -0 (sub) splat
// through ``add_pos_zero`` — MPSGraph would fold it away and keep a -0 that
// IEEE turns into +0 (_SignedZero.h).
class AddEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "add"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const bool a_pos_zero = node.inputs.size() == 2 && is_pos_zero_splat(ctx, node.inputs[0]);
        const bool b_pos_zero = node.inputs.size() == 2 && is_pos_zero_splat(ctx, node.inputs[1]);
        return emit_binary(
            ctx, node, [a_pos_zero, b_pos_zero](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
                if (b_pos_zero)
                    return add_pos_zero(g, a, b);
                if (a_pos_zero)
                    return add_pos_zero(g, b, a);
                return [g additionWithPrimaryTensor:a secondaryTensor:b name:@"add"];
            });
    }
};

class SubEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sub"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const bool b_neg_zero = node.inputs.size() == 2 && is_neg_zero_splat(ctx, node.inputs[1]);
        return emit_binary(
            ctx, node, [b_neg_zero](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
                if (b_neg_zero)
                    return add_pos_zero(g, a, [g negativeWithTensor:b name:nil]);
                return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:@"sub"];
            });
    }
};

inline MPSGraphTensor* multiply_avoiding_i64_square(MPSGraph* g, MPSGraphTensor* a,
                                                    MPSGraphTensor* b, NSString* name) {
    // MPSGraph's algebraic simplifier pattern-matches ``mul(x, x)`` (after CSE
    // of identical subgraphs such as ``abs(a) * abs(a)`` or ``y.argmax(1) *
    // y.argmax(1)``) and rewrites it to ``mps.square(x)``.  On 64-bit integers
    // MPSGraph has a working ``mul_i64`` kernel but no ``square_i64`` kernel,
    // so the rewritten executable throws an Objective-C exception ("object
    // cannot be nil (key: square_i64)") on its first run.
    //
    // Wrapping the primary operand in ``maximum(a, a)`` and the secondary
    // operand in ``minimum(b, b)`` is an exact identity for all 64-bit integer
    // bit patterns (integers have no NaNs or signed zeros), and MPSGraph does
    // not fold ``maximum(x, x)`` or ``minimum(x, x)``.  Because the defining
    // MLIR ops of the two operands are distinct (``mps.maximum`` vs
    // ``mps.minimum``), CSE can never unify them into the same SSA value even
    // if ``a`` and ``b`` already end in ``maximum`` or ``minimum``, so
    // ``mul_i64`` always runs directly.
    if (a.dataType == MPSDataTypeInt64 || a.dataType == MPSDataTypeUInt64) {
        MPSGraphTensor* lhs = [g maximumWithPrimaryTensor:a secondaryTensor:a name:nil];
        MPSGraphTensor* rhs = [g minimumWithPrimaryTensor:b secondaryTensor:b name:nil];
        return [g multiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:name];
    }
    return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:name];
}

class MulEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "mul"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return multiply_avoiding_i64_square(g, a, b, @"mul");
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

inline MPSGraphTensor* emit_float_floordiv(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
    // Match eager ``DivMod.h`` / ``gpu/DivMod.h``: floating-point floor
    // division runs in float32 via exact ``fmod`` so an upward-rounded
    // quotient (e.g. ``1.5 / 0.1 == 15.0`` when ``0.1f > 0.1``) or
    // reciprocal multiplication (``1.7 / 1.7 < 1.0``) does not land one
    // integer off.
    const MPSDataType orig_dt = a.dataType;
    MPSGraphTensor* af = (orig_dt == MPSDataTypeFloat32)
                             ? a
                             : [g castTensor:a toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* bf = (orig_dt == MPSDataTypeFloat32)
                             ? b
                             : [g castTensor:b toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:MPSDataTypeFloat32];

    MPSGraphTensor* raw_div = [g divisionWithPrimaryTensor:af secondaryTensor:bf name:nil];
    MPSGraphTensor* mod = [g moduloWithPrimaryTensor:af secondaryTensor:bf name:nil];
    MPSGraphTensor* mod_nonzero = [g notEqualWithPrimaryTensor:mod secondaryTensor:zero name:nil];
    MPSGraphTensor* mod_neg = [g lessThanWithPrimaryTensor:mod secondaryTensor:zero name:nil];
    MPSGraphTensor* div_neg = [g lessThanWithPrimaryTensor:bf secondaryTensor:zero name:nil];
    MPSGraphTensor* signs_differ = [g logicalXORWithPrimaryTensor:mod_neg
                                                  secondaryTensor:div_neg
                                                             name:nil];
    MPSGraphTensor* needs_adjust = [g logicalANDWithPrimaryTensor:mod_nonzero
                                                  secondaryTensor:signs_differ
                                                             name:nil];
    MPSGraphTensor* adjust = [g selectWithPredicateTensor:needs_adjust
                                      truePredicateTensor:one
                                     falsePredicateTensor:zero
                                                     name:nil];
    MPSGraphTensor* num = [g subtractionWithPrimaryTensor:af secondaryTensor:mod name:nil];
    MPSGraphTensor* q_exact = [g divisionWithPrimaryTensor:num secondaryTensor:bf name:nil];
    MPSGraphTensor* div = [g subtractionWithPrimaryTensor:q_exact secondaryTensor:adjust name:nil];
    MPSGraphTensor* whole = [g floorWithTensor:div name:nil];
    MPSGraphTensor* frac = [g subtractionWithPrimaryTensor:div secondaryTensor:whole name:nil];
    MPSGraphTensor* round_up = [g greaterThanWithPrimaryTensor:frac secondaryTensor:half name:nil];
    MPSGraphTensor* whole_plus_one = [g additionWithPrimaryTensor:whole
                                                  secondaryTensor:one
                                                             name:nil];
    whole = [g selectWithPredicateTensor:round_up
                     truePredicateTensor:whole_plus_one
                    falsePredicateTensor:whole
                                    name:nil];
    MPSGraphTensor* signed_zero = [g multiplicationWithPrimaryTensor:raw_div
                                                     secondaryTensor:zero
                                                                name:nil];
    MPSGraphTensor* div_is_zero = [g equalWithPrimaryTensor:div secondaryTensor:zero name:nil];
    MPSGraphTensor* res = [g selectWithPredicateTensor:div_is_zero
                                   truePredicateTensor:signed_zero
                                  falsePredicateTensor:whole
                                                  name:nil];
    MPSGraphTensor* b_is_zero = [g equalWithPrimaryTensor:bf secondaryTensor:zero name:nil];
    res = [g selectWithPredicateTensor:b_is_zero
                   truePredicateTensor:raw_div
                  falsePredicateTensor:res
                                  name:@"floordiv"];
    return (orig_dt == MPSDataTypeFloat32)
               ? res
               : [g castTensor:res toType:orig_dt name:@"floordiv_cast"];
}

class FloordivEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "floordiv"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            const MPSDataType dt = a.dataType;
            if (dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16 || dt == MPSDataTypeFloat32) {
                return emit_float_floordiv(g, a, b);
            }

            // Integer floor division: MPSGraph's integer division truncates
            // toward zero, so subtract one when the remainder is non-zero and
            // disagrees in sign with the divisor.
            MPSGraphTensor* q = [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
            q = [g floorWithTensor:q name:nil];

            MPSGraphTensor* prod = multiply_avoiding_i64_square(g, q, b, nil);
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
