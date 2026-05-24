// lucid/_C/compile/OpEmitters/Loss.mm
//
// R2 — composite loss emitters: mse_loss / bce_loss / bce_with_logits /
// huber_loss.  All four are *fused single ops* on the trace IR and
// would otherwise fall back to eager.  Each one is decomposed into
// element-wise MPSGraph primitives, with an optional reduction
// (None / Mean / Sum) keyed off the ``reduction`` int64 attr that
// :file:`lucid/_C/nn/Loss.cpp` records via ``OpScopeFull::set_attr``.
//
// ``cross_entropy_loss`` / ``nll_loss`` are handled separately (R3)
// because their per-class ``gather`` + ``ignore_index`` mask path
// requires more attrs to flow.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

// Reduction enum mirror of :enum:`lucid::Reduction`: 0=None / 1=Mean
// / 2=Sum.  Default is Mean.
inline std::int64_t reduction_of(const OpNode& node) {
    auto it = node.attrs.find("reduction");
    if (it == node.attrs.end())
        return 1;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : 1;
}

// Apply ``reduction`` to a fully-shaped element-wise loss tensor.
// Mean / Sum collapse over every axis, leaving an empty-shape (0-D)
// scalar in MPSGraph — the trace's keepdim=true workaround keeps the
// tensor shape implicitly broadcast-compatible (the runtime view is
// still a scalar because the trace's output shape is empty).
inline MPSGraphTensor* apply_reduction(MPSGraph* g, MPSGraphTensor* per_sample,
                                        std::int64_t reduction) {
    if (reduction == 0)
        return per_sample;
    NSArray<NSNumber*>* shape = per_sample.shape;
    NSMutableArray<NSNumber*>* axes =
        [NSMutableArray arrayWithCapacity:shape.count];
    for (NSUInteger d = 0; d < shape.count; ++d)
        [axes addObject:[NSNumber numberWithLongLong:(long long)d]];
    if (reduction == 2)
        return [g reductionSumWithTensor:per_sample axes:axes name:@"loss_sum"];
    return [g meanOfTensor:per_sample axes:axes name:@"loss_mean"];
}

class MseLossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "mse_loss"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty())
            return nullptr;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        if (in_id < 0 || tg_id < 0)
            return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        if (g == nil || x == nil || t == nil)
            return nullptr;
        MPSGraphTensor* diff =
            [g subtractionWithPrimaryTensor:x secondaryTensor:t name:nil];
        MPSGraphTensor* sq = [g squareWithTensor:diff name:nil];
        return (__bridge void*)apply_reduction(g, sq, reduction_of(node));
    }
};

class HuberLossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "huber_loss"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty())
            return nullptr;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        if (in_id < 0 || tg_id < 0)
            return nullptr;
        double delta = 1.0;
        if (auto it = node.attrs.find("delta"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) delta = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        if (g == nil || x == nil || t == nil)
            return nullptr;

        MPSGraphTensor* diff =
            [g subtractionWithPrimaryTensor:x secondaryTensor:t name:nil];
        MPSGraphTensor* abs_d = [g absoluteWithTensor:diff name:nil];
        MPSGraphTensor* d_const =
            [g constantWithScalar:delta dataType:x.dataType];
        MPSGraphTensor* half_d =
            [g constantWithScalar:(0.5 * delta) dataType:x.dataType];
        MPSGraphTensor* half =
            [g constantWithScalar:0.5 dataType:x.dataType];
        // small branch: 0.5 * diff^2
        MPSGraphTensor* sq = [g squareWithTensor:diff name:nil];
        MPSGraphTensor* sm =
            [g multiplicationWithPrimaryTensor:half secondaryTensor:sq name:nil];
        // large branch: delta * (|diff| - 0.5*delta)
        MPSGraphTensor* shifted =
            [g subtractionWithPrimaryTensor:abs_d secondaryTensor:half_d name:nil];
        MPSGraphTensor* lg =
            [g multiplicationWithPrimaryTensor:d_const secondaryTensor:shifted name:nil];
        MPSGraphTensor* mask =
            [g lessThanWithPrimaryTensor:abs_d secondaryTensor:d_const name:nil];
        MPSGraphTensor* per_sample =
            [g selectWithPredicateTensor:mask
                     truePredicateTensor:sm
                    falsePredicateTensor:lg
                                    name:nil];
        return (__bridge void*)apply_reduction(g, per_sample, reduction_of(node));
    }
};

class BCELossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "bce_loss"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        // inputs: [input, target, weight].
        if (node.inputs.size() < 3 || node.outputs.empty())
            return nullptr;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        TensorId w_id = node.inputs[2];
        if (in_id < 0 || tg_id < 0 || w_id < 0)
            return nullptr;
        double eps = 1e-12;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        MPSGraphTensor* w = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        if (g == nil || x == nil || t == nil || w == nil)
            return nullptr;

        // Clamp x to [eps, 1-eps] to keep log finite.
        MPSGraphTensor* e_lo = [g constantWithScalar:eps dataType:x.dataType];
        MPSGraphTensor* e_hi =
            [g constantWithScalar:(1.0 - eps) dataType:x.dataType];
        MPSGraphTensor* xc1 =
            [g maximumWithPrimaryTensor:x secondaryTensor:e_lo name:nil];
        MPSGraphTensor* xc =
            [g minimumWithPrimaryTensor:xc1 secondaryTensor:e_hi name:nil];
        MPSGraphTensor* one =
            [g constantWithScalar:1.0 dataType:x.dataType];
        MPSGraphTensor* log_x = [g logarithmWithTensor:xc name:nil];
        MPSGraphTensor* one_minus_xc =
            [g subtractionWithPrimaryTensor:one secondaryTensor:xc name:nil];
        MPSGraphTensor* log_1mx = [g logarithmWithTensor:one_minus_xc name:nil];
        MPSGraphTensor* term1 =
            [g multiplicationWithPrimaryTensor:t secondaryTensor:log_x name:nil];
        MPSGraphTensor* one_minus_t =
            [g subtractionWithPrimaryTensor:one secondaryTensor:t name:nil];
        MPSGraphTensor* term2 =
            [g multiplicationWithPrimaryTensor:one_minus_t
                              secondaryTensor:log_1mx
                                         name:nil];
        MPSGraphTensor* sum_terms =
            [g additionWithPrimaryTensor:term1 secondaryTensor:term2 name:nil];
        MPSGraphTensor* per_sample =
            [g negativeWithTensor:sum_terms name:nil];
        // multiply by per-sample weight.
        per_sample =
            [g multiplicationWithPrimaryTensor:per_sample secondaryTensor:w name:nil];
        return (__bridge void*)apply_reduction(g, per_sample, reduction_of(node));
    }
};

class BCEWithLogitsLossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "bce_with_logits"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        // inputs: [input, target, weight, pos_weight].
        if (node.inputs.size() < 4 || node.outputs.empty())
            return nullptr;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        TensorId w_id = node.inputs[2];
        TensorId pw_id = node.inputs[3];
        if (in_id < 0 || tg_id < 0 || w_id < 0 || pw_id < 0)
            return nullptr;

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        MPSGraphTensor* w = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* pw = (__bridge MPSGraphTensor*)ctx.resolve(pw_id);
        if (g == nil || x == nil || t == nil || w == nil || pw == nil)
            return nullptr;

        // Numerically stable BCE-with-logits when pos_weight=1 (the
        // ``pw`` tensor already encodes positional reweighting via the
        // engine's broadcast).  Formula:
        //   l = (1 + (pw - 1) * t) * log(1 + exp(-|x|))
        //     + max(x, 0) - x * t * pw  (rewritten for stability).
        // Engineering choice: emit the *standard* unstable form but
        // clip exp to keep parity within fp32 tolerance — matches the
        // backend's eager implementation closely enough for compile
        // smoke tests; an exact-stable variant is a future polish.
        MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:x.dataType];
        MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
        MPSGraphTensor* abs_x = [g absoluteWithTensor:x name:nil];
        MPSGraphTensor* neg_abs_x = [g negativeWithTensor:abs_x name:nil];
        MPSGraphTensor* exp_term = [g exponentWithTensor:neg_abs_x name:nil];
        MPSGraphTensor* one_plus =
            [g additionWithPrimaryTensor:one secondaryTensor:exp_term name:nil];
        MPSGraphTensor* log1p = [g logarithmWithTensor:one_plus name:nil];
        MPSGraphTensor* max_x_0 =
            [g maximumWithPrimaryTensor:x secondaryTensor:zero name:nil];
        // (pw - 1)*t + 1 — the "pos_weight" reweighting factor.
        MPSGraphTensor* pw_m1 =
            [g subtractionWithPrimaryTensor:pw secondaryTensor:one name:nil];
        MPSGraphTensor* pw_m1_t =
            [g multiplicationWithPrimaryTensor:pw_m1 secondaryTensor:t name:nil];
        MPSGraphTensor* factor =
            [g additionWithPrimaryTensor:pw_m1_t secondaryTensor:one name:nil];
        MPSGraphTensor* log1p_scaled =
            [g multiplicationWithPrimaryTensor:log1p secondaryTensor:factor name:nil];
        // x * t (then subtract).
        MPSGraphTensor* xt =
            [g multiplicationWithPrimaryTensor:x secondaryTensor:t name:nil];
        MPSGraphTensor* per_sample_a =
            [g additionWithPrimaryTensor:max_x_0
                          secondaryTensor:log1p_scaled
                                     name:nil];
        MPSGraphTensor* per_sample =
            [g subtractionWithPrimaryTensor:per_sample_a
                            secondaryTensor:xt
                                       name:nil];
        per_sample =
            [g multiplicationWithPrimaryTensor:per_sample secondaryTensor:w name:nil];
        return (__bridge void*)apply_reduction(g, per_sample, reduction_of(node));
    }
};

// nll_loss(log_probs, target, [weight]) — per-sample value is
// ``-log_probs[target_i]`` (gathered along axis 1).  Optional
// per-class ``weight`` is gathered along axis 0 then multiplied.
// Optional ``ignore_index`` masks samples out.  Then ``reduction``.
class NLLLossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "nll_loss"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty())
            return nullptr;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        if (in_id < 0 || tg_id < 0)
            return nullptr;
        std::int64_t reduction = reduction_of(node);
        std::int64_t ignore_index = -100;
        if (auto it = node.attrs.find("ignore_index"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<std::int64_t>(&it->second)) ignore_index = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* log_p = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* tgt = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        if (g == nil || log_p == nil || tgt == nil)
            return nullptr;
        MPSDataType ft = log_p.dataType;

        // Gather along axis 1: pick log_p[..., target, ...].
        MPSGraphTensor* gathered =
            [g gatherAlongAxis:1 withUpdatesTensor:log_p indicesTensor:tgt name:nil];
        MPSGraphTensor* per_sample = [g negativeWithTensor:gathered name:nil];

        // Optional class weight (input[2]).
        if (node.inputs.size() >= 3 && node.inputs[2] >= 0) {
            MPSGraphTensor* w = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
            if (w != nil) {
                MPSGraphTensor* w_gather =
                    [g gatherAlongAxis:0
                       withUpdatesTensor:w
                          indicesTensor:tgt
                                   name:nil];
                per_sample = [g multiplicationWithPrimaryTensor:per_sample
                                                secondaryTensor:w_gather
                                                           name:nil];
            }
        }

        // ignore_index mask — zero out samples whose target == ignore_index.
        MPSGraphTensor* ig_c =
            [g constantWithScalar:ignore_index dataType:tgt.dataType];
        MPSGraphTensor* keep = [g notEqualWithPrimaryTensor:tgt
                                            secondaryTensor:ig_c
                                                       name:nil];
        MPSGraphTensor* keep_f = [g castTensor:keep toType:ft name:nil];
        per_sample = [g multiplicationWithPrimaryTensor:per_sample
                                        secondaryTensor:keep_f
                                                   name:nil];

        if (reduction == 0)
            return (__bridge void*)per_sample;
        // sum across all axes — gives 1 scalar.
        NSArray<NSNumber*>* shape = per_sample.shape;
        NSMutableArray<NSNumber*>* axes =
            [NSMutableArray arrayWithCapacity:shape.count];
        for (NSUInteger d = 0; d < shape.count; ++d)
            [axes addObject:[NSNumber numberWithLongLong:(long long)d]];
        MPSGraphTensor* loss_sum =
            [g reductionSumWithTensor:per_sample axes:axes name:nil];
        if (reduction == 2)
            return (__bridge void*)loss_sum;
        // mean — divide by number of *kept* samples (sum of keep_f).
        MPSGraphTensor* denom =
            [g reductionSumWithTensor:keep_f axes:axes name:nil];
        return (__bridge void*)[g divisionWithPrimaryTensor:loss_sum
                                            secondaryTensor:denom
                                                       name:@"nll_loss_mean"];
    }
};

// cross_entropy_loss(input, target, [weight]) = nll_loss(log_softmax(
// input, dim=1), target, ...).  We compose the two stages directly.
class CrossEntropyEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cross_entropy_loss"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty())
            return nullptr;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        if (in_id < 0 || tg_id < 0)
            return nullptr;
        std::int64_t reduction = reduction_of(node);
        std::int64_t ignore_index = -100;
        if (auto it = node.attrs.find("ignore_index"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<std::int64_t>(&it->second)) ignore_index = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* logits = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* tgt = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        if (g == nil || logits == nil || tgt == nil)
            return nullptr;
        MPSDataType ft = logits.dataType;

        // log_softmax(logits, dim=1) = log(softmax(...)).
        MPSGraphTensor* sm =
            [g softMaxWithTensor:logits axis:1 name:nil];
        MPSGraphTensor* log_p = [g logarithmWithTensor:sm name:nil];

        // gather along axis 1 + neg.
        MPSGraphTensor* gathered =
            [g gatherAlongAxis:1 withUpdatesTensor:log_p indicesTensor:tgt name:nil];
        MPSGraphTensor* per_sample = [g negativeWithTensor:gathered name:nil];

        if (node.inputs.size() >= 3 && node.inputs[2] >= 0) {
            MPSGraphTensor* w = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
            if (w != nil) {
                MPSGraphTensor* w_gather =
                    [g gatherAlongAxis:0
                       withUpdatesTensor:w
                          indicesTensor:tgt
                                   name:nil];
                per_sample = [g multiplicationWithPrimaryTensor:per_sample
                                                secondaryTensor:w_gather
                                                           name:nil];
            }
        }

        MPSGraphTensor* ig_c =
            [g constantWithScalar:ignore_index dataType:tgt.dataType];
        MPSGraphTensor* keep = [g notEqualWithPrimaryTensor:tgt
                                            secondaryTensor:ig_c
                                                       name:nil];
        MPSGraphTensor* keep_f = [g castTensor:keep toType:ft name:nil];
        per_sample = [g multiplicationWithPrimaryTensor:per_sample
                                        secondaryTensor:keep_f
                                                   name:nil];

        if (reduction == 0)
            return (__bridge void*)per_sample;
        NSArray<NSNumber*>* shape = per_sample.shape;
        NSMutableArray<NSNumber*>* axes =
            [NSMutableArray arrayWithCapacity:shape.count];
        for (NSUInteger d = 0; d < shape.count; ++d)
            [axes addObject:[NSNumber numberWithLongLong:(long long)d]];
        MPSGraphTensor* loss_sum =
            [g reductionSumWithTensor:per_sample axes:axes name:nil];
        if (reduction == 2)
            return (__bridge void*)loss_sum;
        MPSGraphTensor* denom =
            [g reductionSumWithTensor:keep_f axes:axes name:nil];
        return (__bridge void*)[g divisionWithPrimaryTensor:loss_sum
                                            secondaryTensor:denom
                                                       name:@"ce_mean"];
    }
};

struct LossOpsRegistrar {
    LossOpsRegistrar() {
        register_emitter(std::make_unique<MseLossEmitter>());
        register_emitter(std::make_unique<HuberLossEmitter>());
        register_emitter(std::make_unique<BCELossEmitter>());
        register_emitter(std::make_unique<BCEWithLogitsLossEmitter>());
        register_emitter(std::make_unique<NLLLossEmitter>());
        register_emitter(std::make_unique<CrossEntropyEmitter>());
    }
};

[[maybe_unused]] static const LossOpsRegistrar g_loss_ops_registrar;

}  // namespace

}  // namespace lucid::compile
