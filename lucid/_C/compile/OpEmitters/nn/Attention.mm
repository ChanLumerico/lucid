// lucid/_C/compile/OpEmitters/nn/Attention.mm
//
// scaled_dot_product_attention emitter — matmul + softmax + matmul.
//
// Inputs: ``{q, k, v}`` plus an optional ``{attn_mask}``.  Attrs:
//   - ``scale``     (double) — multiplier on Q·Kᵀ; 0.0 means "use
//     1/√D_k" per Lucid convention
//   - ``is_causal`` (bool)   — lower-triangular causal masking
//   - ``has_mask``  (bool)   — whether an additive mask is wired
//
// Decomposition::
//
//     scores  = (Q @ Kᵀ) * scale
//     scores += attn_mask   (when has_mask)
//     attn    = softmax(scores, axis=-1)
//     out     = attn @ V
//
// ``is_causal=True`` adds a lower-triangular −∞ mask before the softmax.
// The mask is built in-graph from O(Lq+Lk) host index vectors (an iota per
// axis) compared with ``lessThanOrEqualTo`` + ``select`` — cheaper than a
// baked O(Lq·Lk) constant per layer.  It uses the bottom-right alignment
// ``j ≤ i + (Lk − Lq)`` so the non-square (cached-decode) case matches the
// eager fused-causal convention.  An explicit additive mask takes
// precedence over ``is_causal`` (mirrors the eager backend, which ignores
// ``is_causal`` once a float ``attn_mask`` is supplied).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

class SdpaEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "scaled_dot_product_attention"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 3 || node.outputs.empty())
            return false;
        TensorId q_id = node.inputs[0];
        TensorId k_id = node.inputs[1];
        TensorId v_id = node.inputs[2];
        if (q_id < 0 || k_id < 0 || v_id < 0)
            return false;
        const bool has_mask = bool_attr(node, "has_mask", false);
        const bool is_causal = bool_attr(node, "is_causal", false);
        const double scale = double_attr(node, "scale", 0.0);
        // A declared additive ``attn_mask`` is a non-differentiable auxiliary
        // input the tracer does not record in the autograd input set, so it
        // never reaches ``node.inputs`` (only q/k/v do).  Without the mask
        // tensor the executable cannot reproduce the op — fall back to eager
        // rather than silently dropping the mask (and, since the eager backend
        // lets an additive mask win over ``is_causal``, this keeps the
        // causal+mask combination correct too).
        if (has_mask && node.inputs.size() < 4)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* q = (__bridge MPSGraphTensor*)ctx.resolve(q_id);
        MPSGraphTensor* k = (__bridge MPSGraphTensor*)ctx.resolve(k_id);
        MPSGraphTensor* v = (__bridge MPSGraphTensor*)ctx.resolve(v_id);
        if (g == nil || q == nil || k == nil || v == nil)
            return false;
        NSUInteger nd_k = k.shape.count;
        if (nd_k < 2)
            return false;
        MPSGraphTensor* k_t = [g transposeTensor:k
                                       dimension:(NSInteger)(nd_k - 1)
                                   withDimension:(NSInteger)(nd_k - 2)
                                            name:nil];
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q
                                                          secondaryTensor:k_t
                                                                     name:@"sdpa_qk"];
        double scale_val = scale;
        if (scale_val == 0.0) {
            NSUInteger nd_q = q.shape.count;
            if (nd_q < 1)
                return false;
            double Dk = (double)q.shape[nd_q - 1].longLongValue;
            if (Dk <= 0.0)
                return false;
            scale_val = 1.0 / std::sqrt(Dk);
        }
        MPSGraphTensor* scale_c = [g constantWithScalar:scale_val dataType:scores.dataType];
        scores = [g multiplicationWithPrimaryTensor:scores secondaryTensor:scale_c name:nil];
        if (has_mask && node.inputs.size() >= 4) {
            TensorId m_id = node.inputs[3];
            if (m_id >= 0) {
                MPSGraphTensor* m = (__bridge MPSGraphTensor*)ctx.resolve(m_id);
                if (m != nil) {
                    if (m.dataType != scores.dataType) {
                        m = [g castTensor:m toType:scores.dataType name:nil];
                    }
                    scores = [g additionWithPrimaryTensor:scores secondaryTensor:m name:nil];
                }
            }
        }
        // Causal masking — applied only when no explicit additive mask is
        // wired (the eager backend lets a float ``attn_mask`` win over
        // ``is_causal``).  Build a lower-triangular −∞ additive mask over the
        // trailing (Lq, Lk) score axes and broadcast it across batch/head.
        if (is_causal && !has_mask) {
            NSUInteger nd_c = scores.shape.count;
            if (nd_c < 2)
                return false;
            const long long Lq = scores.shape[nd_c - 2].longLongValue;
            const long long Lk = scores.shape[nd_c - 1].longLongValue;
            if (Lq <= 0 || Lk <= 0)
                return false;
            // Per-axis index vectors (host iota, O(Lq+Lk)).
            std::vector<float> row_data((std::size_t)Lq), col_data((std::size_t)Lk);
            for (long long i = 0; i < Lq; ++i)
                row_data[(std::size_t)i] = (float)i;
            for (long long j = 0; j < Lk; ++j)
                col_data[(std::size_t)j] = (float)j;
            NSData* row_nsd = [NSData dataWithBytes:row_data.data()
                                             length:row_data.size() * sizeof(float)];
            NSData* col_nsd = [NSData dataWithBytes:col_data.data()
                                             length:col_data.size() * sizeof(float)];
            MPSGraphTensor* rows = [g constantWithData:row_nsd
                                                 shape:@[ [NSNumber numberWithLongLong:Lq], @1 ]
                                              dataType:MPSDataTypeFloat32];
            MPSGraphTensor* cols = [g constantWithData:col_nsd
                                                 shape:@[ @1, [NSNumber numberWithLongLong:Lk] ]
                                              dataType:MPSDataTypeFloat32];
            // Keep key j for query i iff  j ≤ i + (Lk − Lq)  (bottom-right
            // aligned; reduces to lower-triangular when Lq == Lk).
            MPSGraphTensor* offset_c = [g constantWithScalar:(double)(Lk - Lq)
                                                    dataType:MPSDataTypeFloat32];
            MPSGraphTensor* rows_off = [g additionWithPrimaryTensor:rows
                                                    secondaryTensor:offset_c
                                                               name:nil];
            MPSGraphTensor* keep = [g lessThanOrEqualToWithPrimaryTensor:cols
                                                         secondaryTensor:rows_off
                                                                    name:nil];
            MPSGraphTensor* zero_c = [g constantWithScalar:0.0 dataType:scores.dataType];
            MPSGraphTensor* neg_inf_c =
                [g constantWithScalar:-std::numeric_limits<double>::infinity()
                             dataType:scores.dataType];
            MPSGraphTensor* causal_mask = [g selectWithPredicateTensor:keep
                                                   truePredicateTensor:zero_c
                                                  falsePredicateTensor:neg_inf_c
                                                                  name:@"sdpa_causal_mask"];
            scores = [g additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
        }
        NSUInteger nd_s = scores.shape.count;
        MPSGraphTensor* attn = [g softMaxWithTensor:scores
                                               axis:(NSInteger)(nd_s - 1)
                                               name:@"sdpa_softmax"];
        ctx.bind(node.outputs[0].id,
                 (__bridge void*)([g matrixMultiplicationWithPrimaryTensor:attn
                                                           secondaryTensor:v
                                                                      name:@"sdpa_av"]));
        return true;
    }
};

struct AttentionRegistrar {
    AttentionRegistrar() { register_emitter(std::make_unique<SdpaEmitter>()); }
};

[[maybe_unused]] static const AttentionRegistrar g_attention_registrar;

}  // namespace

}  // namespace lucid::compile
