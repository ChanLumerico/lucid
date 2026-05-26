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
// ``is_causal=True`` would require constructing a lower-triangular
// −∞ mask sized to (Lq, Lk).  ``bandPart`` could provide it but the
// (Lq, Lk) inference from the trace metadata adds rough edges; for
// now the causal path returns nullptr → eager fallback.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <string_view>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

class SdpaEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "scaled_dot_product_attention"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 3 || node.outputs.empty()) return false;
        TensorId q_id = node.inputs[0];
        TensorId k_id = node.inputs[1];
        TensorId v_id = node.inputs[2];
        if (q_id < 0 || k_id < 0 || v_id < 0) return false;
        const bool has_mask = bool_attr(node, "has_mask", false);
        const bool is_causal = bool_attr(node, "is_causal", false);
        const double scale = double_attr(node, "scale", 0.0);
        if (is_causal) return false;  // causal path defers to eager
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* q = (__bridge MPSGraphTensor*)ctx.resolve(q_id);
        MPSGraphTensor* k = (__bridge MPSGraphTensor*)ctx.resolve(k_id);
        MPSGraphTensor* v = (__bridge MPSGraphTensor*)ctx.resolve(v_id);
        if (g == nil || q == nil || k == nil || v == nil) return false;
        NSUInteger nd_k = k.shape.count;
        if (nd_k < 2) return false;
        MPSGraphTensor* k_t = [g transposeTensor:k
                                       dimension:(NSInteger)(nd_k - 1)
                                   withDimension:(NSInteger)(nd_k - 2)
                                            name:nil];
        MPSGraphTensor* scores =
            [g matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k_t name:@"sdpa_qk"];
        double scale_val = scale;
        if (scale_val == 0.0) {
            NSUInteger nd_q = q.shape.count;
            if (nd_q < 1) return false;
            double Dk = (double)q.shape[nd_q - 1].longLongValue;
            if (Dk <= 0.0) return false;
            scale_val = 1.0 / std::sqrt(Dk);
        }
        MPSGraphTensor* scale_c =
            [g constantWithScalar:scale_val dataType:scores.dataType];
        scores = [g multiplicationWithPrimaryTensor:scores
                                    secondaryTensor:scale_c
                                               name:nil];
        if (has_mask && node.inputs.size() >= 4) {
            TensorId m_id = node.inputs[3];
            if (m_id >= 0) {
                MPSGraphTensor* m = (__bridge MPSGraphTensor*)ctx.resolve(m_id);
                if (m != nil) {
                    if (m.dataType != scores.dataType) {
                        m = [g castTensor:m toType:scores.dataType name:nil];
                    }
                    scores =
                        [g additionWithPrimaryTensor:scores secondaryTensor:m name:nil];
                }
            }
        }
        NSUInteger nd_s = scores.shape.count;
        MPSGraphTensor* attn = [g softMaxWithTensor:scores
                                                axis:(NSInteger)(nd_s - 1)
                                                name:@"sdpa_softmax"];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g matrixMultiplicationWithPrimaryTensor:attn
                                                        secondaryTensor:v
                                                                   name:@"sdpa_av"]));
        return true;
    }
};

struct AttentionRegistrar {
    AttentionRegistrar() {
        register_emitter(std::make_unique<SdpaEmitter>());
    }
};

[[maybe_unused]] static const AttentionRegistrar g_attention_registrar;

}  // namespace

}  // namespace lucid::compile
