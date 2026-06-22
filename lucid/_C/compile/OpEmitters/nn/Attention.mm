// lucid/_C/compile/OpEmitters/nn/Attention.mm
//
// scaled_dot_product_attention emitter.
//
// Inputs: ``{q, k, v}`` plus an optional ``{attn_mask}``.  Attrs:
//   - ``scale``     (double) — multiplier on Q·Kᵀ; 0.0 means "use
//     1/√D_k" per Lucid convention
//   - ``is_causal`` (bool)   — lower-triangular causal masking
//   - ``has_mask``  (bool)   — whether an additive mask is wired
//
// FAST PATH — Apple's *fused* SDPA kernel.  For the universal 4-D
// ``[B, H, L, D]`` case (every transformer) we emit a single
// ``scaledDotProductAttentionWithQueryTensor:...descriptor:`` op
// (``MPSGraphSDPADescriptor`` on macOS 26: ``scale`` + additive ``maskTensor``
// + ``isCausal``).  This computes ``softmax(scale·QKᵀ + M)V`` in one fused
// flash-style kernel — matching the eager MLX fused SDPA — instead of the
// decomposed ``matmul + scale + (mask) + softmax + matmul`` below, which is
// measurably *slower* than eager (1.1–1.4× at L≥128) and was dragging down
// compiled-transformer speedup.
//
// DECOMPOSED FALLBACK — kept for non-4-D inputs and for non-square causal
// (where the fused ``isCausal`` alignment for ``Lq != Lk`` is unspecified; the
// host-baked mask below uses the bottom-right alignment ``j ≤ i + (Lk − Lq)``
// that matches eager).  ``is_causal`` is realised as a host-baked
// large-finite-negative ``constantWithData`` mask (NOT in-graph select / −inf,
// which crash the MPSGraph compiler on some drivers).  An explicit additive
// mask takes precedence over ``is_causal`` in both paths (mirrors eager).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
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
        const NSUInteger nd_q = q.shape.count;
        const NSUInteger nd_k = k.shape.count;
        const NSUInteger nd_v = v.shape.count;
        if (nd_k < 2)
            return false;

        // Effective scale: 0.0 ⇒ 1/√D_k per Lucid convention.
        double scale_val = scale;
        if (scale_val == 0.0) {
            if (nd_q < 1)
                return false;
            const double Dk = (double)q.shape[nd_q - 1].longLongValue;
            if (Dk <= 0.0)
                return false;
            scale_val = 1.0 / std::sqrt(Dk);
        }

        // Resolve the additive mask once (float only — the eager backend treats
        // a bool mask as a set-mask, not an add, so those route to eager).
        MPSGraphTensor* mask = nil;
        if (has_mask && node.inputs.size() >= 4) {
            const TensorId m_id = node.inputs[3];
            if (m_id < 0)
                return false;
            mask = (__bridge MPSGraphTensor*)ctx.resolve(m_id);
            if (mask == nil)
                return false;
            if ((mask.dataType & MPSDataTypeFloatBit) == 0)
                return false;
        }

        // ── Fused fast path: 4-D q/k/v (every transformer).  One Apple flash-
        // style SDPA kernel instead of the decomposed graph below.  isCausal is
        // used only for the square (Lq == Lk) case — the fused op's alignment
        // for Lq != Lk is unspecified, so non-square causal falls through to the
        // host-baked bottom-right mask that provably matches eager.  An additive
        // mask takes precedence over causal (mirrors eager + the descriptor's
        // mutually-exclusive contract).
        bool can_fuse = (nd_q == 4 && nd_k == 4 && nd_v == 4);
        if (can_fuse && is_causal && mask == nil) {
            const long long Lq = q.shape[2].longLongValue;
            const long long Lk = k.shape[2].longLongValue;
            if (Lq != Lk)
                can_fuse = false;
        }
        if (can_fuse) {
            MPSGraphSDPADescriptor* desc =
                [MPSGraphSDPADescriptor descriptorWithScale:(float)scale_val];
            if (mask != nil) {
                MPSGraphTensor* mm = mask;
                if (mm.dataType != q.dataType)
                    mm = [g castTensor:mm toType:q.dataType name:nil];
                desc.maskTensor = mm;
            } else if (is_causal) {
                desc.isCausal = YES;
            }
            MPSGraphTensor* out = [g scaledDotProductAttentionWithQueryTensor:q
                                                                    keyTensor:k
                                                                  valueTensor:v
                                                                   descriptor:desc
                                                                         name:@"sdpa_fused"];
            if (out == nil)
                return false;
            ctx.bind(node.outputs[0].id, (__bridge void*)out);
            return true;
        }

        // ── Decomposed fallback (non-4-D / non-square causal). ──
        MPSGraphTensor* k_t = [g transposeTensor:k
                                       dimension:(NSInteger)(nd_k - 1)
                                   withDimension:(NSInteger)(nd_k - 2)
                                            name:nil];
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q
                                                          secondaryTensor:k_t
                                                                     name:@"sdpa_qk"];
        MPSGraphTensor* scale_c = [g constantWithScalar:scale_val dataType:scores.dataType];
        scores = [g multiplicationWithPrimaryTensor:scores secondaryTensor:scale_c name:nil];
        if (mask != nil) {
            MPSGraphTensor* m = mask;
            if (m.dataType != scores.dataType)
                m = [g castTensor:m toType:scores.dataType name:nil];
            scores = [g additionWithPrimaryTensor:scores secondaryTensor:m name:nil];
        }
        // Causal masking — applied only when no explicit additive mask is
        // wired (the eager backend lets a float ``attn_mask`` win over
        // ``is_causal``).  The (Lq, Lk) additive mask is fully precomputed on
        // the host and baked as a ``constantWithData`` tensor (the same proven
        // path the RoPE table uses) — deliberately NOT built in-graph with
        // ``select`` / compare / ``bandPart``, which crash the MPSGraph graph
        // compiler on some Metal drivers.  Only ``constantWithData`` + cast +
        // add reach the executable; those are exercised CI-wide.
        if (is_causal && !has_mask) {
            NSUInteger nd_c = scores.shape.count;
            if (nd_c < 2)
                return false;
            const long long Lq = scores.shape[nd_c - 2].longLongValue;
            const long long Lk = scores.shape[nd_c - 1].longLongValue;
            if (Lq <= 0 || Lk <= 0)
                return false;
            // Disallowed positions get a large *finite* negative (NOT −inf,
            // which can crash the graph compiler) whose exp() underflows to 0,
            // so it is softmax-equivalent.  f16 saturates at 65504, so the
            // half-precision score path uses a smaller magnitude.
            const float neg_big = (scores.dataType == MPSDataTypeFloat16) ? -6.0e4f : -1.0e30f;
            // Keep key j for query i iff  j ≤ i + (Lk − Lq)  (bottom-right
            // aligned; reduces to lower-triangular when Lq == Lk).
            const long long offset = Lk - Lq;
            std::vector<float> mask_data((std::size_t)(Lq * Lk));
            for (long long i = 0; i < Lq; ++i)
                for (long long j = 0; j < Lk; ++j)
                    mask_data[(std::size_t)(i * Lk + j)] = (j <= i + offset) ? 0.0f : neg_big;
            NSData* mask_nsd = [NSData dataWithBytes:mask_data.data()
                                              length:mask_data.size() * sizeof(float)];
            MPSGraphTensor* causal_mask =
                [g constantWithData:mask_nsd
                              shape:@[
                                  [NSNumber numberWithLongLong:Lq], [NSNumber numberWithLongLong:Lk]
                              ]
                           dataType:MPSDataTypeFloat32];
            if (causal_mask.dataType != scores.dataType)
                causal_mask = [g castTensor:causal_mask toType:scores.dataType name:nil];
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
