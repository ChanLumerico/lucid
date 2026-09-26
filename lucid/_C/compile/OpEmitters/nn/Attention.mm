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
// ``is_causal=True`` adds a lower-triangular large-negative additive mask
// before the softmax.  The (Lq, Lk) mask is precomputed on the host and baked
// as a ``constantWithData`` tensor (the proven RoPE-table path) rather than
// built in-graph with ``select`` / compare / ``bandPart`` — those crash the
// MPSGraph graph compiler on some Metal drivers.  The masked value is a large
// *finite* negative (−inf likewise crashes the compiler) whose exp() underflows
// to 0, so it is softmax-equivalent.  The engine records ``is_causal`` only
// for a square score matrix with no other mask — ``F.scaled_dot_product_attention``
// folds every other causal call into the additive mask, and the engine refuses
// one that reaches it directly — so the triangle here is always square.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstdlib>
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
        // rather than silently dropping the mask.
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
        if (MPSGraphTensor* fused = emit_fused(g, q, k, v, ctx, node, has_mask, is_causal, scale)) {
            ctx.bind(node.outputs[0].id, (__bridge void*)fused);
            return true;
        }
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
                if (m == nil)
                    return false;  // mask wired but unresolved → can't honor it
                // Only a floating (additive) mask is reproducible by an add:
                // a bool keep-mask selects rather than adds (−inf
                // where false), so route those to eager instead.
                if ((m.dataType & MPSDataTypeFloatBit) == 0)
                    return false;
                if (m.dataType != scores.dataType) {
                    m = [g castTensor:m toType:scores.dataType name:nil];
                }
                scores = [g additionWithPrimaryTensor:scores secondaryTensor:m name:nil];
            }
        }
        // Causal masking — the engine never records ``is_causal`` together
        // with a mask (see the header).  The (Lq, Lk) additive mask is fully precomputed on
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
            // Keep key j for query i iff  j ≤ i + (Lk − Lq): lower-triangular,
            // since the engine only records a square causal call.
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
        // ``attn @ v`` is the attention value-projection.  On GPUs affected by
        // the MPSGraph fused-attention bug (seq len in [17,24], batch >= 3 —
        // e.g. M1 Pro / macOS 26) MPSGraph pattern-matches it onto a kernel
        // that silently miscompiles, so emit it transposed
        // (``attn @ v == (vᵀ @ attnᵀ)ᵀ``) to break the match.  That costs
        // ~+70% on the attention path, so the capability probe gates it: on
        // unaffected hardware ``apply_attention_workaround()`` is false and we
        // emit the plain (fast) matmul.  (Mirrors MatmulEmitter for manual
        // attention.)
        //
        // A value head width unlike the key's (EfficientFormer: 128 against
        // 32) takes the transposed form everywhere: on macOS 26 MPSGraph
        // matches that attention too and then aborts the process in its
        // optimisation passes (``MLIR pass manager failed``) — an abort the
        // capability probe, which compares values, cannot observe.
        const NSUInteger nd_qk = q.shape.count;
        const NSUInteger nd_vv = v.shape.count;
        const bool widths_differ =
            nd_qk >= 1 && nd_vv >= 1 &&
            q.shape[nd_qk - 1].longLongValue != v.shape[nd_vv - 1].longLongValue;
        if (apply_attention_workaround() || widths_differ) {
            const NSUInteger nd_a = attn.shape.count;
            const NSUInteger nd_v2 = v.shape.count;
            MPSGraphTensor* attn_tr = [g transposeTensor:attn
                                               dimension:(NSInteger)(nd_a - 1)
                                           withDimension:(NSInteger)(nd_a - 2)
                                                    name:nil];
            MPSGraphTensor* v_tr = [g transposeTensor:v
                                            dimension:(NSInteger)(nd_v2 - 1)
                                        withDimension:(NSInteger)(nd_v2 - 2)
                                                 name:nil];
            MPSGraphTensor* av = [g matrixMultiplicationWithPrimaryTensor:v_tr
                                                          secondaryTensor:attn_tr
                                                                     name:@"sdpa_av"];
            const NSUInteger nd_av = av.shape.count;
            MPSGraphTensor* out = [g transposeTensor:av
                                           dimension:(NSInteger)(nd_av - 1)
                                       withDimension:(NSInteger)(nd_av - 2)
                                                name:@"sdpa_out"];
            ctx.bind(node.outputs[0].id, (__bridge void*)out);
            return true;
        }
        MPSGraphTensor* out = [g matrixMultiplicationWithPrimaryTensor:attn
                                                       secondaryTensor:v
                                                                  name:@"sdpa_av"];
        ctx.bind(node.outputs[0].id, (__bridge void*)out);
        return true;
    }

private:
    // MPSGraph's own fused attention (macOS 15+), opt-in, where it is known
    // to be right.  The decomposition below materialises the full score matrix;
    // at long key lengths that made a compiled attention block slower than
    // eager MLX's fused kernel (a Wan DiT block, KV = 18 720: 128 ms against
    // 95 ms).  Taken only when the capability probe has cleared this GPU of
    // the attention miscompile — the probe compiles attention with the
    // workaround off, so it exercises exactly this path on the known-bad
    // shapes — and only for the plain shape the fused kernel takes: rank-4
    // (B, H, L, D) operands, value and key heads of one width, a floating
    // additive mask if any.  Everything else keeps the decomposition.
    static MPSGraphTensor* emit_fused(MPSGraph* g,
                                      MPSGraphTensor* q,
                                      MPSGraphTensor* k,
                                      MPSGraphTensor* v,
                                      BuilderContext& ctx,
                                      const OpNode& node,
                                      bool has_mask,
                                      bool is_causal,
                                      double scale) {
        // Opt-in (``LUCID_COMPILE_FUSED_SDPA=1``) until it is measured to
        // pay: on an M1 Pro it matched the decomposition (3.47 against
        // 3.32 ms at KV = 4096, fp16) and neither caught eager MLX, so it is
        // not yet worth a second numeric path by default.  The gap the
        // decomposition leaves at long key lengths is tracked as CHA-14.
        static const bool enabled = [] {
            const char* env = std::getenv("LUCID_COMPILE_FUSED_SDPA");
            return env != nullptr && std::string_view(env) == "1";
        }();
        if (!enabled || apply_attention_workaround())
            return nil;
        if (q.shape.count != 4 || k.shape.count != 4 || v.shape.count != 4)
            return nil;
        if (q.shape[3].longLongValue != v.shape[3].longLongValue ||
            q.shape[3].longLongValue != k.shape[3].longLongValue)
            return nil;
        if (q.dataType != k.dataType || q.dataType != v.dataType)
            return nil;
        double scale_val = scale;
        if (scale_val == 0.0) {
            const double Dk = (double)q.shape[3].longLongValue;
            if (Dk <= 0.0)
                return nil;
            scale_val = 1.0 / std::sqrt(Dk);
        }
        MPSGraphTensor* mask = nil;
        if (has_mask) {
            if (node.inputs.size() < 4 || node.inputs[3] < 0)
                return nil;
            mask = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[3]);
            if (mask == nil || (mask.dataType & MPSDataTypeFloatBit) == 0)
                return nil;
            if (mask.dataType != q.dataType)
                mask = [g castTensor:mask toType:q.dataType name:nil];
        } else if (is_causal) {
            const long long Lq = q.shape[2].longLongValue;
            const long long Lk = k.shape[2].longLongValue;
            if (Lq <= 0 || Lk <= 0)
                return nil;
            const float neg_big = (q.dataType == MPSDataTypeFloat16) ? -6.0e4f : -1.0e30f;
            const long long offset = Lk - Lq;
            std::vector<float> mask_data((std::size_t)(Lq * Lk));
            for (long long i = 0; i < Lq; ++i)
                for (long long j = 0; j < Lk; ++j)
                    mask_data[(std::size_t)(i * Lk + j)] = (j <= i + offset) ? 0.0f : neg_big;
            NSData* bytes = [NSData dataWithBytes:mask_data.data()
                                           length:mask_data.size() * sizeof(float)];
            mask =
                [g constantWithData:bytes
                              shape:@[
                                  [NSNumber numberWithLongLong:Lq], [NSNumber numberWithLongLong:Lk]
                              ]
                           dataType:MPSDataTypeFloat32];
            if (mask.dataType != q.dataType)
                mask = [g castTensor:mask toType:q.dataType name:nil];
        }
        return [g scaledDotProductAttentionWithQueryTensor:q
                                                 keyTensor:k
                                               valueTensor:v
                                                maskTensor:mask
                                                     scale:(float)scale_val
                                                      name:@"sdpa_fused"];
    }
};

struct AttentionRegistrar {
    AttentionRegistrar() { register_emitter(std::make_unique<SdpaEmitter>()); }
};

[[maybe_unused]] static const AttentionRegistrar g_attention_registrar;

}  // namespace

}  // namespace lucid::compile
