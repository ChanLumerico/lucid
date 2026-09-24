// lucid/_C/compile/VjpEmitters/nn/Attention.mm
//
// Scaled dot-product attention VJP.
//
// Without it every transformer trained compiled through MPSGraph's
// autodiff, and once that fallback was restricted to ops measured to
// differentiate correctly, ViT, CoaT, CrossViT and CvT ran their training
// step eager.  The forward (``OpEmitters/nn/Attention.mm``) does not bind
// the attention probabilities to a trace id, so they are rebuilt here the
// same way — scores, the same additive or causal mask, softmax — and the
// standard backward follows:
//
//   dV = Pᵀ · dO          dP = dO · Vᵀ
//   dS = P ⊙ (dP − Σ_k dP ⊙ P)
//   dQ = dS · K · scale   dK = dSᵀ · Q · scale
//
// An additive mask takes dS (before the scale) — it may be a learnt bias.
// Leading batch / head axes that broadcast between q, k, v and the mask are
// summed back to each input's own shape.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

bool attr_bool(const OpNode& node, const char* key, bool def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    if (const auto* p = std::get_if<bool>(&it->second))
        return *p;
    if (const auto* p = std::get_if<std::int64_t>(&it->second))
        return *p != 0;
    return def;
}

double attr_double(const OpNode& node, const char* key, double def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    const auto* p = std::get_if<double>(&it->second);
    return p ? *p : def;
}

MPSGraphTensor* swap_last2(MPSGraph* g, MPSGraphTensor* t) {
    const NSInteger n = (NSInteger)t.shape.count;
    return [g transposeTensor:t dimension:n - 1 withDimension:n - 2 name:nil];
}

class SdpaVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "scaled_dot_product_attention"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() < 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        const bool has_mask = attr_bool(node, "has_mask", false);
        const bool is_causal = attr_bool(node, "is_causal", false);
        if (has_mask && node.inputs.size() < 4)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        const MPSDataType dt = go.dataType;
        MPSGraphTensor* q = as_tensor(bctx.forward(node.inputs[0]));
        MPSGraphTensor* k = as_tensor(bctx.forward(node.inputs[1]));
        MPSGraphTensor* v = as_tensor(bctx.forward(node.inputs[2]));
        if (g == nil || go == nil || q == nil || k == nil || v == nil)
            return false;
        q = cast_if_needed(g, q, dt);
        k = cast_if_needed(g, k, dt);
        v = cast_if_needed(g, v, dt);
        if (q.shape.count < 2 || k.shape.count < 2 || v.shape.count < 2)
            return false;

        double scale = attr_double(node, "scale", 0.0);
        if (scale == 0.0) {
            const double dk = (double)q.shape[q.shape.count - 1].longLongValue;
            if (dk <= 0.0)
                return false;
            scale = 1.0 / std::sqrt(dk);
        }
        MPSGraphTensor* scale_c = [g constantWithScalar:scale dataType:dt];

        // Rebuild the forward's probabilities, mask for mask.
        MPSGraphTensor* scores = [g multiplicationWithPrimaryTensor:
                                        [g matrixMultiplicationWithPrimaryTensor:q
                                                                 secondaryTensor:swap_last2(g, k)
                                                                            name:nil]
                                                    secondaryTensor:scale_c
                                                               name:nil];
        if (has_mask) {
            MPSGraphTensor* m = as_tensor(bctx.forward(node.inputs[3]));
            if (m == nil || (m.dataType & MPSDataTypeFloatBit) == 0)
                return false;
            scores = [g additionWithPrimaryTensor:scores
                                  secondaryTensor:cast_if_needed(g, m, dt)
                                             name:nil];
        } else if (is_causal) {
            const NSUInteger n = scores.shape.count;
            const long long lq = scores.shape[n - 2].longLongValue;
            const long long lk = scores.shape[n - 1].longLongValue;
            if (lq <= 0 || lk <= 0)
                return false;
            // Same host-built constant as the forward — see there for why it
            // is not a select, and why the fill is finite.
            const float neg_big = dt == MPSDataTypeFloat16 ? -6.0e4f : -1.0e30f;
            std::vector<float> mask((std::size_t)(lq * lk));
            for (long long i = 0; i < lq; ++i)
                for (long long j = 0; j < lk; ++j)
                    mask[(std::size_t)(i * lk + j)] = j <= i + (lk - lq) ? 0.0f : neg_big;
            MPSGraphTensor* cm =
                [g constantWithData:[NSData dataWithBytes:mask.data()
                                                   length:mask.size() * sizeof(float)]
                              shape:@[ @(lq), @(lk) ]
                           dataType:MPSDataTypeFloat32];
            scores = [g additionWithPrimaryTensor:scores
                                  secondaryTensor:cast_if_needed(g, cm, dt)
                                             name:nil];
        }
        const NSInteger last = (NSInteger)scores.shape.count - 1;
        MPSGraphTensor* p = [g softMaxWithTensor:scores axis:last name:nil];

        MPSGraphTensor* dv = [g matrixMultiplicationWithPrimaryTensor:swap_last2(g, p)
                                                      secondaryTensor:go
                                                                 name:nil];
        MPSGraphTensor* dp = [g matrixMultiplicationWithPrimaryTensor:go
                                                      secondaryTensor:swap_last2(g, v)
                                                                 name:nil];
        MPSGraphTensor* row = [g reductionSumWithTensor:[g multiplicationWithPrimaryTensor:dp
                                                                           secondaryTensor:p
                                                                                      name:nil]
                                                   axis:last
                                                   name:nil];
        MPSGraphTensor* ds = [g multiplicationWithPrimaryTensor:p
                                                secondaryTensor:[g subtractionWithPrimaryTensor:dp
                                                                                secondaryTensor:row
                                                                                           name:nil]
                                                           name:nil];
        // An additive mask can be learnt — CoaT's relative-position bias is
        // one — and its gradient is dS before the scale.  A mask built from
        // constants has no parameter upstream and the walker skips it.
        if (has_mask && node.inputs[3] >= 0) {
            MPSGraphTensor* m = as_tensor(bctx.forward(node.inputs[3]));
            const auto want = shape_of_mps(m);
            const auto have = shape_of_mps(ds);
            void* dm = want == have ? from_tensor(ds) : bctx.unreduce(from_tensor(ds), want, have);
            if (dm == nullptr)
                return false;
            bctx.accumulate_grad(node.inputs[3], dm);
        }
        ds = [g multiplicationWithPrimaryTensor:ds secondaryTensor:scale_c name:nil];
        MPSGraphTensor* dq = [g matrixMultiplicationWithPrimaryTensor:ds
                                                      secondaryTensor:k
                                                                 name:nil];
        MPSGraphTensor* dk = [g matrixMultiplicationWithPrimaryTensor:swap_last2(g, ds)
                                                      secondaryTensor:q
                                                                 name:nil];

        NSArray<MPSGraphTensor*>* grads = @[ dq, dk, dv ];
        NSArray<MPSGraphTensor*>* fwd = @[ q, k, v ];
        for (NSUInteger i = 0; i < 3; ++i) {
            const TensorId id = node.inputs[i];
            if (id < 0)
                continue;
            MPSGraphTensor* gi = grads[i];
            const auto want = shape_of_mps(fwd[i]);
            const auto have = shape_of_mps(gi);
            void* out = want == have ? from_tensor(gi) : bctx.unreduce(from_tensor(gi), want, have);
            if (out == nullptr)
                return false;
            bctx.accumulate_grad(id, out);
        }
        return true;
    }
};

struct AttentionVjpRegistrar {
    AttentionVjpRegistrar() { register_vjp_emitter(std::make_unique<SdpaVjp>()); }
};

[[maybe_unused]] static const AttentionVjpRegistrar g_attention_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
