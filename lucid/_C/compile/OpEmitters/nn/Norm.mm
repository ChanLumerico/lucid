// lucid/_C/compile/OpEmitters/Norm.mm
//
// LayerNorm + RMSNorm emitters.  BatchNorm / GroupNorm carry running
// stats / channel splits that need a longer composition; deferred to
// the next chunk.
//
// LayerNorm:
//     y = ((x - mean) / sqrt(var + eps)) * gamma + beta
//   reduction axes = trailing K dims (K = gamma.rank())
//
// RMSNorm:
//     y = x / sqrt(mean(x²) + eps) * gamma
//   reduction axes = trailing K dims (K = gamma.rank())
//
// Each op reports a single double "eps" attribute via OpScopeFull::set_attr.
// The reduction axes are inferred from the trailing dim count of
// the gamma input tensor's MPSGraph shape.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

inline double eps_attr(const OpNode& node) {
    auto it = node.attrs.find("eps");
    if (it == node.attrs.end())
        return -1.0;
    const auto* p = std::get_if<double>(&it->second);
    return p ? *p : -1.0;
}

// Build [rank-K, rank-K+1, …, rank-1] as the trailing-K axes list.
inline NSArray<NSNumber*>* trailing_axes(NSUInteger rank, NSUInteger K) {
    NSMutableArray<NSNumber*>* axes = [NSMutableArray arrayWithCapacity:K];
    for (NSUInteger i = rank - K; i < rank; ++i)
        [axes addObject:[NSNumber numberWithUnsignedInteger:i]];
    return axes;
}

class LayerNormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "layer_norm"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || g_id < 0 || b_id < 0)
            return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(g_id);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (x_t == nil || g_t == nil || b_t == nil || graph == nil)
            return false;

        NSUInteger x_rank = x_t.shape.count;
        NSUInteger g_rank = g_t.shape.count;
        if (g_rank == 0 || g_rank > x_rank)
            return false;
        NSArray<NSNumber*>* axes = trailing_axes(x_rank, g_rank);

        MPSGraphTensor* mean = [graph meanOfTensor:x_t axes:axes name:nil];
        MPSGraphTensor* var = [graph varianceOfTensor:x_t axes:axes name:nil];
        MPSGraphTensor* y =
            [graph normalizationWithTensor:x_t
                                meanTensor:mean
                            varianceTensor:var
                               gammaTensor:g_t
                                betaTensor:b_t
                                   epsilon:static_cast<float>(eps)
                                      name:@"layer_norm"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

class RMSNormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "rms_norm"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        if (x_id < 0 || g_id < 0)
            return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(g_id);
        if (x_t == nil || g_t == nil || graph == nil)
            return false;

        NSUInteger x_rank = x_t.shape.count;
        NSUInteger g_rank = g_t.shape.count;
        if (g_rank == 0 || g_rank > x_rank)
            return false;
        NSArray<NSNumber*>* axes = trailing_axes(x_rank, g_rank);

        MPSDataType dt = x_t.dataType;
        MPSGraphTensor* eps_t =
            [graph constantWithScalar:eps dataType:dt];

        // RMS: rstd = 1 / sqrt(mean(x²) + eps)
        MPSGraphTensor* x_sq =
            [graph multiplicationWithPrimaryTensor:x_t secondaryTensor:x_t name:nil];
        MPSGraphTensor* mean_sq = [graph meanOfTensor:x_sq axes:axes name:nil];
        MPSGraphTensor* var_plus_eps =
            [graph additionWithPrimaryTensor:mean_sq secondaryTensor:eps_t name:nil];
        MPSGraphTensor* rstd =
            [graph reciprocalSquareRootWithTensor:var_plus_eps name:nil];
        MPSGraphTensor* x_scaled =
            [graph multiplicationWithPrimaryTensor:x_t secondaryTensor:rstd name:nil];
        MPSGraphTensor* y =
            [graph multiplicationWithPrimaryTensor:x_scaled
                                   secondaryTensor:g_t
                                              name:@"rms_norm"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// BatchNormEval — inference-mode BN (no running-stats update).
// Inputs: x (N, C, H, W), running_mean (C,), running_var (C,), gamma (C,), beta (C,).
// Reduction axes: implicit on the channel dim.
// y_nchw = ((x - mean[None,:,None,None]) / sqrt(var[None,:,None,None] + eps)) *
//          gamma[None,:,None,None] + beta[None,:,None,None]
class BatchNormEvalEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "batch_norm_eval"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 5)
            return false;
        for (TensorId id : node.inputs)
            if (id < 0)
                return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* m_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        MPSGraphTensor* v_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[3]);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[4]);
        if (x_t == nil || m_t == nil || v_t == nil || g_t == nil || b_t == nil || graph == nil)
            return false;

        // Reshape (C,) → (1, C, 1, …, 1) so the affine + normalisation
        // broadcast cleanly against an N-D NCHW input.  Channel-dim size
        // is read from the input ``x_t`` (NCHW) so we don't depend on
        // the running-stats placeholder rank — a placeholder might
        // arrive as (1, C) instead of (C,) after a cast or reshape op.
        NSUInteger rank = x_t.shape.count;
        if (rank < 2)
            return false;
        NSMutableArray<NSNumber*>* affine_shape = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger i = 0; i < rank; ++i)
            [affine_shape addObject:[NSNumber numberWithLongLong:1]];
        affine_shape[1] = x_t.shape[1];  // channel dim from NCHW input

        MPSGraphTensor* mean = [graph reshapeTensor:m_t withShape:affine_shape name:nil];
        MPSGraphTensor* var = [graph reshapeTensor:v_t withShape:affine_shape name:nil];
        MPSGraphTensor* gamma = [graph reshapeTensor:g_t withShape:affine_shape name:nil];
        MPSGraphTensor* beta = [graph reshapeTensor:b_t withShape:affine_shape name:nil];

        MPSGraphTensor* y =
            [graph normalizationWithTensor:x_t
                                meanTensor:mean
                            varianceTensor:var
                               gammaTensor:gamma
                                betaTensor:beta
                                   epsilon:static_cast<float>(eps)
                                      name:@"batch_norm_eval"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// BatchNorm train mode — inputs (x, gamma, beta), reduce axes = all
// except channel dim 1.  Engine schema "batch_norm" (BatchNorm2dBackward).
class BatchNormTrainEmitter final : public OpEmitter {
public:
    explicit BatchNormTrainEmitter(std::string_view n) : name_(n) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3)
            return false;
        for (TensorId id : node.inputs)
            if (id < 0)
                return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
        if (x_t == nil || g_t == nil || b_t == nil || graph == nil)
            return false;

        NSUInteger rank = x_t.shape.count;
        if (rank < 2)
            return false;

        // Reduction over all axes except channel dim 1.
        NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
        for (NSUInteger i = 0; i < rank; ++i)
            if (i != 1)
                [reduce_axes addObject:[NSNumber numberWithUnsignedInteger:i]];

        MPSGraphTensor* mean = [graph meanOfTensor:x_t axes:reduce_axes name:nil];
        MPSGraphTensor* var = [graph varianceOfTensor:x_t axes:reduce_axes name:nil];

        // Reshape gamma / beta (C,) → (1, C, 1, ...).
        NSMutableArray<NSNumber*>* affine_shape = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger i = 0; i < rank; ++i)
            [affine_shape addObject:[NSNumber numberWithLongLong:1]];
        affine_shape[1] = x_t.shape[1];
        MPSGraphTensor* gamma = [graph reshapeTensor:g_t withShape:affine_shape name:nil];
        MPSGraphTensor* beta = [graph reshapeTensor:b_t withShape:affine_shape name:nil];

        MPSGraphTensor* y =
            [graph normalizationWithTensor:x_t
                                meanTensor:mean
                            varianceTensor:var
                               gammaTensor:gamma
                                betaTensor:beta
                                   epsilon:static_cast<float>(eps)
                                      name:@"batch_norm"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }

private:
    std::string name_;
};

// GroupNorm — inputs (x, gamma, beta), splits channels into G groups,
// reduces over (per-sample, per-group, all spatial).
// Implementation: reshape x to (N, G, C/G, *spatial), reduce over
// (G-index axis + spatial), normalize, reshape back, then apply
// per-channel gamma/beta broadcast.
class GroupNormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "group_norm"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3)
            return false;
        for (TensorId id : node.inputs)
            if (id < 0)
                return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;
        auto it = node.attrs.find("num_groups");
        if (it == node.attrs.end())
            return false;
        const auto* G_p = std::get_if<std::int64_t>(&it->second);
        if (G_p == nullptr || *G_p <= 0)
            return false;
        const NSInteger G = static_cast<NSInteger>(*G_p);

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
        if (x_t == nil || g_t == nil || b_t == nil || graph == nil)
            return false;

        NSArray<NSNumber*>* x_shape = x_t.shape;
        NSUInteger rank = x_shape.count;
        if (rank < 2)
            return false;
        const NSInteger N = [x_shape[0] integerValue];
        const NSInteger C = [x_shape[1] integerValue];
        if (C % G != 0)
            return false;
        const NSInteger C_per_G = C / G;

        // Reshape x from (N, C, *spatial) → (N, G, C/G, *spatial).
        NSMutableArray<NSNumber*>* grouped_shape =
            [NSMutableArray arrayWithCapacity:rank + 1];
        [grouped_shape addObject:@(N)];
        [grouped_shape addObject:@(G)];
        [grouped_shape addObject:@(C_per_G)];
        for (NSUInteger i = 2; i < rank; ++i)
            [grouped_shape addObject:x_shape[i]];
        MPSGraphTensor* x_grouped =
            [graph reshapeTensor:x_t withShape:grouped_shape name:nil];

        // Reduction axes = [2, 3, …, rank] (C/G + spatial).  Group axis (1)
        // and batch axis (0) are kept.
        NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
        for (NSUInteger i = 2; i < grouped_shape.count; ++i)
            [reduce_axes addObject:[NSNumber numberWithUnsignedInteger:i]];
        MPSGraphTensor* mean =
            [graph meanOfTensor:x_grouped axes:reduce_axes name:nil];
        MPSGraphTensor* var =
            [graph varianceOfTensor:x_grouped axes:reduce_axes name:nil];

        // Per-group normalisation (no affine yet).  Pass identity gamma=1, beta=0
        // (we'll multiply per-channel gamma/beta after the reshape back).
        MPSDataType dt = x_t.dataType;
        MPSGraphTensor* one_t = [graph constantWithScalar:1.0 dataType:dt];
        MPSGraphTensor* zero_t = [graph constantWithScalar:0.0 dataType:dt];
        MPSGraphTensor* normed_grouped =
            [graph normalizationWithTensor:x_grouped
                                meanTensor:mean
                            varianceTensor:var
                               gammaTensor:one_t
                                betaTensor:zero_t
                                   epsilon:static_cast<float>(eps)
                                      name:nil];

        // Reshape back to original (N, C, *spatial).
        MPSGraphTensor* normed =
            [graph reshapeTensor:normed_grouped withShape:x_shape name:nil];

        // Apply per-channel gamma/beta broadcast.
        NSMutableArray<NSNumber*>* affine_shape =
            [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger i = 0; i < rank; ++i)
            [affine_shape addObject:[NSNumber numberWithLongLong:1]];
        affine_shape[1] = x_shape[1];
        MPSGraphTensor* gamma_r =
            [graph reshapeTensor:g_t withShape:affine_shape name:nil];
        MPSGraphTensor* beta_r =
            [graph reshapeTensor:b_t withShape:affine_shape name:nil];
        MPSGraphTensor* scaled =
            [graph multiplicationWithPrimaryTensor:normed
                                   secondaryTensor:gamma_r
                                              name:nil];
        MPSGraphTensor* y =
            [graph additionWithPrimaryTensor:scaled
                              secondaryTensor:beta_r
                                         name:@"group_norm"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// ── lp_normalize (ord=2 only) — x / max(‖x‖_2, eps) along ``axis``.
// Other orders (1, inf, generic p) defer to eager fallback.
class LpNormalizeEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "lp_normalize"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        double ord = double_attr(node, "ord", 2.0);
        if (ord != 2.0) return false;
        std::int64_t axis = int_attr(node, "axis", -1);
        double eps = double_attr(node, "eps", 1e-12);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSInteger nd = (NSInteger)x.shape.count;
        if (axis < 0) axis += nd;
        if (axis < 0 || axis >= nd) return false;
        MPSGraphTensor* x_sq = [g squareWithTensor:x name:nil];
        MPSGraphTensor* sum = [g reductionSumWithTensor:x_sq
                                                    axis:(NSInteger)axis
                                                    name:nil];
        MPSGraphTensor* norm = [g squareRootWithTensor:sum name:nil];
        MPSGraphTensor* eps_c = [g constantWithScalar:eps dataType:norm.dataType];
        MPSGraphTensor* denom =
            [g maximumWithPrimaryTensor:norm secondaryTensor:eps_c name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g divisionWithPrimaryTensor:x
                                             secondaryTensor:denom
                                                        name:@"lp_normalize"]));
        return true;
    }
};

// ── global_response_norm — ConvNeXt-V2 GRN.
// Inputs: {x: (N, C, H, W), gamma: (C,), beta: (C,)}.  Formula
// matching :file:`CpuBackend.h::grn_forward`::
//
//     Gx[b, c]   = sqrt(Σ_s x[b, c, s]^2)            # L2 over (H, W)
//     mean[b]    = Σ_c Gx[b, c] / C
//     Nx[b, c]   = Gx[b, c] / (mean[b] + eps)
//     y[b, c, s] = x[b, c, s] * (gamma[c] * Nx[b, c] + beta[c])
class GlobalResponseNormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "global_response_norm"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 3 || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || g_id < 0 || b_id < 0) return false;
        double eps = double_attr(node, "eps", 1e-6);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* gamma = (__bridge MPSGraphTensor*)ctx.resolve(g_id);
        MPSGraphTensor* beta = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (g == nil || x == nil || gamma == nil || beta == nil) return false;
        NSArray<NSNumber*>* x_sh = x.shape;
        if (x_sh.count != 4) return false;
        NSNumber* C = x_sh[1];
        MPSGraphTensor* x_sq = [g squareWithTensor:x name:nil];
        MPSGraphTensor* sum_sp =
            [g reductionSumWithTensor:x_sq axes:@[@2, @3] name:nil];
        sum_sp = [g reshapeTensor:sum_sp
                        withShape:@[x_sh[0], C, @1, @1]
                             name:nil];
        MPSGraphTensor* Gx = [g squareRootWithTensor:sum_sp name:nil];
        MPSGraphTensor* mean = [g meanOfTensor:Gx axes:@[@1] name:nil];
        mean = [g reshapeTensor:mean
                      withShape:@[x_sh[0], @1, @1, @1]
                           name:nil];
        MPSGraphTensor* eps_c = [g constantWithScalar:eps dataType:mean.dataType];
        MPSGraphTensor* denom =
            [g additionWithPrimaryTensor:mean secondaryTensor:eps_c name:nil];
        MPSGraphTensor* Nx =
            [g divisionWithPrimaryTensor:Gx secondaryTensor:denom name:nil];
        NSArray<NSNumber*>* g_sh = @[@1, C, @1, @1];
        MPSGraphTensor* gamma_r = [g reshapeTensor:gamma withShape:g_sh name:nil];
        MPSGraphTensor* beta_r = [g reshapeTensor:beta withShape:g_sh name:nil];
        MPSGraphTensor* term1 =
            [g multiplicationWithPrimaryTensor:gamma_r secondaryTensor:Nx name:nil];
        MPSGraphTensor* scale =
            [g additionWithPrimaryTensor:term1 secondaryTensor:beta_r name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g multiplicationWithPrimaryTensor:x
                                                   secondaryTensor:scale
                                                              name:@"grn"]));
        return true;
    }
};

struct NormEmitterRegistrar {
    NormEmitterRegistrar() {
        register_emitter(std::make_unique<LayerNormEmitter>());
        register_emitter(std::make_unique<RMSNormEmitter>());
        register_emitter(std::make_unique<BatchNormEvalEmitter>());
        // BN train: register all three N-dim names since they share the
        // same forward path (Lucid uses N=1/2/3 explicit schemas).
        register_emitter(std::make_unique<BatchNormTrainEmitter>("batch_norm"));
        register_emitter(std::make_unique<BatchNormTrainEmitter>("batch_norm1d"));
        register_emitter(std::make_unique<BatchNormTrainEmitter>("batch_norm3d"));
        register_emitter(std::make_unique<GroupNormEmitter>());
        register_emitter(std::make_unique<LpNormalizeEmitter>());
        register_emitter(std::make_unique<GlobalResponseNormEmitter>());
    }
};

[[maybe_unused]] static const NormEmitterRegistrar g_norm_registrar;

}  // namespace

}  // namespace lucid::compile
