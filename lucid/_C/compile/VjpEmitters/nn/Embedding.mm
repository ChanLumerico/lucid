// lucid/_C/compile/VjpEmitters/nn/Embedding.mm
//
// VJP for ``embedding``.  Forward: y = W[idx] where W is (V, D) and
// idx has arbitrary rank Si — output is Si + (D,).  Backward:
//
//   dW[v, :] += sum_{(b,n,...) where idx[b,n,...]=v} grad[b,n,...,:]
//
// In MPSGraph terms, that's a single ``scatterAlongAxis:0`` with
// mode Add:
//
//   base    = zeros(V, D)        // accumulator
//   M       = numel(idx)
//   idx_f   = reshape(idx, [M])
//   idx_2d  = broadcast(reshape(idx_f, [M, 1]), [M, D])
//   grad_2d = reshape(grad, [M, D])
//   dW      = scatterAlongAxis:0 base + grad_2d at idx_2d
//
// d(idx) = no gradient (idx is integer — the walker treats it as a
// grad-sink for free since we just don't accumulate onto it).
//
// padding_idx is currently ignored (Phase 5+ TODO).  When set, the
// rows with ``idx == padding_idx`` should contribute 0 — mask via
// ``select(equal(idx, padding_idx), zero, grad_2d)`` before the
// scatter.  The standard transformer training path doesn't use a
// padding_idx so this gap is harmless for the smoke surface.
//
// Why this is the centerpiece VJP: MPSGraph's
// ``gradientForPrimaryTensor:`` asserts on
// ``gatherWithUpdatesTensor:`` (the embedding forward) with
// ``"Not a predecessor of primaryTensor"``.  Manual VJP routes around
// the assertion entirely.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

class EmbeddingVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "embedding"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() < 2 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId w_id = node.inputs[0];
        TensorId i_id = node.inputs[1];
        if (w_id < 0 || i_id < 0) return false;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* W = as_tensor(bctx.forward(w_id));
        MPSGraphTensor* idx = as_tensor(bctx.forward(i_id));
        if (g == nil || grad == nil || W == nil || idx == nil) return false;

        // SDK guard — the API matches the forward scatter_add path.
        if (![g respondsToSelector:@selector(scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:)])
            return false;

        std::vector<std::int64_t> w_shape = shape_of_mps(W);
        if (w_shape.size() != 2) return false;
        const std::int64_t V = w_shape[0];
        const std::int64_t D = w_shape[1];

        // Compute M = ∏ idx.shape explicitly — MPSGraph's
        // broadcastTensor:toShape: drops -1 placeholders into the
        // downstream scatter signature and the verifier rejects the
        // resulting shape-mismatch.  Trace placeholders carry static
        // shapes by construction so this is fine.
        std::vector<std::int64_t> idx_shape = shape_of_mps(idx);
        if (idx_shape.empty()) return false;
        std::int64_t M = 1;
        for (std::int64_t d : idx_shape) M *= d;
        if (M <= 0) return false;

        // (idx_shape...) → (M,) → (M, 1)
        NSArray<NSNumber*>* idx_col_shape =
            @[ [NSNumber numberWithLongLong:M], @1 ];
        MPSGraphTensor* idx_col =
            [g reshapeTensor:idx withShape:idx_col_shape name:nil];
        // (M, 1) → (M, D)  — replicate the same scalar idx across D cols.
        std::vector<std::int64_t> idx_grid_shape{ M, D };
        NSArray<NSNumber*>* idx_grid_ns = shape_to_ns(idx_grid_shape);
        MPSGraphTensor* idx_grid =
            [g broadcastTensor:idx_col toShape:idx_grid_ns name:nil];

        // grad: (Si..., D) → (M, D).
        std::vector<std::int64_t> grad_grid_shape{ M, D };
        NSArray<NSNumber*>* grad_grid_ns = shape_to_ns(grad_grid_shape);
        MPSGraphTensor* grad_grid =
            [g reshapeTensor:grad withShape:grad_grid_ns name:nil];

        // base = zeros((V, D))   dtype = W.dtype.
        std::vector<std::int64_t> base_shape{ V, D };
        NSArray<NSNumber*>* base_ns = shape_to_ns(base_shape);
        MPSGraphTensor* base = [g constantWithScalar:0.0
                                                shape:base_ns
                                             dataType:W.dataType];

        MPSGraphTensor* dW =
            [g scatterAlongAxis:(NSInteger)0
                  withDataTensor:base
                   updatesTensor:grad_grid
                   indicesTensor:idx_grid
                            mode:MPSGraphScatterModeAdd
                            name:@"embedding_vjp_dW"];
        bctx.accumulate_grad(w_id, from_tensor(dW));
        // idx has no grad — leave it.
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// gather (narrow gather along given axis).
//
// Forward: y = gather(data, idx, axis).  data and idx have the same
// rank; y has the same rank but with idx's size on the gather axis.
//
// Backward is the dual scatter-add:
//   d(data)[a, ..., v, ..., z] += sum_{i where idx[a,...,i,...,z]=v}
//                                 grad[a, ..., i, ..., z]
// Implemented as ``scatterAlongAxis:axis mode:Add`` with
// data tensor = zeros_like(data), updates = grad, indices = idx.
// d(idx) = none.
// ────────────────────────────────────────────────────────────────────
class GatherVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "gather"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 2 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId d_id = node.inputs[0];
        TensorId i_id = node.inputs[1];
        if (d_id < 0 || i_id < 0) return false;

        auto ax_it = node.attrs.find("axis");
        if (ax_it == node.attrs.end()) return false;
        const auto* axp = std::get_if<std::int64_t>(&ax_it->second);
        if (axp == nullptr) return false;
        const NSInteger axis = (NSInteger)*axp;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        if (![g respondsToSelector:@selector(scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:)])
            return false;

        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* data = as_tensor(bctx.forward(d_id));
        MPSGraphTensor* idx = as_tensor(bctx.forward(i_id));
        if (grad == nil || data == nil || idx == nil) return false;

        std::vector<std::int64_t> d_shape = shape_of_mps(data);
        if (d_shape.empty()) return false;
        NSArray<NSNumber*>* base_ns = shape_to_ns(d_shape);
        MPSGraphTensor* base = [g constantWithScalar:0.0
                                                shape:base_ns
                                             dataType:data.dataType];

        MPSGraphTensor* d_data =
            [g scatterAlongAxis:axis
                  withDataTensor:base
                   updatesTensor:grad
                   indicesTensor:idx
                            mode:MPSGraphScatterModeAdd
                            name:@"gather_vjp"];
        bctx.accumulate_grad(d_id, from_tensor(d_data));
        return true;
    }
};

struct EmbeddingVjpRegistrar {
    EmbeddingVjpRegistrar() {
        register_vjp_emitter(std::make_unique<EmbeddingVjp>());
        register_vjp_emitter(std::make_unique<GatherVjp>());
    }
};

[[maybe_unused]] static const EmbeddingVjpRegistrar g_embedding_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
