// lucid/_C/compile/VjpEmitters/reduce/Reduction.mm
//
// VJPs for ``sum`` and ``mean`` reductions.
//
// Forward emitter (see :file:`OpEmitters/reduce/Reduction.mm`) carries
// ``dims`` (vector<int64>) + ``keepdim`` (bool) attrs and squeezes the
// output if keepdim=false.  Backward:
//
//   sum:  dA = broadcast(grad, in_shape) — with axis re-insertion when
//         keepdim=false.
//   mean: same as sum, then divided by the number of reduced elements
//         (∏ in_shape[d] over d in dims).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// Re-insert squeezed reduction axes back into the grad tensor.  If
// the forward op had keepdim=false, grad arrives with rank
// ``in_rank - n_reduced``; we need to reshape it so each reduced
// axis is size 1 before broadcasting back to ``in_shape``.
MPSGraphTensor* unsqueeze_reduced(MPSGraph* g, MPSGraphTensor* grad,
                                  const std::vector<std::int64_t>& in_shape,
                                  const std::vector<std::int64_t>& dims) {
    // Build the keepdim-style shape: in_shape but with reduced axes
    // replaced by 1.
    std::vector<bool> is_reduced(in_shape.size(), false);
    for (std::int64_t d : dims) {
        std::int64_t ax = (d < 0) ? d + (std::int64_t)in_shape.size() : d;
        if (ax >= 0 && ax < (std::int64_t)in_shape.size())
            is_reduced[ax] = true;
    }
    NSMutableArray<NSNumber*>* shape =
        [NSMutableArray arrayWithCapacity:in_shape.size()];
    for (std::size_t i = 0; i < in_shape.size(); ++i) {
        std::int64_t s = is_reduced[i] ? 1 : in_shape[i];
        [shape addObject:[NSNumber numberWithLongLong:s]];
    }
    return [g reshapeTensor:grad withShape:shape name:nil];
}

// Common ``dims`` + ``keepdim`` attr extraction for reduction VJPs.
// Returns ``{dims_ptr, keepdim_ptr}`` with one or both null on failure.
struct ReduceAttrs {
    const std::vector<std::int64_t>* dims = nullptr;
    const bool* keepdim = nullptr;
};

inline ReduceAttrs extract_reduce_attrs(const OpNode& node) {
    ReduceAttrs r;
    auto dims_it = node.attrs.find("dims");
    auto kd_it = node.attrs.find("keepdim");
    if (dims_it == node.attrs.end() || kd_it == node.attrs.end()) return r;
    r.dims = std::get_if<std::vector<std::int64_t>>(&dims_it->second);
    r.keepdim = std::get_if<bool>(&kd_it->second);
    return r;
}

class SumVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sum"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        ReduceAttrs a = extract_reduce_attrs(node);
        if (a.dims == nullptr || a.keepdim == nullptr) return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&a](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* grad) -> MPSGraphTensor* {
                std::vector<std::int64_t> in_shape = shape_of_mps(x);
                if (in_shape.empty()) return nil;
                MPSGraphTensor* grad_keep =
                    *a.keepdim ? grad : unsqueeze_reduced(g, grad, in_shape, *a.dims);
                return [g broadcastTensor:grad_keep
                                  toShape:shape_to_ns(in_shape)
                                     name:@"sum_vjp"];
            });
    }
};

class MeanVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "mean"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        ReduceAttrs a = extract_reduce_attrs(node);
        if (a.dims == nullptr || a.keepdim == nullptr) return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&a](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* grad) -> MPSGraphTensor* {
                std::vector<std::int64_t> in_shape = shape_of_mps(x);
                if (in_shape.empty()) return nil;
                // N = product of reduced-axis sizes.
                double N = 1.0;
                for (std::int64_t d : *a.dims) {
                    std::int64_t ax = (d < 0) ? d + (std::int64_t)in_shape.size() : d;
                    if (ax >= 0 && ax < (std::int64_t)in_shape.size())
                        N *= (double)in_shape[ax];
                }
                MPSGraphTensor* grad_keep =
                    *a.keepdim ? grad : unsqueeze_reduced(g, grad, in_shape, *a.dims);
                MPSGraphTensor* inv_n = [g constantWithScalar:(1.0 / N)
                                                     dataType:grad_keep.dataType];
                MPSGraphTensor* scaled =
                    [g multiplicationWithPrimaryTensor:grad_keep
                                       secondaryTensor:inv_n name:nil];
                return [g broadcastTensor:scaled
                                  toShape:shape_to_ns(in_shape)
                                     name:@"mean_vjp"];
            });
    }
};

struct ReductionVjpRegistrar {
    ReductionVjpRegistrar() {
        register_vjp_emitter(std::make_unique<SumVjp>());
        register_vjp_emitter(std::make_unique<MeanVjp>());
    }
};

[[maybe_unused]] static const ReductionVjpRegistrar g_reduction_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
