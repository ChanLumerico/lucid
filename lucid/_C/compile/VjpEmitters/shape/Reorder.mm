// lucid/_C/compile/VjpEmitters/shape/Reorder.mm
//
// VJPs for ops that move elements without combining them — ``roll``,
// ``sort`` (and so ``kthvalue``, which traces as a sort), ``scatter_add``
// (and ``scatter``, which traces as one) and ``repeat`` (repeat_interleave).
// None had one, so a training step through them relied on MPSGraph's
// autodiff, which a train-mode batch norm elsewhere rules out.
//
//   roll(x, s)           dx = roll(g, −s)
//   sort(x)              dx[perm[k]] = g[k]   (perm recomputed by argsort;
//                                              a tie may pair differently
//                                              from eager, with equal values)
//   scatter_add(s, i, u) ds = g, du = gather(g, i)
//   repeat(x, r, axis)   dx = g folded to (…, n, r, …) and summed over r
//   detach(x)            nothing

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

bool known(MPSGraphTensor* t) {
    for (NSNumber* d in t.shape)
        if (d.longLongValue < 0)
            return false;
    return true;
}

bool int_attr_of(const OpNode& node, const char* key, std::int64_t& out) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return false;
    const auto* v = std::get_if<std::int64_t>(&it->second);
    if (v == nullptr)
        return false;
    out = *v;
    return true;
}

bool norm_axis(std::int64_t a, NSUInteger rank, NSInteger& out) {
    if (a < 0)
        a += (std::int64_t)rank;
    if (a < 0 || a >= (std::int64_t)rank)
        return false;
    out = (NSInteger)a;
    return true;
}

// Rotate ``t`` right by ``s`` along ``axis`` (a negative ``s`` rotates left).
MPSGraphTensor* rotate(MPSGraph* g, MPSGraphTensor* t, NSInteger axis, std::int64_t s) {
    const long long n = t.shape[(NSUInteger)axis].longLongValue;
    if (n <= 0)
        return t;
    long long k = s % n;
    if (k < 0)
        k += n;
    if (k == 0)
        return t;
    MPSGraphTensor* tail = [g sliceTensor:t dimension:axis start:n - k length:k name:nil];
    MPSGraphTensor* head = [g sliceTensor:t dimension:axis start:0 length:n - k name:nil];
    return [g concatTensors:@[ tail, head ] dimension:axis name:nil];
}

class RollVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "roll"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        auto vec = [&](const char* key) -> const std::vector<std::int64_t>* {
            auto it = node.attrs.find(key);
            return it == node.attrs.end() ? nullptr
                                          : std::get_if<std::vector<std::int64_t>>(&it->second);
        };
        const auto* axes = vec("axes");
        const auto* shifts = vec("shifts");
        // A roll with no axes rolls the flattened tensor; not lowered here.
        if (axes == nullptr || shifts == nullptr || axes->empty() || axes->size() != shifts->size())
            return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor*, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (!known(go))
                    return nil;
                MPSGraphTensor* dx = go;
                for (std::size_t i = 0; i < axes->size(); ++i) {
                    NSInteger ax = 0;
                    if (!norm_axis((*axes)[i], go.shape.count, ax))
                        return nil;
                    dx = rotate(g, dx, ax, -(*shifts)[i]);
                }
                return dx;
            });
    }
};

class SortVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sort"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        std::int64_t a = 0;
        if (!int_attr_of(node, "axis", a))
            return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                NSInteger axis = 0;
                if (!known(x) || !norm_axis(a, x.shape.count, axis))
                    return nil;
                MPSGraphTensor* perm = [g argSortWithTensor:x axis:axis name:nil];
                perm = [g castTensor:perm toType:MPSDataTypeInt32 name:nil];
                MPSGraphTensor* zeros = [g constantWithScalar:0.0 shape:x.shape dataType:go.dataType];
                return [g scatterAlongAxis:axis
                            withDataTensor:zeros
                             updatesTensor:go
                             indicesTensor:perm
                                      mode:MPSGraphScatterModeAdd
                                      name:@"sort_vjp"];
            });
    }
};

class ScatterAddVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "scatter_add"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        std::int64_t d = 0;
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr ||
            !int_attr_of(node, "dim", d) || node.inputs[1] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        MPSGraphTensor* idx = as_tensor(bctx.forward(node.inputs[1]));
        NSInteger axis = 0;
        if (g == nil || go == nil || idx == nil || !norm_axis(d, go.shape.count, axis))
            return false;
        if (node.inputs[0] >= 0)
            bctx.accumulate_grad(node.inputs[0], from_tensor(go));
        if (node.inputs[2] >= 0) {
            if (idx.dataType != MPSDataTypeInt32)
                idx = [g castTensor:idx toType:MPSDataTypeInt32 name:nil];
            MPSGraphTensor* du = [g gatherAlongAxis:axis
                                  withUpdatesTensor:go
                                      indicesTensor:idx
                                               name:@"scatter_add_vjp"];
            bctx.accumulate_grad(node.inputs[2], from_tensor(du));
        }
        return true;
    }
};

// repeat_interleave with one repeat count: element i of ``axis`` becomes
// positions i·r … i·r + r − 1.
class RepeatVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "repeat"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        std::int64_t a = 0, r = 0;
        if (!int_attr_of(node, "axis", a) || !int_attr_of(node, "repeats", r) || r <= 0)
            return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                NSInteger axis = 0;
                if (!known(x) || !norm_axis(a, x.shape.count, axis))
                    return nil;
                NSMutableArray<NSNumber*>* split = [NSMutableArray array];
                for (NSUInteger i = 0; i < x.shape.count; ++i) {
                    [split addObject:x.shape[i]];
                    if ((NSInteger)i == axis)
                        [split addObject:@(r)];
                }
                MPSGraphTensor* folded = [g reshapeTensor:go withShape:split name:nil];
                MPSGraphTensor* summed = [g reductionSumWithTensor:folded axis:axis + 1 name:nil];
                return [g reshapeTensor:summed withShape:x.shape name:@"repeat_vjp"];
            });
    }
};

// detach: the value, with no gradient back to its input.  Registered so
// the walker reaches it and stops there; MPSGraph's autodiff, which would
// differentiate straight through the identity, is not on its safe list.
class DetachVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "detach"; }
    bool emit(BackwardContext&, const OpNode&, const std::vector<void*>&) override {
        return true;
    }
};

// meshgrid: output k is input k broadcast along every axis but its own, so
// its gradient is output k's gradient summed over those axes.
class MeshgridVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "meshgrid"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        const std::size_t n = node.inputs.size();
        if (n == 0 || node.outputs.size() != n)
            return false;
        std::int64_t xy_flag = 0;
        int_attr_of(node, "indexing_xy", xy_flag);
        const bool xy = xy_flag != 0;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        if (g == nil)
            return false;
        for (std::size_t k = 0; k < n && k < grad_outs.size(); ++k) {
            if (grad_outs[k] == nullptr || node.inputs[k] < 0)
                continue;
            MPSGraphTensor* go = as_tensor(grad_outs[k]);
            const std::size_t axis = (xy && n >= 2 && k < 2) ? 1 - k : k;
            NSMutableArray<NSNumber*>* others = [NSMutableArray array];
            for (std::size_t d = 0; d < n; ++d)
                if (d != axis)
                    [others addObject:@(d)];
            MPSGraphTensor* dx = others.count ? [g reductionSumWithTensor:go axes:others name:nil] : go;
            dx = [g reshapeTensor:dx withShape:@[ go.shape[axis] ] name:nil];
            bctx.accumulate_grad(node.inputs[k], from_tensor(dx));
        }
        return true;
    }
};

struct ReorderVjpRegistrar {
    ReorderVjpRegistrar() {
        register_vjp_emitter(std::make_unique<RollVjp>());
        register_vjp_emitter(std::make_unique<SortVjp>());
        register_vjp_emitter(std::make_unique<ScatterAddVjp>());
        register_vjp_emitter(std::make_unique<RepeatVjp>());
        register_vjp_emitter(std::make_unique<DetachVjp>());
        register_vjp_emitter(std::make_unique<MeshgridVjp>());
    }
};

[[maybe_unused]] static const ReorderVjpRegistrar g_reorder_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
