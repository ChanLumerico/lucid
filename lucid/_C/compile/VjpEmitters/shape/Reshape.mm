// lucid/_C/compile/VjpEmitters/shape/Reshape.mm
//
// VJPs for shape-only ops: view / reshape / squeeze / unsqueeze /
// flatten / contiguous / permute / transpose / broadcast_to.
//
// All trivial: the gradient just gets reshaped back to the original
// input shape (or inverse-permuted for permute/transpose).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// ────────────────────────────────────────────────────────────────────
// Shared "reshape grad back to input shape" emitter — handles view /
// reshape / squeeze / unsqueeze / flatten / contiguous.  Body just
// reshapes the incoming grad back to the input's shape.
// ────────────────────────────────────────────────────────────────────
class ReshapeFamilyVjp final : public VjpEmitter {
public:
    explicit ReshapeFamilyVjp(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* grad) -> MPSGraphTensor* {
                std::vector<std::int64_t> in_shape = shape_of_mps(x);
                if (in_shape.empty()) return nil;
                return [g reshapeTensor:grad
                              withShape:shape_to_ns(in_shape)
                                   name:@"reshape_vjp"];
            });
    }

private:
    std::string name_;
};

// ────────────────────────────────────────────────────────────────────
// permute / transpose: dA = inverse_permute(grad).
// ────────────────────────────────────────────────────────────────────
class PermuteVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "permute"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        auto it = node.attrs.find("permutation");
        if (it == node.attrs.end()) return false;
        const auto* perm = std::get_if<std::vector<std::int64_t>>(&it->second);
        if (perm == nullptr) return false;

        // Inverse permutation: inv[perm[i]] = i.
        std::vector<std::int64_t> inv(perm->size());
        for (std::size_t i = 0; i < perm->size(); ++i)
            inv[(*perm)[i]] = (std::int64_t)i;
        NSMutableArray<NSNumber*>* ns_inv =
            [NSMutableArray arrayWithCapacity:inv.size()];
        for (std::int64_t p : inv)
            [ns_inv addObject:[NSNumber numberWithLongLong:p]];

        return emit_unary_vjp(bctx, node, grad_outs,
            [ns_inv](MPSGraph* g, MPSGraphTensor* /*x*/, MPSGraphTensor* grad) {
                return [g transposeTensor:grad permutation:ns_inv name:@"permute_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// broadcast_to: dA = unreduce(grad, input_shape).  The forward op
// records the target shape in outputs[0]; backward needs the original
// input shape, which is the live MPSGraph shape of the input tensor
// (the broadcastTensor:toShape: op did not modify that).
//
// Unreduce isn't expressed cleanly through emit_unary_vjp's "return
// dx" body because it needs the bctx-side helper.  Keeps explicit
// shape.
// ────────────────────────────────────────────────────────────────────
class BroadcastToVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "broadcast_to"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (grad == nil || x == nil) return false;
        std::vector<std::int64_t> in_shape = shape_of_mps(x);
        std::vector<std::int64_t> out_shape =
            node.outputs.empty() ? shape_of_mps(grad) : node.outputs[0].shape;
        if (in_shape.empty() || out_shape.empty()) return false;
        void* dx = bctx.unreduce(from_tensor(grad), in_shape, out_shape);
        bctx.accumulate_grad(x_id, dx);
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// flip: the gradient flips back along the same axes.  tril / triu: the
// gradient keeps exactly the elements the forward kept, so it is the same
// mask applied to the incoming gradient (built from coordinates so any k
// works, where the forward uses bandPart at k = 0).
// ────────────────────────────────────────────────────────────────────
class FlipVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "flip"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || node.inputs[0] < 0 || grad_outs.empty() ||
            grad_outs[0] == nullptr)
            return false;
        auto it = node.attrs.find("dims");
        if (it == node.attrs.end())
            return false;
        const auto* dims = std::get_if<std::vector<std::int64_t>>(&it->second);
        if (dims == nullptr)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        NSMutableArray<NSNumber*>* axes = [NSMutableArray arrayWithCapacity:dims->size()];
        for (std::int64_t d : *dims)
            [axes addObject:@(d)];
        bctx.accumulate_grad(node.inputs[0],
                             from_tensor([g reverseTensor:go axes:axes name:@"flip_vjp"]));
        return true;
    }
};

class TriVjp final : public VjpEmitter {
public:
    explicit TriVjp(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || node.inputs[0] < 0 || grad_outs.empty() ||
            grad_outs[0] == nullptr)
            return false;
        std::int64_t k = 0;
        bool upper = name_ == "triu";
        if (auto it = node.attrs.find("k"); it != node.attrs.end())
            if (const auto* p = std::get_if<std::int64_t>(&it->second))
                k = *p;
        if (auto it = node.attrs.find("upper"); it != node.attrs.end())
            if (const auto* p = std::get_if<bool>(&it->second))
                upper = *p;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        NSArray<NSNumber*>* sh = go.shape;
        if (sh.count < 2 || sh[sh.count - 2].longLongValue < 0 || sh[sh.count - 1].longLongValue < 0)
            return false;
        MPSShape* plane = @[ sh[sh.count - 2], sh[sh.count - 1] ];
        MPSGraphTensor* i = [g coordinateAlongAxis:0 withShape:plane name:nil];
        MPSGraphTensor* j = [g coordinateAlongAxis:1 withShape:plane name:nil];
        MPSGraphTensor* off = [g subtractionWithPrimaryTensor:j secondaryTensor:i name:nil];
        MPSGraphTensor* kk = [g constantWithScalar:(double)k dataType:off.dataType];
        MPSGraphTensor* keep =
            upper ? [g greaterThanOrEqualToWithPrimaryTensor:off secondaryTensor:kk name:nil]
                  : [g lessThanOrEqualToWithPrimaryTensor:off secondaryTensor:kk name:nil];
        MPSGraphTensor* d = [g selectWithPredicateTensor:keep
                                     truePredicateTensor:go
                                    falsePredicateTensor:[g constantWithScalar:0.0
                                                                      dataType:go.dataType]
                                                    name:@"tri_vjp"];
        bctx.accumulate_grad(node.inputs[0], from_tensor(d));
        return true;
    }

private:
    std::string name_;
};

struct ReshapeVjpRegistrar {
    ReshapeVjpRegistrar() {
        register_vjp_emitter(std::make_unique<ReshapeFamilyVjp>("view"));
        register_vjp_emitter(std::make_unique<FlipVjp>());
        register_vjp_emitter(std::make_unique<TriVjp>("tril"));
        register_vjp_emitter(std::make_unique<TriVjp>("triu"));
        register_vjp_emitter(std::make_unique<ReshapeFamilyVjp>("reshape"));
        register_vjp_emitter(std::make_unique<ReshapeFamilyVjp>("squeeze"));
        register_vjp_emitter(std::make_unique<ReshapeFamilyVjp>("unsqueeze"));
        register_vjp_emitter(std::make_unique<ReshapeFamilyVjp>("flatten"));
        register_vjp_emitter(std::make_unique<ReshapeFamilyVjp>("contiguous"));
        register_vjp_emitter(std::make_unique<PermuteVjp>());
        register_vjp_emitter(std::make_unique<BroadcastToVjp>());
    }
};

[[maybe_unused]] static const ReshapeVjpRegistrar g_reshape_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
