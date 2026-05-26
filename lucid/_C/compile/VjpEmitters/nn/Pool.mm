// lucid/_C/compile/VjpEmitters/nn/Pool.mm
//
// Pool VJPs — avg_pool2d / max_pool2d / adaptive_avg_pool2d.
//
// Critical for ResNet-class workloads: the manual VJP walker
// previously fell back to MPSGraph autograd for these ops, which
// breaks under autocast (gradientForPrimaryTensor: emits constants
// in F32 regardless of the F16 input chain → MPSGraph type-check
// failure deep in MLIR pass manager).  Manual VJP keeps the chain
// in a single dtype.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../_VjpHelpers.h"
#include "../../OpEmitters/_AttrHelpers.h"

namespace lucid::compile {

namespace {

// Pull a list-of-int attr or return nullptr.
inline const std::vector<std::int64_t>* int_vec_attr_local(
    const OpNode& node, const char* key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end()) return nullptr;
    return std::get_if<std::vector<std::int64_t>>(&it->second);
}

// avg_pool2d backward: MPSGraph provides a dedicated gradient API
// (``avgPooling2DGradientWithGradientTensor:sourceTensor:descriptor:``)
// that handles include-pad averaging correctly.  The "source tensor"
// is the original forward input — we look it up via the saved forward
// activation.
class AvgPool2dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "avg_pool2d"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;

        const auto* K = int_vec_attr_local(node, "kernel_size");
        const auto* S = int_vec_attr_local(node, "stride");
        const auto* P = int_vec_attr_local(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr) return false;
        if (K->size() != 2 || S->size() != 2 || P->size() != 2) return false;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (g == nil || grad == nil || x == nil) return false;

        // Mixed-dtype reconciliation: avg_pool's gradient kernel
        // needs grad + source tensor in matching dtypes (MPSGraph
        // rejects otherwise under autocast).
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);

        MPSGraphPooling2DOpDescriptor* d =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)(*K)[1]
                                                       kernelHeight:(NSUInteger)(*K)[0]
                                                          strideInX:(NSUInteger)(*S)[1]
                                                          strideInY:(NSUInteger)(*S)[0]
                                                     paddingStyle:MPSGraphPaddingStyleExplicit
                                                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        d.paddingLeft = (NSUInteger)(*P)[1];
        d.paddingRight = (NSUInteger)(*P)[1];
        d.paddingTop = (NSUInteger)(*P)[0];
        d.paddingBottom = (NSUInteger)(*P)[0];
        d.includeZeroPadToAverage = YES;  // match forward emit

        MPSGraphTensor* dx =
            [g avgPooling2DGradientWithGradientTensor:grad
                                          sourceTensor:x
                                            descriptor:d
                                                  name:@"avg_pool2d_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

// max_pool2d backward: needs the forward INDICES tensor.  MPSGraph
// provides ``maxPooling2DGradientWithGradientTensor:sourceTensor:descriptor:``
// which recomputes the max-positions from the source tensor (no
// indices feed needed) — same convention as the forward emitter.
class MaxPool2dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "max_pool2d"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;

        const auto* K = int_vec_attr_local(node, "kernel_size");
        const auto* S = int_vec_attr_local(node, "stride");
        const auto* P = int_vec_attr_local(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr) return false;
        if (K->size() != 2 || S->size() != 2 || P->size() != 2) return false;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (g == nil || grad == nil || x == nil) return false;

        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);

        MPSGraphPooling2DOpDescriptor* d =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)(*K)[1]
                                                       kernelHeight:(NSUInteger)(*K)[0]
                                                          strideInX:(NSUInteger)(*S)[1]
                                                          strideInY:(NSUInteger)(*S)[0]
                                                     paddingStyle:MPSGraphPaddingStyleExplicit
                                                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        d.paddingLeft = (NSUInteger)(*P)[1];
        d.paddingRight = (NSUInteger)(*P)[1];
        d.paddingTop = (NSUInteger)(*P)[0];
        d.paddingBottom = (NSUInteger)(*P)[0];

        MPSGraphTensor* dx =
            [g maxPooling2DGradientWithGradientTensor:grad
                                          sourceTensor:x
                                            descriptor:d
                                                  name:@"max_pool2d_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

struct PoolVjpRegistrar {
    PoolVjpRegistrar() {
        register_vjp_emitter(std::make_unique<AvgPool2dVjp>());
        register_vjp_emitter(std::make_unique<MaxPool2dVjp>());
    }
};

[[maybe_unused]] static const PoolVjpRegistrar g_pool_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
