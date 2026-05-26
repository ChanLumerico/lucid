// lucid/_C/compile/OpEmitters/Pool.mm
//
// MaxPool2d / AvgPool2d emitters.
//
// Op names match the engine schemas in lucid/_C/nn/PoolNd.cpp
// ("max_pool2d", "avg_pool2d").  Each carries vector<int64> attrs
// ``kernel_size`` / ``stride`` / ``padding`` reported by the forward
// via :func:`OpScopeFull::set_attr`.
//
// Lucid's tensors are NCHW; MPSGraph's 2-D pool descriptors default to
// NHWC, so the descriptor's ``dataLayout`` is set to
// ``NCHW`` explicitly.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

inline const std::vector<std::int64_t>* int_vec_attr(const OpNode& node, const char* key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return nullptr;
    return std::get_if<std::vector<std::int64_t>>(&it->second);
}

class MaxPool2dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "max_pool2d"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr)
            return false;
        if (K->size() != 2 || S->size() != 2 || P->size() != 2)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return false;

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

        MPSGraphTensor* y =
            [graph maxPooling2DWithSourceTensor:x_t descriptor:d name:@"max_pool2d"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

class AvgPool2dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "avg_pool2d"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr)
            return false;
        if (K->size() != 2 || S->size() != 2 || P->size() != 2)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return false;

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
        // Lucid AvgPool2d matches the reference framework's default
        // ``count_include_pad=True``; MPSGraph defaults to NO, so flip
        // the divisor flag so padded zeros participate in the mean.
        d.includeZeroPadToAverage = YES;

        MPSGraphTensor* y =
            [graph avgPooling2DWithSourceTensor:x_t descriptor:d name:@"avg_pool2d"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// ── MaxPool1d — reshape (B,C,L)→(B,C,1,L), maxPool2D, reshape back.
template <bool IS_MAX>
class Pool1dEmitterT final : public OpEmitter {
public:
    explicit Pool1dEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr) return false;
        if (K->size() != 1 || S->size() != 1 || P->size() != 1) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        if (x.shape.count != 3) return false;
        NSArray<NSNumber*>* x4 = @[x.shape[0], x.shape[1], @1, x.shape[2]];
        MPSGraphTensor* x_r = [g reshapeTensor:x withShape:x4 name:nil];
        MPSGraphPooling2DOpDescriptor* d =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)(*K)[0]
                                                       kernelHeight:1
                                                          strideInX:(NSUInteger)(*S)[0]
                                                          strideInY:1
                                                     paddingStyle:MPSGraphPaddingStyleExplicit
                                                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        d.paddingLeft = (NSUInteger)(*P)[0];
        d.paddingRight = (NSUInteger)(*P)[0];
        d.paddingTop = 0;
        d.paddingBottom = 0;
        MPSGraphTensor* y4;
        if (IS_MAX) {
            y4 = [g maxPooling2DWithSourceTensor:x_r descriptor:d name:@"max_pool1d_lifted"];
        } else {
            d.includeZeroPadToAverage = YES;
            y4 = [g avgPooling2DWithSourceTensor:x_r descriptor:d name:@"avg_pool1d_lifted"];
        }
        // Squeeze H=1.
        NSArray<NSNumber*>* out_sh = @[y4.shape[0], y4.shape[1], y4.shape[3]];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g reshapeTensor:y4 withShape:out_sh name:nil]));
        return true;
    }

private:
    std::string name_;
};

struct PoolEmitterRegistrar {
    PoolEmitterRegistrar() {
        register_emitter(std::make_unique<MaxPool2dEmitter>());
        register_emitter(std::make_unique<AvgPool2dEmitter>());
        register_emitter(std::make_unique<Pool1dEmitterT<true>>("max_pool1d"));
        register_emitter(std::make_unique<Pool1dEmitterT<false>>("avg_pool1d"));
    }
};

[[maybe_unused]] static const PoolEmitterRegistrar g_pool_registrar;

}  // namespace

}  // namespace lucid::compile
