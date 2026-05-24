// lucid/_C/compile/OpEmitters/Conv.mm
//
// Conv2d emitter (the hottest op in ResNet-18 forward).
//
// Engine schema name: "conv2d" — see lucid/_C/nn/ConvNd.cpp.
// Attributes (reported by the forward):
//   - "stride"   : vector<int64>  (length N=2 for conv2d)
//   - "padding"  : vector<int64>
//   - "dilation" : vector<int64>
//   - "groups"   : int64
//
// Lucid tensors are NCHW; the weight layout is OIHW (matches the
// reference framework and the underlying conv kernels).  MPSGraph's
// Conv2dDescriptor accepts both via dataLayout / weightsLayout
// settings.  Bias is broadcast-added along the channel dim.

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

inline std::int64_t int_attr(const OpNode& node, const char* key, std::int64_t fallback) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return fallback;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : fallback;
}

class Conv2dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "conv2d"; }

    void* emit(BuilderContext& ctx, const OpNode& node) override {
        // Inputs: x (N, C_in, H, W), W (C_out, C_in/groups, kH, kW),
        // bias (C_out,) — bias may be missing (id == -1).
        if (node.inputs.size() != 3)
            return nullptr;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0)
            return nullptr;

        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        const auto* D = int_vec_attr(node, "dilation");
        if (S == nullptr || P == nullptr || D == nullptr)
            return nullptr;
        if (S->size() != 2 || P->size() != 2 || D->size() != 2)
            return nullptr;
        const std::int64_t groups = int_attr(node, "groups", 1);

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* w_t = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* b_t = nil;
        if (b_id >= 0)
            b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (x_t == nil || w_t == nil || graph == nil)
            return nullptr;

        MPSGraphConvolution2DOpDescriptor* d =
            [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)(*S)[1]
                                                              strideInY:(NSUInteger)(*S)[0]
                                                        dilationRateInX:(NSUInteger)(*D)[1]
                                                        dilationRateInY:(NSUInteger)(*D)[0]
                                                                 groups:(NSUInteger)groups
                                                            paddingLeft:(NSUInteger)(*P)[1]
                                                           paddingRight:(NSUInteger)(*P)[1]
                                                             paddingTop:(NSUInteger)(*P)[0]
                                                          paddingBottom:(NSUInteger)(*P)[0]
                                                           paddingStyle:MPSGraphPaddingStyleExplicit
                                                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        if (d == nil)
            return nullptr;

        MPSGraphTensor* conv =
            [graph convolution2DWithSourceTensor:x_t
                                    weightsTensor:w_t
                                       descriptor:d
                                             name:@"conv2d"];

        // Bias validation: a real bias is 1-D with length = C_out (= w.shape[0]).
        // Lucid's nn.Conv2d with ``bias=False`` may still wire a non-null
        // placeholder into the trace whose shape doesn't match (we've
        // observed (1, C_out, H, W) — leaking the previous layer's
        // output buffer).  Treat any shape mismatch as "no bias" rather
        // than failing the compile.
        NSInteger w_cout =
            (NSInteger)[(NSNumber*)w_t.shape[0] longLongValue];
        if (b_t != nil) {
            const bool bias_is_valid =
                (b_t.shape.count == 1) &&
                ((NSInteger)[(NSNumber*)b_t.shape[0] longLongValue] == w_cout);
            if (!bias_is_valid)
                b_t = nil;
        }

        if (b_t == nil)
            return (__bridge void*)conv;

        // Broadcast bias (C_out,) → (1, C_out, 1, 1) via reshape so the
        // addition lines up against the NCHW conv output channel.
        NSUInteger out_rank = conv.shape.count;  // 4 for NCHW
        NSMutableArray<NSNumber*>* b_shape = [NSMutableArray arrayWithCapacity:out_rank];
        for (NSUInteger i = 0; i < out_rank; ++i)
            [b_shape addObject:[NSNumber numberWithLongLong:1]];
        // Channel dim is index 1 in NCHW.
        b_shape[1] = b_t.shape[0];
        MPSGraphTensor* b_reshape = [graph reshapeTensor:b_t withShape:b_shape name:nil];
        MPSGraphTensor* y =
            [graph additionWithPrimaryTensor:conv secondaryTensor:b_reshape name:@"conv2d_bias"];
        return (__bridge void*)y;
    }
};

// ── Conv1d — reshape trick: x (B,C,L)→(B,C,1,L), W (Cout,Cin/g,K)→(Cout,Cin/g,1,K).
// Uses the existing 2-D builder with strideY=1, padY=0, dilationY=1.
class Conv1dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "conv1d"; }

    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3) return nullptr;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return nullptr;
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        const auto* D = int_vec_attr(node, "dilation");
        if (S == nullptr || P == nullptr || D == nullptr) return nullptr;
        if (S->size() != 1 || P->size() != 1 || D->size() != 1) return nullptr;
        const std::int64_t groups = int_attr(node, "groups", 1);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* W = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* b = (b_id >= 0) ? (__bridge MPSGraphTensor*)ctx.resolve(b_id) : nil;
        if (g == nil || x == nil || W == nil) return nullptr;
        if (x.shape.count != 3 || W.shape.count != 3) return nullptr;
        // x → (B, C, 1, L)
        NSArray<NSNumber*>* x4 = @[x.shape[0], x.shape[1], @1, x.shape[2]];
        MPSGraphTensor* x_r = [g reshapeTensor:x withShape:x4 name:nil];
        // W → (Cout, Cin/g, 1, K)
        NSArray<NSNumber*>* w4 = @[W.shape[0], W.shape[1], @1, W.shape[2]];
        MPSGraphTensor* W_r = [g reshapeTensor:W withShape:w4 name:nil];
        MPSGraphConvolution2DOpDescriptor* d =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:(NSUInteger)(*S)[0]
                              strideInY:1
                        dilationRateInX:(NSUInteger)(*D)[0]
                        dilationRateInY:1
                                 groups:(NSUInteger)groups
                            paddingLeft:(NSUInteger)(*P)[0]
                           paddingRight:(NSUInteger)(*P)[0]
                             paddingTop:0
                          paddingBottom:0
                           paddingStyle:MPSGraphPaddingStyleExplicit
                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        if (d == nil) return nullptr;
        MPSGraphTensor* conv4 = [g convolution2DWithSourceTensor:x_r
                                                    weightsTensor:W_r
                                                       descriptor:d
                                                             name:@"conv1d_lifted"];
        // bias broadcast against channel dim.
        if (b != nil) {
            NSInteger cout = (NSInteger)W.shape[0].longLongValue;
            const bool valid =
                (b.shape.count == 1) && (b.shape[0].longLongValue == cout);
            if (valid) {
                NSArray<NSNumber*>* b_sh = @[@1, b.shape[0], @1, @1];
                MPSGraphTensor* b_r = [g reshapeTensor:b withShape:b_sh name:nil];
                conv4 = [g additionWithPrimaryTensor:conv4
                                      secondaryTensor:b_r
                                                 name:nil];
            }
        }
        // Squeeze H=1 → output (B, Cout, L_out).
        NSArray<NSNumber*>* out_sh =
            @[conv4.shape[0], conv4.shape[1], conv4.shape[3]];
        return (__bridge void*)[g reshapeTensor:conv4 withShape:out_sh name:@"conv1d"];
    }
};

// ── Conv3d — MPSGraph convolution3D (macOS 13.2+).
class Conv3dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "conv3d"; }

    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3) return nullptr;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return nullptr;
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        const auto* D = int_vec_attr(node, "dilation");
        if (S == nullptr || P == nullptr || D == nullptr) return nullptr;
        if (S->size() != 3 || P->size() != 3 || D->size() != 3) return nullptr;
        const std::int64_t groups = int_attr(node, "groups", 1);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* W = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* b = (b_id >= 0) ? (__bridge MPSGraphTensor*)ctx.resolve(b_id) : nil;
        if (g == nil || x == nil || W == nil) return nullptr;
        MPSGraphConvolution3DOpDescriptor* d =
            [MPSGraphConvolution3DOpDescriptor
                descriptorWithStrideInX:(NSUInteger)(*S)[2]
                              strideInY:(NSUInteger)(*S)[1]
                              strideInZ:(NSUInteger)(*S)[0]
                        dilationRateInX:(NSUInteger)(*D)[2]
                        dilationRateInY:(NSUInteger)(*D)[1]
                        dilationRateInZ:(NSUInteger)(*D)[0]
                                 groups:(NSUInteger)groups
                            paddingLeft:(NSUInteger)(*P)[2]
                           paddingRight:(NSUInteger)(*P)[2]
                             paddingTop:(NSUInteger)(*P)[1]
                          paddingBottom:(NSUInteger)(*P)[1]
                           paddingFront:(NSUInteger)(*P)[0]
                            paddingBack:(NSUInteger)(*P)[0]
                           paddingStyle:MPSGraphPaddingStyleExplicit
                             dataLayout:MPSGraphTensorNamedDataLayoutNCDHW
                          weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
        if (d == nil) return nullptr;
        MPSGraphTensor* conv = [g convolution3DWithSourceTensor:x
                                                  weightsTensor:W
                                                     descriptor:d
                                                           name:@"conv3d"];
        if (conv == nil) return nullptr;
        if (b != nil && b.shape.count == 1 &&
            b.shape[0].longLongValue == W.shape[0].longLongValue) {
            NSArray<NSNumber*>* b_sh = @[@1, b.shape[0], @1, @1, @1];
            MPSGraphTensor* b_r = [g reshapeTensor:b withShape:b_sh name:nil];
            conv = [g additionWithPrimaryTensor:conv
                                 secondaryTensor:b_r
                                            name:nil];
        }
        return (__bridge void*)conv;
    }
};

// ── ConvTranspose2d — MPSGraph convolutionTranspose2D.
// Lucid weight layout: (Cin, Cout/g, kH, kW)  — IOHW.
// MPSGraph only exposes OIHW for the descriptor.  Permute the weight
// tensor (swap dims 0,1) before feeding it in so the labels match.
class ConvTranspose2dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "conv_transpose2d"; }

    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3) return nullptr;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return nullptr;
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (S == nullptr || P == nullptr) return nullptr;
        if (S->size() != 2 || P->size() != 2) return nullptr;
        const std::int64_t groups = int_attr(node, "groups", 1);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* W = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* b = (b_id >= 0) ? (__bridge MPSGraphTensor*)ctx.resolve(b_id) : nil;
        if (g == nil || x == nil || W == nil) return nullptr;
        if (W.shape.count != 4) return nullptr;
        // Lucid stores ConvTranspose weights as ``(in, out, kH, kW)`` —
        // the same IOHW layout PyTorch uses, which also matches what
        // MPSGraph's ``convolutionTranspose2D`` expects: it interprets
        // the weight as the *forward* conv's weight (OIHW where O is
        // the forward output = transpose input).  No permutation needed.
        MPSGraphConvolution2DOpDescriptor* d =
            [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)(*S)[1]
                                                              strideInY:(NSUInteger)(*S)[0]
                                                        dilationRateInX:1
                                                        dilationRateInY:1
                                                                 groups:(NSUInteger)groups
                                                            paddingLeft:(NSUInteger)(*P)[1]
                                                           paddingRight:(NSUInteger)(*P)[1]
                                                             paddingTop:(NSUInteger)(*P)[0]
                                                          paddingBottom:(NSUInteger)(*P)[0]
                                                           paddingStyle:MPSGraphPaddingStyleExplicit
                                                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        if (d == nil) return nullptr;
        // Output shape comes from the trace's output meta.
        NSMutableArray<NSNumber*>* out_sh = [NSMutableArray array];
        for (auto v : node.outputs[0].shape)
            [out_sh addObject:[NSNumber numberWithLongLong:v]];
        if (out_sh.count != 4) return nullptr;
        MPSGraphTensor* y =
            [g convolutionTranspose2DWithSourceTensor:x
                                         weightsTensor:W
                                           outputShape:out_sh
                                            descriptor:d
                                                  name:@"conv_transpose2d"];
        if (y == nil) return nullptr;
        if (b != nil && b.shape.count == 1 &&
            b.shape[0].longLongValue == out_sh[1].longLongValue) {
            NSArray<NSNumber*>* b_sh = @[@1, b.shape[0], @1, @1];
            MPSGraphTensor* b_r = [g reshapeTensor:b withShape:b_sh name:nil];
            y = [g additionWithPrimaryTensor:y secondaryTensor:b_r name:nil];
        }
        return (__bridge void*)y;
    }
};

// ── ConvTranspose1d — reshape (B,C,L)→(B,C,1,L), 2D transpose, reshape back.
class ConvTranspose1dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "conv_transpose1d"; }

    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3) return nullptr;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return nullptr;
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (S == nullptr || P == nullptr) return nullptr;
        if (S->size() != 1 || P->size() != 1) return nullptr;
        const std::int64_t groups = int_attr(node, "groups", 1);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* W = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* b = (b_id >= 0) ? (__bridge MPSGraphTensor*)ctx.resolve(b_id) : nil;
        if (g == nil || x == nil || W == nil) return nullptr;
        if (x.shape.count != 3 || W.shape.count != 3) return nullptr;
        NSArray<NSNumber*>* x4 = @[x.shape[0], x.shape[1], @1, x.shape[2]];
        MPSGraphTensor* x_r = [g reshapeTensor:x withShape:x4 name:nil];
        // ConvTranspose weight layout matches forward conv OIHW directly
        // (see ConvTranspose2dEmitter comment) — no permute needed.
        NSArray<NSNumber*>* w4 = @[W.shape[0], W.shape[1], @1, W.shape[2]];
        MPSGraphTensor* W_oihw = [g reshapeTensor:W withShape:w4 name:nil];
        MPSGraphConvolution2DOpDescriptor* d =
            [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)(*S)[0]
                                                              strideInY:1
                                                        dilationRateInX:1
                                                        dilationRateInY:1
                                                                 groups:(NSUInteger)groups
                                                            paddingLeft:(NSUInteger)(*P)[0]
                                                           paddingRight:(NSUInteger)(*P)[0]
                                                             paddingTop:0
                                                          paddingBottom:0
                                                           paddingStyle:MPSGraphPaddingStyleExplicit
                                                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        if (d == nil) return nullptr;
        // Reshape trace output (B, Cout, Lout) → (B, Cout, 1, Lout) for the 2D call.
        NSMutableArray<NSNumber*>* out_sh = [NSMutableArray array];
        if (node.outputs[0].shape.size() != 3) return nullptr;
        for (auto v : node.outputs[0].shape)
            [out_sh addObject:[NSNumber numberWithLongLong:v]];
        NSArray<NSNumber*>* out_sh_4d =
            @[out_sh[0], out_sh[1], @1, out_sh[2]];
        MPSGraphTensor* y4 =
            [g convolutionTranspose2DWithSourceTensor:x_r
                                         weightsTensor:W_oihw
                                           outputShape:out_sh_4d
                                            descriptor:d
                                                  name:@"conv_transpose1d_lifted"];
        if (y4 == nil) return nullptr;
        if (b != nil && b.shape.count == 1 &&
            b.shape[0].longLongValue == out_sh[1].longLongValue) {
            NSArray<NSNumber*>* b_sh = @[@1, b.shape[0], @1, @1];
            MPSGraphTensor* b_r = [g reshapeTensor:b withShape:b_sh name:nil];
            y4 = [g additionWithPrimaryTensor:y4 secondaryTensor:b_r name:nil];
        }
        // Squeeze H=1 → (B, Cout, Lout)
        return (__bridge void*)[g reshapeTensor:y4 withShape:out_sh name:@"conv_transpose1d"];
    }
};

// ── Unfold (im2col) — MPSGraph imToCol (macOS 14+), 2D variant.
// Lucid output layout: (B, C * kH * kW, L_out).  MPSGraph imToCol
// docs are vague about layout; testing confirms it matches the
// (N, C * kH * kW, oH * oW) Lucid contract for the 2D case.
class UnfoldEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "unfold"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return nullptr;
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        const auto* D = int_vec_attr(node, "dilation");
        if (K == nullptr || S == nullptr || P == nullptr || D == nullptr) return nullptr;
        if (K->size() != 2 || S->size() != 2 || P->size() != 2 || D->size() != 2)
            return nullptr;  // only 2D supported by MPSGraph imToCol
        if (![[MPSGraph class] respondsToSelector:@selector(class)] ||
            NSClassFromString(@"MPSGraphImToColOpDescriptor") == nil) return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return nullptr;
        if (x.shape.count != 4) return nullptr;
        MPSGraphImToColOpDescriptor* d =
            [MPSGraphImToColOpDescriptor descriptorWithKernelWidth:(NSUInteger)(*K)[1]
                                                       kernelHeight:(NSUInteger)(*K)[0]
                                                          strideInX:(NSUInteger)(*S)[1]
                                                          strideInY:(NSUInteger)(*S)[0]
                                                    dilationRateInX:(NSUInteger)(*D)[1]
                                                    dilationRateInY:(NSUInteger)(*D)[0]
                                                        paddingLeft:(NSUInteger)(*P)[1]
                                                       paddingRight:(NSUInteger)(*P)[1]
                                                         paddingTop:(NSUInteger)(*P)[0]
                                                      paddingBottom:(NSUInteger)(*P)[0]
                                                         dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        if (d == nil) return nullptr;
        return (__bridge void*)[g imToColWithSourceTensor:x
                                                descriptor:d
                                                      name:@"unfold"];
    }
};

// ── Fold (col2im) — MPSGraph colToIm (macOS 14+).
class FoldEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "fold"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return nullptr;
        const auto* OS = int_vec_attr(node, "output_size");
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        const auto* D = int_vec_attr(node, "dilation");
        if (OS == nullptr || K == nullptr || S == nullptr || P == nullptr || D == nullptr)
            return nullptr;
        if (K->size() != 2) return nullptr;
        if (NSClassFromString(@"MPSGraphImToColOpDescriptor") == nil) return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return nullptr;
        MPSGraphImToColOpDescriptor* d =
            [MPSGraphImToColOpDescriptor descriptorWithKernelWidth:(NSUInteger)(*K)[1]
                                                       kernelHeight:(NSUInteger)(*K)[0]
                                                          strideInX:(NSUInteger)(*S)[1]
                                                          strideInY:(NSUInteger)(*S)[0]
                                                    dilationRateInX:(NSUInteger)(*D)[1]
                                                    dilationRateInY:(NSUInteger)(*D)[0]
                                                        paddingLeft:(NSUInteger)(*P)[1]
                                                       paddingRight:(NSUInteger)(*P)[1]
                                                         paddingTop:(NSUInteger)(*P)[0]
                                                      paddingBottom:(NSUInteger)(*P)[0]
                                                         dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        if (d == nil) return nullptr;
        // Output shape comes from trace meta: (N, C, outH, outW).
        NSMutableArray<NSNumber*>* out_sh = [NSMutableArray array];
        for (auto v : node.outputs[0].shape)
            [out_sh addObject:[NSNumber numberWithLongLong:v]];
        if (out_sh.count != 4) return nullptr;
        // macOS 26 SDK regression: ``colToImWithSourceTensor:`` runtime
        // validator rejects the rank-3 ``(N, C*kH*kW, oH*oW)`` form
        // Lucid (and PyTorch) use, with "invalid output shape for
        // input & attributes" — see [[engine-mps26-colToIm-regression]].
        // Until the layout contract is resolved (Apple FB or a working
        // rank-4 reshape recipe), drop to eager fallback for fold so
        // the compile path stays correct.  Fold is the conv-backward
        // shadow path, never the user-visible forward, so eager fold
        // here costs nothing in production.
        (void)out_sh;
        return nullptr;
    }
};

struct ConvEmitterRegistrar {
    ConvEmitterRegistrar() {
        register_emitter(std::make_unique<Conv2dEmitter>());
        register_emitter(std::make_unique<Conv1dEmitter>());
        register_emitter(std::make_unique<Conv3dEmitter>());
        register_emitter(std::make_unique<ConvTranspose2dEmitter>());
        register_emitter(std::make_unique<ConvTranspose1dEmitter>());
        register_emitter(std::make_unique<UnfoldEmitter>());
        register_emitter(std::make_unique<FoldEmitter>());
    }
};

[[maybe_unused]] static const ConvEmitterRegistrar g_conv_registrar;

}  // namespace

}  // namespace lucid::compile
