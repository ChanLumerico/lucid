// lucid/_C/compile/VjpEmitters/nn/Conv.mm
//
// VJPs for the conv2d / conv1d family.  Uses MPSGraph's bespoke
// convolution-gradient ops:
//   * ``convolution2DDataGradientWithIncomingGradientTensor:`` → dX
//   * ``convolution2DWeightsGradientWithIncomingGradientTensor:`` → dW
//
// Both APIs require the original input/weight *shapes* as ``MPSShape*``
// (NSArray<NSNumber*>) plus the original 2-D descriptor — we reuse
// the forward emitter's descriptor verbatim.
//
// dX is shaped like the original input ``x``.
// dW is shaped like the original weight ``w``.
// dB is the channel-wise sum of grad across (N, H, W) for conv2d
// (axes 0, 2, 3 in NCHW).  Conv1d uses (0, 2) post-reshape-to-2d.

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

// Local attr helpers (replicate to keep VJP file self-contained — same
// as the forward Conv.mm).
inline const std::vector<std::int64_t>* iv_attr(const OpNode& node, const char* k) {
    auto it = node.attrs.find(k);
    if (it == node.attrs.end()) return nullptr;
    return std::get_if<std::vector<std::int64_t>>(&it->second);
}

inline std::int64_t i_attr(const OpNode& node, const char* k, std::int64_t f) {
    auto it = node.attrs.find(k);
    if (it == node.attrs.end()) return f;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : f;
}

// Build the 2-D conv descriptor from a node's stride / padding /
// dilation / groups attrs.  Returns nil on attr failure.
inline MPSGraphConvolution2DOpDescriptor* make_conv2d_desc(
    std::int64_t sX, std::int64_t sY,
    std::int64_t dX, std::int64_t dY,
    std::int64_t pX, std::int64_t pY,
    std::int64_t groups) {
    return [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:(NSUInteger)sX
                      strideInY:(NSUInteger)sY
                dilationRateInX:(NSUInteger)dX
                dilationRateInY:(NSUInteger)dY
                         groups:(NSUInteger)groups
                    paddingLeft:(NSUInteger)pX
                   paddingRight:(NSUInteger)pX
                     paddingTop:(NSUInteger)pY
                  paddingBottom:(NSUInteger)pY
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
}

// ────────────────────────────────────────────────────────────────────
// conv2d VJP — dX via DataGradient, dW via WeightsGradient, dB via
// channel-wise sum of grad over (N, H, W).
// ────────────────────────────────────────────────────────────────────
class Conv2dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "conv2d"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return false;

        const auto* S = iv_attr(node, "stride");
        const auto* P = iv_attr(node, "padding");
        const auto* D = iv_attr(node, "dilation");
        if (S == nullptr || P == nullptr || D == nullptr) return false;
        if (S->size() != 2 || P->size() != 2 || D->size() != 2) return false;
        const std::int64_t groups = i_attr(node, "groups", 1);

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* w = as_tensor(bctx.forward(w_id));
        if (g == nil || grad == nil || x == nil || w == nil) return false;

        // Mixed-dtype reconciliation (autocast).  MPSGraph's
        // convolution2DDataGradient + WeightsGradient APIs require
        // matching dtypes on grad + w (for data-grad) and grad + x
        // (for weights-grad).  Cast forward x / w to grad's dtype.
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);
        w = cast_if_needed(g, w, chain_dt);

        MPSGraphConvolution2DOpDescriptor* desc =
            make_conv2d_desc((*S)[1], (*S)[0], (*D)[1], (*D)[0], (*P)[1], (*P)[0], groups);
        if (desc == nil) return false;

        // PERF: MPSGraph's depthwise/grouped (groups>1) convolution2D
        // gradient kernels fall onto a ~14×-slower path when their
        // incomingGradient is the DIRECT output of a transpose — exactly
        // ConvNeXt's NHWC-LayerNorm block, where the channels-last permute's
        // VJP feeds a transposed grad straight into the depthwise weights/
        // data gradient (291 ms vs 20 ms / block at C=96, 56×56, BS=32;
        // regular groups==1 convs are unaffected, ~44 ms).  Inserting any
        // NON-FOLDABLE elementwise op between the transpose and the gradient
        // kernel restores the fast path; a ``*1.0`` / ``+0.0`` launder is
        // constant-folded away by MPSGraph, so scale the grad by 2 (survives
        // folding) and unscale the linear gradient outputs by 0.5.
        // groups==1 leaves the grad untouched (zero regression for regular
        // convs / ResNet / VGG).
        MPSGraphTensor* grad_k = grad;
        MPSGraphTensor* unscale = nil;
        if (groups > 1) {
            MPSGraphTensor* two = [g constantWithScalar:2.0 dataType:grad.dataType];
            grad_k = [g multiplicationWithPrimaryTensor:grad
                                        secondaryTensor:two
                                                   name:@"conv2d_vjp_dw_launder"];
            unscale = [g constantWithScalar:0.5 dataType:grad.dataType];
        }

        // dX = convolution2D_dataGradient(grad, w, original_x_shape, desc)
        MPSGraphTensor* dX =
            [g convolution2DDataGradientWithIncomingGradientTensor:grad_k
                                                    weightsTensor:w
                                                      outputShape:x.shape
                                       forwardConvolutionDescriptor:desc
                                                             name:@"conv2d_vjp_dx"];
        if (unscale != nil)
            dX = [g multiplicationWithPrimaryTensor:dX
                                    secondaryTensor:unscale
                                               name:@"conv2d_vjp_dx_unscale"];
        bctx.accumulate_grad(x_id, from_tensor(dX));

        // dW = convolution2D_weightsGradient(grad, x, original_w_shape, desc)
        //
        // PERF: MPSGraph's GROUPED convolution2DWeightsGradient(groups=C) is
        // catastrophically slow for a DEPTHWISE conv — ~9x the reference at 7x7,
        // and it degrades with kernel size (3x3 ~1.9x, 7x7 ~9x: 17.3ms vs 1.9ms
        // at C=96 56x56 BS=32).  Apple ships a DEDICATED, purpose-built kernel —
        // depthwiseConvolution2DWeightsGradient (MPSGraphDepthwiseConvolution2D
        // OpDescriptor) — that is a DISTINCT path from the grouped one.  Route the
        // true depthwise case (groups>1, weight (C,1,kH,kW), stride==1, dilation==1)
        // through it; everything else (strided / dilated / genuinely grouped
        // C_in/groups>1) falls back to the grouped kernel unchanged.  forward + dX
        // stay on their existing paths (only dW was pathological).
        std::vector<std::int64_t> w_shape_dw = shape_of_mps(w);
        const bool is_depthwise =
            (groups > 1) && (w_shape_dw.size() == 4) &&
            (w_shape_dw[0] == groups) && (w_shape_dw[1] == 1) &&
            ((*S)[0] == 1) && ((*S)[1] == 1) &&
            ((*D)[0] == 1) && ((*D)[1] == 1);
        MPSGraphTensor* dW = nil;
        if (is_depthwise) {
            // Apple's dedicated depthwise weight-grad kernel uses a leading-pad-0
            // tap alignment and IGNORES the descriptor padding (verified by a
            // padding sweep that left the output unchanged) — so a "same" conv's
            // dW comes out spatially shifted by (+pad,+pad).  Restore the symmetric
            // alignment WITHOUT relying on the descriptor: PRE-PAD x by `pad` on all
            // four sides, then the kernel's only consistent reading of (x_pad size
            // H+2p, grad size H, K=2p+1 taps) is the VALID correlation
            //   dW[kh,kw] = sum grad[oh,ow] * x_pad[oh+kh, ow+kw]
            //             = sum grad * x[oh+kh-pad, ow+kw-pad]
            // which is exactly the symmetric-conv weight gradient.  p is per-layer
            // so this is kernel-size-general.
            const long pT = (long)(*P)[0], pL = (long)(*P)[1];
            MPSGraphTensor* x_dw = x;
            if (pT > 0 || pL > 0) {
                NSArray<NSNumber*>* lo = @[ @0, @0, @(pT), @(pL) ];
                NSArray<NSNumber*>* hi = @[ @0, @0, @(pT), @(pL) ];
                x_dw = [g padTensor:x
                    withPaddingMode:MPSGraphPaddingModeConstant
                        leftPadding:lo
                       rightPadding:hi
                      constantValue:0.0
                               name:@"conv2d_vjp_dw_prepad"];
            }
            MPSGraphDepthwiseConvolution2DOpDescriptor* dwd =
                [MPSGraphDepthwiseConvolution2DOpDescriptor
                    descriptorWithStrideInX:1
                                  strideInY:1
                            dilationRateInX:1
                            dilationRateInY:1
                                paddingLeft:0
                               paddingRight:0
                                 paddingTop:0
                              paddingBottom:0
                               paddingStyle:MPSGraphPaddingStyleExplicit
                                 dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
            if (dwd != nil) {
                // Depthwise weight layout: 'O' = channel-multiplier (=1), 'I' =
                // channel (=C).  So the op's weight shape is (1, C, kH, kW); that
                // is a FREE reshape of lucid's (C, 1, kH, kW) (only a size-1 axis
                // is swapped, so the row-major bytes are identical).
                std::vector<std::int64_t> dw_out = {
                    1, w_shape_dw[0], w_shape_dw[2], w_shape_dw[3]};
                MPSGraphTensor* dW_dw =
                    [g depthwiseConvolution2DWeightsGradientWithIncomingGradientTensor:grad_k
                                                                         sourceTensor:x_dw
                                                                          outputShape:shape_to_ns(dw_out)
                                                                           descriptor:dwd
                                                                                 name:@"conv2d_vjp_dw_depthwise"];
                if (dW_dw != nil)
                    dW = [g reshapeTensor:dW_dw
                                withShape:w.shape
                                     name:@"conv2d_vjp_dw_reshape"];
            }
        }
        if (dW == nil) {
            dW = [g convolution2DWeightsGradientWithIncomingGradientTensor:grad_k
                                                             sourceTensor:x
                                                              outputShape:w.shape
                                             forwardConvolutionDescriptor:desc
                                                                    name:@"conv2d_vjp_dw"];
        }
        if (unscale != nil)
            dW = [g multiplicationWithPrimaryTensor:dW
                                    secondaryTensor:unscale
                                               name:@"conv2d_vjp_dw_unscale"];
        bctx.accumulate_grad(w_id, from_tensor(dW));

        // dB = sum(grad, axes=(0, 2, 3)) → (C_out,).  Only if bias was real.
        if (b_id >= 0) {
            MPSGraphTensor* b = as_tensor(bctx.forward(b_id));
            // Match the forward emitter's bias validation: only treat
            // as a real bias if it's 1-D with size matching w.shape[0].
            std::vector<std::int64_t> w_shape = shape_of_mps(w);
            std::vector<std::int64_t> b_shape = b != nil ? shape_of_mps(b) : std::vector<std::int64_t>{};
            const bool bias_valid =
                (b != nil) && (b_shape.size() == 1) &&
                (!w_shape.empty()) && (b_shape[0] == w_shape[0]);
            if (bias_valid) {
                NSArray<NSNumber*>* axes_nhw = @[ @0, @2, @3 ];
                MPSGraphTensor* db_keep =
                    [g reductionSumWithTensor:grad axes:axes_nhw name:nil];
                NSArray<NSNumber*>* db_shape =
                    @[ [NSNumber numberWithLongLong:w_shape[0]] ];
                MPSGraphTensor* dB =
                    [g reshapeTensor:db_keep withShape:db_shape name:@"conv2d_vjp_db"];
                bctx.accumulate_grad(b_id, from_tensor(dB));
            }
        }
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// conv1d VJP — applies the same reshape-as-2D trick as the forward.
// Forward reshapes x (B,C,L)→(B,C,1,L), w (Cout,Cin/g,K)→(Cout,Cin/g,1,K),
// emits 2D conv with strideY=1 padY=0 dilationY=1, then reshapes back.
//
// Backward symmetric: the conv2d grad APIs produce dX_2D and dW_2D in
// the reshaped layouts; we just reshape them back to the original
// 1-D rank.
// ────────────────────────────────────────────────────────────────────
class Conv1dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "conv1d"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return false;

        const auto* S = iv_attr(node, "stride");
        const auto* P = iv_attr(node, "padding");
        const auto* D = iv_attr(node, "dilation");
        if (S == nullptr || P == nullptr || D == nullptr) return false;
        if (S->size() != 1 || P->size() != 1 || D->size() != 1) return false;
        const std::int64_t groups = i_attr(node, "groups", 1);

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* w = as_tensor(bctx.forward(w_id));
        if (g == nil || grad == nil || x == nil || w == nil) return false;

        // Mixed-dtype reconciliation (autocast).
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);
        w = cast_if_needed(g, w, chain_dt);

        // Reshape x (B, C, L) → (B, C, 1, L); w (Cout, Cin/g, K) → (Cout, Cin/g, 1, K).
        // grad shape is (B, C_out, L_out) → reshape to (B, C_out, 1, L_out).
        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        std::vector<std::int64_t> w_shape = shape_of_mps(w);
        std::vector<std::int64_t> g_shape = shape_of_mps(grad);
        if (x_shape.size() != 3 || w_shape.size() != 3 || g_shape.size() != 3) return false;

        auto insert_1 = [](std::vector<std::int64_t> v) {
            v.insert(v.begin() + 2, 1);
            return v;
        };
        std::vector<std::int64_t> x_2d_shape = insert_1(x_shape);
        std::vector<std::int64_t> w_2d_shape = insert_1(w_shape);
        std::vector<std::int64_t> g_2d_shape = insert_1(g_shape);
        MPSGraphTensor* x_2d =
            [g reshapeTensor:x withShape:shape_to_ns(x_2d_shape) name:nil];
        MPSGraphTensor* w_2d =
            [g reshapeTensor:w withShape:shape_to_ns(w_2d_shape) name:nil];
        MPSGraphTensor* grad_2d =
            [g reshapeTensor:grad withShape:shape_to_ns(g_2d_shape) name:nil];

        MPSGraphConvolution2DOpDescriptor* desc =
            make_conv2d_desc((*S)[0], 1, (*D)[0], 1, (*P)[0], 0, groups);
        if (desc == nil) return false;

        // dX_2d / dW_2d via the 2D conv-grad APIs.
        MPSGraphTensor* dX_2d =
            [g convolution2DDataGradientWithIncomingGradientTensor:grad_2d
                                                    weightsTensor:w_2d
                                                      outputShape:x_2d.shape
                                       forwardConvolutionDescriptor:desc
                                                             name:nil];
        MPSGraphTensor* dW_2d =
            [g convolution2DWeightsGradientWithIncomingGradientTensor:grad_2d
                                                       sourceTensor:x_2d
                                                        outputShape:w_2d.shape
                                       forwardConvolutionDescriptor:desc
                                                              name:nil];

        // Reshape grads back to 3-D.
        MPSGraphTensor* dX =
            [g reshapeTensor:dX_2d withShape:shape_to_ns(x_shape) name:@"conv1d_vjp_dx"];
        MPSGraphTensor* dW =
            [g reshapeTensor:dW_2d withShape:shape_to_ns(w_shape) name:@"conv1d_vjp_dw"];
        bctx.accumulate_grad(x_id, from_tensor(dX));
        bctx.accumulate_grad(w_id, from_tensor(dW));

        // dB = sum(grad, axes=(0, 2)) over (N, L) → (C_out,).  Use grad_2d
        // for consistency with conv2d (axes 0, 2, 3).
        if (b_id >= 0) {
            MPSGraphTensor* b = as_tensor(bctx.forward(b_id));
            std::vector<std::int64_t> b_shape =
                (b != nil) ? shape_of_mps(b) : std::vector<std::int64_t>{};
            const bool bias_valid =
                (b != nil) && (b_shape.size() == 1) && (b_shape[0] == w_shape[0]);
            if (bias_valid) {
                NSArray<NSNumber*>* axes_nl = @[ @0, @2 ];
                MPSGraphTensor* db_keep =
                    [g reductionSumWithTensor:grad axes:axes_nl name:nil];
                NSArray<NSNumber*>* db_shape =
                    @[ [NSNumber numberWithLongLong:w_shape[0]] ];
                MPSGraphTensor* dB =
                    [g reshapeTensor:db_keep withShape:db_shape name:@"conv1d_vjp_db"];
                bctx.accumulate_grad(b_id, from_tensor(dB));
            }
        }
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// Conv3d VJP — analogous to conv2d, using MPSGraph's 3D APIs
// (macOS 13.2+).  Bias gradient reduces over (N, D, H, W) → (C_out,).
// ────────────────────────────────────────────────────────────────────
class Conv3dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "conv3d"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return false;

        const auto* S = iv_attr(node, "stride");
        const auto* P = iv_attr(node, "padding");
        const auto* D = iv_attr(node, "dilation");
        if (S == nullptr || P == nullptr || D == nullptr) return false;
        if (S->size() != 3 || P->size() != 3 || D->size() != 3) return false;
        const std::int64_t groups = i_attr(node, "groups", 1);

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* w = as_tensor(bctx.forward(w_id));
        if (g == nil || grad == nil || x == nil || w == nil) return false;

        // Mixed-dtype reconciliation (autocast).
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);
        w = cast_if_needed(g, w, chain_dt);

        MPSGraphConvolution3DOpDescriptor* desc =
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
        if (desc == nil) return false;

        MPSGraphTensor* dX =
            [g convolution3DDataGradientWithIncomingGradientTensor:grad
                                                     weightsTensor:w
                                                       outputShape:x.shape
                                       forwardConvolutionDescriptor:desc
                                                              name:@"conv3d_vjp_dx"];
        bctx.accumulate_grad(x_id, from_tensor(dX));

        MPSGraphTensor* dW =
            [g convolution3DWeightsGradientWithIncomingGradientTensor:grad
                                                        sourceTensor:x
                                                         outputShape:w.shape
                                       forwardConvolutionDescriptor:desc
                                                               name:@"conv3d_vjp_dw"];
        bctx.accumulate_grad(w_id, from_tensor(dW));

        if (b_id >= 0) {
            MPSGraphTensor* b = as_tensor(bctx.forward(b_id));
            std::vector<std::int64_t> w_shape = shape_of_mps(w);
            std::vector<std::int64_t> b_shape =
                (b != nil) ? shape_of_mps(b) : std::vector<std::int64_t>{};
            const bool bias_valid =
                (b != nil) && (b_shape.size() == 1) &&
                (!w_shape.empty()) && (b_shape[0] == w_shape[0]);
            if (bias_valid) {
                NSArray<NSNumber*>* axes_ndhw = @[ @0, @2, @3, @4 ];
                MPSGraphTensor* db_keep =
                    [g reductionSumWithTensor:grad axes:axes_ndhw name:nil];
                NSArray<NSNumber*>* db_shape =
                    @[ [NSNumber numberWithLongLong:w_shape[0]] ];
                MPSGraphTensor* dB =
                    [g reshapeTensor:db_keep withShape:db_shape name:@"conv3d_vjp_db"];
                bctx.accumulate_grad(b_id, from_tensor(dB));
            }
        }
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// ConvTranspose2d VJP.
//
// The data gradient of a transpose-conv is a regular forward conv2d
// using ``grad`` as the source:
//   convT(x, w, desc) backward dX = conv2d(grad, w, desc)
//
// For the weights gradient, MPSGraph provides
// ``convolutionTranspose2DWeightsGradient...`` which takes the
// incoming grad + the forward source ``x`` + the weight shape.
//
// Bias gradient identical to conv2d's: sum over (N, H, W).
// ────────────────────────────────────────────────────────────────────
class ConvTranspose2dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "conv_transpose2d"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return false;

        const auto* S = iv_attr(node, "stride");
        const auto* P = iv_attr(node, "padding");
        if (S == nullptr || P == nullptr) return false;
        if (S->size() != 2 || P->size() != 2) return false;
        const std::int64_t groups = i_attr(node, "groups", 1);

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* w = as_tensor(bctx.forward(w_id));
        if (g == nil || grad == nil || x == nil || w == nil) return false;

        // Mixed-dtype reconciliation (autocast).
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);
        w = cast_if_needed(g, w, chain_dt);

        MPSGraphConvolution2DOpDescriptor* desc =
            make_conv2d_desc((*S)[1], (*S)[0], 1, 1, (*P)[1], (*P)[0], groups);
        if (desc == nil) return false;

        // dX = conv2d_forward(grad, w, desc).  The transpose-conv's
        // "data gradient" is just the corresponding forward conv.
        MPSGraphTensor* dX =
            [g convolution2DWithSourceTensor:grad
                              weightsTensor:w
                                 descriptor:desc
                                       name:@"conv_transpose2d_vjp_dx"];
        bctx.accumulate_grad(x_id, from_tensor(dX));

        // dW via the dedicated MPSGraph API.
        MPSGraphTensor* dW =
            [g convolutionTranspose2DWeightsGradientWithIncomingGradientTensor:grad
                                                                  sourceTensor:x
                                                                   outputShape:w.shape
                                                  forwardConvolutionDescriptor:desc
                                                                          name:@"conv_transpose2d_vjp_dw"];
        bctx.accumulate_grad(w_id, from_tensor(dW));

        // Bias bwd — same shape pattern as conv2d.  For conv-transpose,
        // ``C_out`` is the output channel of grad (NCHW axis 1).
        if (b_id >= 0) {
            MPSGraphTensor* b = as_tensor(bctx.forward(b_id));
            std::vector<std::int64_t> g_shape = shape_of_mps(grad);
            std::vector<std::int64_t> b_shape =
                (b != nil) ? shape_of_mps(b) : std::vector<std::int64_t>{};
            const bool bias_valid =
                (b != nil) && (b_shape.size() == 1) &&
                (g_shape.size() >= 2) && (b_shape[0] == g_shape[1]);
            if (bias_valid) {
                NSArray<NSNumber*>* axes_nhw = @[ @0, @2, @3 ];
                MPSGraphTensor* db_keep =
                    [g reductionSumWithTensor:grad axes:axes_nhw name:nil];
                NSArray<NSNumber*>* db_shape =
                    @[ [NSNumber numberWithLongLong:g_shape[1]] ];
                MPSGraphTensor* dB =
                    [g reshapeTensor:db_keep withShape:db_shape
                                name:@"conv_transpose2d_vjp_db"];
                bctx.accumulate_grad(b_id, from_tensor(dB));
            }
        }
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// ConvTranspose1d VJP — reshape-to-2D trick (same as forward).
// ────────────────────────────────────────────────────────────────────
class ConvTranspose1dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "conv_transpose1d"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId w_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || w_id < 0) return false;

        const auto* S = iv_attr(node, "stride");
        const auto* P = iv_attr(node, "padding");
        if (S == nullptr || P == nullptr) return false;
        if (S->size() != 1 || P->size() != 1) return false;
        const std::int64_t groups = i_attr(node, "groups", 1);

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* w = as_tensor(bctx.forward(w_id));
        if (g == nil || grad == nil || x == nil || w == nil) return false;

        // Mixed-dtype reconciliation (autocast).
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);
        w = cast_if_needed(g, w, chain_dt);

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        std::vector<std::int64_t> w_shape = shape_of_mps(w);
        std::vector<std::int64_t> g_shape = shape_of_mps(grad);
        if (x_shape.size() != 3 || w_shape.size() != 3 || g_shape.size() != 3) return false;

        auto insert_1 = [](std::vector<std::int64_t> v) {
            v.insert(v.begin() + 2, 1);
            return v;
        };
        std::vector<std::int64_t> x_2d_shape = insert_1(x_shape);
        std::vector<std::int64_t> w_2d_shape = insert_1(w_shape);
        std::vector<std::int64_t> g_2d_shape = insert_1(g_shape);
        MPSGraphTensor* x_2d =
            [g reshapeTensor:x withShape:shape_to_ns(x_2d_shape) name:nil];
        MPSGraphTensor* w_2d =
            [g reshapeTensor:w withShape:shape_to_ns(w_2d_shape) name:nil];
        MPSGraphTensor* grad_2d =
            [g reshapeTensor:grad withShape:shape_to_ns(g_2d_shape) name:nil];

        MPSGraphConvolution2DOpDescriptor* desc =
            make_conv2d_desc((*S)[0], 1, 1, 1, (*P)[0], 0, groups);
        if (desc == nil) return false;

        // dX = forward conv2d(grad, w).
        MPSGraphTensor* dX_2d =
            [g convolution2DWithSourceTensor:grad_2d
                              weightsTensor:w_2d
                                 descriptor:desc
                                       name:nil];
        // dW via transpose weights-gradient API.
        MPSGraphTensor* dW_2d =
            [g convolutionTranspose2DWeightsGradientWithIncomingGradientTensor:grad_2d
                                                                  sourceTensor:x_2d
                                                                   outputShape:w_2d.shape
                                                  forwardConvolutionDescriptor:desc
                                                                          name:nil];
        MPSGraphTensor* dX =
            [g reshapeTensor:dX_2d withShape:shape_to_ns(x_shape)
                        name:@"conv_transpose1d_vjp_dx"];
        MPSGraphTensor* dW =
            [g reshapeTensor:dW_2d withShape:shape_to_ns(w_shape)
                        name:@"conv_transpose1d_vjp_dw"];
        bctx.accumulate_grad(x_id, from_tensor(dX));
        bctx.accumulate_grad(w_id, from_tensor(dW));

        if (b_id >= 0) {
            MPSGraphTensor* b = as_tensor(bctx.forward(b_id));
            std::vector<std::int64_t> b_shape =
                (b != nil) ? shape_of_mps(b) : std::vector<std::int64_t>{};
            const bool bias_valid =
                (b != nil) && (b_shape.size() == 1) && (b_shape[0] == g_shape[1]);
            if (bias_valid) {
                NSArray<NSNumber*>* axes_nl = @[ @0, @2 ];
                MPSGraphTensor* db_keep =
                    [g reductionSumWithTensor:grad axes:axes_nl name:nil];
                NSArray<NSNumber*>* db_shape =
                    @[ [NSNumber numberWithLongLong:g_shape[1]] ];
                MPSGraphTensor* dB =
                    [g reshapeTensor:db_keep withShape:db_shape
                                name:@"conv_transpose1d_vjp_db"];
                bctx.accumulate_grad(b_id, from_tensor(dB));
            }
        }
        return true;
    }
};

struct ConvVjpRegistrar {
    ConvVjpRegistrar() {
        register_vjp_emitter(std::make_unique<Conv2dVjp>());
        register_vjp_emitter(std::make_unique<Conv1dVjp>());
        register_vjp_emitter(std::make_unique<Conv3dVjp>());
        register_vjp_emitter(std::make_unique<ConvTranspose2dVjp>());
        register_vjp_emitter(std::make_unique<ConvTranspose1dVjp>());
    }
};

[[maybe_unused]] static const ConvVjpRegistrar g_conv_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
