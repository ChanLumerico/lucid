// lucid/_C/compile/VjpEmitters/nn/Spatial.mm
//
// VJPs for ``interpolate_bilinear``, ``interpolate_nearest_2d``, their 3-D
// counterparts ``interpolate_trilinear`` / ``interpolate_nearest_3d``, and
// ``grid_sample``.
//
// Each is MPSGraph's resize-gradient kernel called with exactly the
// parameters the forward emitter (``OpEmitters/nn/Spatial.mm``) resizes
// with — bilinear with ``centerResult`` on and ``align_corners`` passed
// through, nearest with floor rounding and neither — so the gradient
// scatters through the same index map the forward gathered by.  Without
// them, upsampling put every FCN / MaskFormer / YOLO training step on
// MPSGraph's autodiff, which the train-mode batch norms beside it rule
// out: the steps ran eager.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "../../OpEmitters/_AttrHelpers.h"
#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

bool align_corners_of(const OpNode& node) {
    auto it = node.attrs.find("align_corners");
    if (it == node.attrs.end())
        return false;
    if (const auto* b = std::get_if<bool>(&it->second))
        return *b;
    if (const auto* i = std::get_if<std::int64_t>(&it->second))
        return *i != 0;
    return false;
}

template <bool IS_BILINEAR>
class Interpolate2dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override {
        return IS_BILINEAR ? "interpolate_bilinear" : "interpolate_nearest_2d";
    }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        const bool align = align_corners_of(node);
        return emit_unary_vjp(
            bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (x.shape.count != 4 || node.outputs.empty() || node.outputs[0].shape.size() != 4)
                    return nil;
                for (NSNumber* d in x.shape)
                    if (d.longLongValue <= 0)
                        return nil;
                // Static shapes in and out: the gradient kernel inherits an
                // unresolved shape otherwise (see the nearest forward).
                NSMutableArray<NSNumber*>* out_shape = [NSMutableArray array];
                for (const auto d : node.outputs[0].shape)
                    [out_shape addObject:@(d)];
                go = [g reshapeTensor:go withShape:out_shape name:nil];
                MPSGraphTensor* dx = nil;
                if (IS_BILINEAR)
                    dx = [g resizeWithGradientTensor:go
                                               input:x
                                                mode:MPSGraphResizeBilinear
                                        centerResult:YES
                                        alignCorners:align ? YES : NO
                                              layout:MPSGraphTensorNamedDataLayoutNCHW
                                                name:@"interp2d_bilinear_vjp"];
                else
                    dx = [g resizeNearestWithGradientTensor:go
                                                      input:x
                                        nearestRoundingMode:MPSGraphResizeNearestRoundingModeFloor
                                               centerResult:NO
                                               alignCorners:NO
                                                     layout:MPSGraphTensorNamedDataLayoutNCHW
                                                       name:@"interp2d_nearest_vjp"];
                return [g reshapeTensor:dx withShape:x.shape name:nil];
            });
    }
};

// ``interpolate_trilinear`` / ``interpolate_nearest_3d``.  The forward
// (``Interpolate3dEmitterT``) resizes every depth plane in 2-D and then
// blends — or picks — along depth with fixed weights.  Backwards, the depth
// step is a constant (D × Do) matrix whose column ``i`` holds the weights
// output plane ``i`` read, so it transposes to one matmul; the plane step
// is MPSGraph's resize gradient with the forward's parameters, as in the
// 2-D VJP above.  The weights repeat the forward's arithmetic exactly.
template <bool IS_LINEAR>
class Interpolate3dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override {
        return IS_LINEAR ? "interpolate_trilinear" : "interpolate_nearest_3d";
    }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        const bool align = align_corners_of(node);
        return emit_unary_vjp(
            bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (x.shape.count != 5 || node.outputs.empty() || node.outputs[0].shape.size() != 5)
                    return nil;
                const long long B = x.shape[0].longLongValue;
                const long long C = x.shape[1].longLongValue;
                const long long D = x.shape[2].longLongValue;
                const long long H = x.shape[3].longLongValue;
                const long long W = x.shape[4].longLongValue;
                const long long Do = node.outputs[0].shape[2];
                const long long Ho = node.outputs[0].shape[3];
                const long long Wo = node.outputs[0].shape[4];
                if (B <= 0 || C <= 0 || D <= 0 || H <= 0 || W <= 0 || Do <= 0 || Ho <= 0 || Wo <= 0)
                    return nil;

                std::vector<float> depth(static_cast<std::size_t>(D * Do), 0.0f);
                for (long long i = 0; i < Do; ++i) {
                    if (IS_LINEAR) {
                        double p;
                        if (align)
                            p = (Do <= 1) ? 0.0
                                          : static_cast<double>(i) * (D - 1) /
                                                static_cast<double>(Do - 1);
                        else
                            p = (static_cast<double>(i) + 0.5) * D / static_cast<double>(Do) - 0.5;
                        if (p < 0.0)
                            p = 0.0;
                        if (p > static_cast<double>(D - 1))
                            p = static_cast<double>(D - 1);
                        const long long lo = std::min((long long)p, D - 1);
                        const long long hi = std::min(lo + 1, D - 1);
                        const float w = (float)(p - (double)lo);
                        depth[(std::size_t)(lo * Do + i)] += 1.0f - w;
                        depth[(std::size_t)(hi * Do + i)] += w;
                    } else {
                        long long src =
                            (long long)std::floor(static_cast<double>(i) * D / (double)Do);
                        if (src < 0)
                            src = 0;
                        if (src > D - 1)
                            src = D - 1;
                        depth[(std::size_t)(src * Do + i)] = 1.0f;
                    }
                }
                NSData* nsd = [NSData dataWithBytes:depth.data()
                                             length:depth.size() * sizeof(float)];
                MPSGraphTensor* m = [g constantWithData:nsd
                                                  shape:@[ @1, @(D), @(Do) ]
                                               dataType:MPSDataTypeFloat32];
                m = [g castTensor:m toType:go.dataType name:nil];

                MPSGraphTensor* g3 = [g reshapeTensor:go
                                            withShape:@[ @(B * C), @(Do), @(Ho * Wo) ]
                                                 name:nil];
                MPSGraphTensor* planes = [g matrixMultiplicationWithPrimaryTensor:m
                                                                  secondaryTensor:g3
                                                                             name:nil];
                planes = [g reshapeTensor:planes
                                withShape:@[ @(B * C), @(D), @(Ho), @(Wo) ]
                                     name:nil];
                MPSGraphTensor* folded = [g reshapeTensor:x
                                                withShape:@[ @(B * C), @(D), @(H), @(W) ]
                                                     name:nil];
                MPSGraphTensor* dx = nil;
                if (IS_LINEAR)
                    dx = [g resizeWithGradientTensor:planes
                                               input:folded
                                                mode:MPSGraphResizeBilinear
                                        centerResult:YES
                                        alignCorners:align ? YES : NO
                                              layout:MPSGraphTensorNamedDataLayoutNCHW
                                                name:@"interp3d_linear_vjp"];
                else
                    dx = [g resizeNearestWithGradientTensor:planes
                                                      input:folded
                                        nearestRoundingMode:MPSGraphResizeNearestRoundingModeFloor
                                               centerResult:NO
                                               alignCorners:NO
                                                     layout:MPSGraphTensorNamedDataLayoutNCHW
                                                       name:@"interp3d_nearest_vjp"];
                return dx == nil ? nil : [g reshapeTensor:dx withShape:x.shape name:nil];
            });
    }
};

// ``grid_sample`` — the forward's four-corner gather run backwards.
//
// Mirrors ``CpuBackend::grid_sample_backward`` term for term, as the
// forward emitter mirrors its forward: the input gradient scatters each
// output's gradient into the corners it read, weighted as it read them (an
// out-of-range corner is skipped under zero padding and clamped to the
// edge under border padding); the grid gradient is the blend's slope along
// each axis, summed over channels, times the denormalisation scale — and
// zero on an axis the border clamp pinned.  Nearest sampling has no slope.
// Without it every Mask2Former step ran eager: its deformable attention
// samples with ``grid_sample``, and the offsets it samples at are learned.
class GridSampleVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "grid_sample"; }

    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        BinaryVjpCtx c = unpack_binary_vjp(bctx, node, grad_outs);
        if (!c.ok)
            return false;
        MPSGraph* g = c.g;
        MPSGraphTensor* go = c.go;
        const MPSDataType dt = go.dataType;
        MPSGraphTensor* x = cast_if_needed(g, c.a, dt);
        MPSGraphTensor* grid = cast_if_needed(g, c.b, dt);
        if (x.shape.count != 4 || grid.shape.count != 4 || grid.shape[3].longLongValue != 2)
            return false;
        const long long N = x.shape[0].longLongValue;
        const long long C = x.shape[1].longLongValue;
        const long long H = x.shape[2].longLongValue;
        const long long W = x.shape[3].longLongValue;
        const long long Ho = grid.shape[1].longLongValue;
        const long long Wo = grid.shape[2].longLongValue;
        if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || Ho <= 0 || Wo <= 0)
            return false;
        const long long L = Ho * Wo;
        const std::int64_t mode = int_attr(node, "mode", 0);
        const std::int64_t padding = int_attr(node, "padding_mode", 0);
        const bool align = bool_attr(node, "align_corners", false);
        if (mode < 0 || mode > 1 || padding < 0 || padding > 1)
            return false;

        const auto real = [&](double v) { return [g constantWithScalar:v dataType:dt]; };
        const auto add = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
        const auto sub = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
        const auto mul = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
        const auto as_real = [&](MPSGraphTensor* b) { return [g castTensor:b toType:dt name:nil]; };

        MPSGraphTensor* go3 = [g reshapeTensor:go withShape:@[ @(N), @(C), @(L) ] name:nil];
        MPSGraphTensor* flat_x = [g reshapeTensor:x withShape:@[ @(N), @(C), @(H * W) ] name:nil];
        const auto coordinate = [&](NSInteger which) -> MPSGraphTensor* {
            MPSGraphTensor* s = [g sliceTensor:grid dimension:3 start:which length:1 name:nil];
            return [g reshapeTensor:s withShape:@[ @(N), @1, @(L) ] name:nil];
        };
        const auto denormalise = [&](MPSGraphTensor* t, long long extent) -> MPSGraphTensor* {
            MPSGraphTensor* shifted = add(t, real(1.0));
            if (align)
                return mul(shifted, real(static_cast<double>(extent - 1) * 0.5));
            return sub(mul(shifted, real(static_cast<double>(extent) * 0.5)), real(0.5));
        };
        MPSGraphTensor* ix = denormalise(coordinate(0), W);
        MPSGraphTensor* iy = denormalise(coordinate(1), H);
        MPSGraphTensor* x_lo = real(0.0);
        MPSGraphTensor* x_hi = real(static_cast<double>(W - 1));
        MPSGraphTensor* y_lo = real(0.0);
        MPSGraphTensor* y_hi = real(static_cast<double>(H - 1));

        // One corner: its flat index (clamped into range, broadcast over the
        // channels) and whether it counts — always under border padding,
        // only in range under zero padding.
        struct Corner {
            MPSGraphTensor* index;
            MPSGraphTensor* keep;
        };
        const auto corner = [&](MPSGraphTensor* px, MPSGraphTensor* py) -> Corner {
            MPSGraphTensor* keep = real(1.0);
            if (padding == 0) {
                MPSGraphTensor* a = [g greaterThanOrEqualToWithPrimaryTensor:px
                                                             secondaryTensor:x_lo
                                                                        name:nil];
                MPSGraphTensor* b = [g lessThanOrEqualToWithPrimaryTensor:px
                                                          secondaryTensor:x_hi
                                                                     name:nil];
                MPSGraphTensor* cc = [g greaterThanOrEqualToWithPrimaryTensor:py
                                                              secondaryTensor:y_lo
                                                                         name:nil];
                MPSGraphTensor* d = [g lessThanOrEqualToWithPrimaryTensor:py
                                                          secondaryTensor:y_hi
                                                                     name:nil];
                keep = as_real([g logicalANDWithPrimaryTensor:[g logicalANDWithPrimaryTensor:a
                                                                             secondaryTensor:b
                                                                                        name:nil]
                                              secondaryTensor:[g logicalANDWithPrimaryTensor:cc
                                                                             secondaryTensor:d
                                                                                        name:nil]
                                                         name:nil]);
            }
            MPSGraphTensor* cx = [g clampWithTensor:px
                                     minValueTensor:x_lo
                                     maxValueTensor:x_hi
                                               name:nil];
            MPSGraphTensor* cy = [g clampWithTensor:py
                                     minValueTensor:y_lo
                                     maxValueTensor:y_hi
                                               name:nil];
            MPSGraphTensor* xi = [g castTensor:cx toType:MPSDataTypeInt32 name:nil];
            MPSGraphTensor* yi = [g castTensor:cy toType:MPSDataTypeInt32 name:nil];
            MPSGraphTensor* stride = [g constantWithScalar:(double)W dataType:MPSDataTypeInt32];
            MPSGraphTensor* flat =
                [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:yi
                                                                secondaryTensor:stride
                                                                           name:nil]
                             secondaryTensor:xi
                                        name:nil];
            return {
                [g broadcastTensor:flat toShape:@[ @(N), @(C), @(L) ] name:nil], keep
            };
        };
        const auto read = [&](const Corner& k) -> MPSGraphTensor* {
            MPSGraphTensor* v = [g gatherAlongAxis:2
                                 withUpdatesTensor:flat_x
                                     indicesTensor:k.index
                                              name:nil];
            return mul(v, k.keep);
        };

        std::vector<Corner> corners;
        std::vector<MPSGraphTensor*> weights;
        MPSGraphTensor* dgrid = nil;
        if (mode == 1) {
            corners.push_back(corner([g rintWithTensor:ix name:nil], [g rintWithTensor:iy
                                                                                  name:nil]));
            weights.push_back(real(1.0));
            dgrid = [g constantWithScalar:0.0 shape:grid.shape dataType:dt];
        } else {
            MPSGraphTensor* stuck_x = nil;
            MPSGraphTensor* stuck_y = nil;
            if (padding == 1) {
                // Border clamps the coordinate, and a clamped axis has no
                // slope: the engine zeroes its grid gradient.
                stuck_x = as_real([g logicalORWithPrimaryTensor:[g lessThanWithPrimaryTensor:ix
                                                                             secondaryTensor:x_lo
                                                                                        name:nil]
                                                secondaryTensor:[g greaterThanWithPrimaryTensor:ix
                                                                                secondaryTensor:x_hi
                                                                                           name:nil]
                                                           name:nil]);
                stuck_y = as_real([g logicalORWithPrimaryTensor:[g lessThanWithPrimaryTensor:iy
                                                                             secondaryTensor:y_lo
                                                                                        name:nil]
                                                secondaryTensor:[g greaterThanWithPrimaryTensor:iy
                                                                                secondaryTensor:y_hi
                                                                                           name:nil]
                                                           name:nil]);
                ix = [g clampWithTensor:ix minValueTensor:x_lo maxValueTensor:x_hi name:nil];
                iy = [g clampWithTensor:iy minValueTensor:y_lo maxValueTensor:y_hi name:nil];
            }
            MPSGraphTensor* x0 = [g floorWithTensor:ix name:nil];
            MPSGraphTensor* y0 = [g floorWithTensor:iy name:nil];
            MPSGraphTensor* x1 = add(x0, real(1.0));
            MPSGraphTensor* y1 = add(y0, real(1.0));
            MPSGraphTensor* wx1 = sub(x1, ix);
            MPSGraphTensor* wx0 = sub(ix, x0);
            MPSGraphTensor* wy1 = sub(y1, iy);
            MPSGraphTensor* wy0 = sub(iy, y0);
            // a = (y0, x0), b = (y1, x0), c = (y0, x1), d = (y1, x1).
            corners = {corner(x0, y0), corner(x0, y1), corner(x1, y0), corner(x1, y1)};
            weights = {mul(wx1, wy1), mul(wx1, wy0), mul(wx0, wy1), mul(wx0, wy0)};

            MPSGraphTensor* va = read(corners[0]);
            MPSGraphTensor* vb = read(corners[1]);
            MPSGraphTensor* vc = read(corners[2]);
            MPSGraphTensor* vd = read(corners[3]);
            MPSGraphTensor* slope_x = add(mul(sub(vc, va), wy1), mul(sub(vd, vb), wy0));
            MPSGraphTensor* slope_y = add(mul(sub(vb, va), wx1), mul(sub(vd, vc), wx0));
            MPSGraphTensor* dix = [g reductionSumWithTensor:mul(go3, slope_x) axis:1 name:nil];
            MPSGraphTensor* diy = [g reductionSumWithTensor:mul(go3, slope_y) axis:1 name:nil];
            if (padding == 1) {
                dix = mul(dix, sub(real(1.0), stuck_x));
                diy = mul(diy, sub(real(1.0), stuck_y));
            }
            const double sx =
                align ? static_cast<double>(W - 1) * 0.5 : static_cast<double>(W) * 0.5;
            const double sy =
                align ? static_cast<double>(H - 1) * 0.5 : static_cast<double>(H) * 0.5;
            dix = [g reshapeTensor:mul(dix, real(sx))
                         withShape:@[ @(N), @(Ho), @(Wo), @1 ]
                              name:nil];
            diy = [g reshapeTensor:mul(diy, real(sy))
                         withShape:@[ @(N), @(Ho), @(Wo), @1 ]
                              name:nil];
            dgrid = [g concatTensors:@[ dix, diy ] dimension:3 name:nil];
        }

        // Every corner's contribution in one scatter-add along the flattened
        // spatial axis.
        NSMutableArray<MPSGraphTensor*>* updates = [NSMutableArray array];
        NSMutableArray<MPSGraphTensor*>* indices = [NSMutableArray array];
        for (std::size_t k = 0; k < corners.size(); ++k) {
            [updates addObject:mul(go3, mul(weights[k], corners[k].keep))];
            [indices addObject:corners[k].index];
        }
        MPSGraphTensor* base = [g constantWithScalar:0.0
                                               shape:@[ @(N), @(C), @(H * W) ]
                                            dataType:dt];
        MPSGraphTensor* dx = [g scatterAlongAxis:2
                                  withDataTensor:base
                                   updatesTensor:[g concatTensors:updates dimension:2 name:nil]
                                   indicesTensor:[g concatTensors:indices dimension:2 name:nil]
                                            mode:MPSGraphScatterModeAdd
                                            name:@"grid_sample_vjp_x"];
        if (dx == nil || dgrid == nil)
            return false;
        bctx.accumulate_grad(c.a_id, from_tensor([g reshapeTensor:dx withShape:x.shape name:nil]));
        bctx.accumulate_grad(c.b_id, from_tensor(dgrid));
        return true;
    }
};

struct SpatialVjpRegistrar {
    SpatialVjpRegistrar() {
        register_vjp_emitter(std::make_unique<Interpolate2dVjp<true>>());
        register_vjp_emitter(std::make_unique<Interpolate2dVjp<false>>());
        register_vjp_emitter(std::make_unique<Interpolate3dVjp<true>>());
        register_vjp_emitter(std::make_unique<Interpolate3dVjp<false>>());
        register_vjp_emitter(std::make_unique<GridSampleVjp>());
    }
};

[[maybe_unused]] static const SpatialVjpRegistrar g_spatial_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
