// lucid/_C/compile/VjpEmitters/nn/Spatial.mm
//
// VJPs for ``interpolate_bilinear``, ``interpolate_nearest_2d``, their 3-D
// counterparts ``interpolate_trilinear`` / ``interpolate_nearest_3d``, and
// ``grid_sample``.
//
// The interpolate VJPs run the forward emitter's per-axis gathers backwards
// through the same position tables (``OpEmitters/_Resample.h``), as the
// eager adjoint does.  Without them, upsampling put every FCN / MaskFormer /
// YOLO training step on MPSGraph's autodiff, which the train-mode batch
// norms beside it rule out: the steps ran eager.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "../../OpEmitters/_AttrHelpers.h"
#include "../../OpEmitters/_IndexBounds.h"
#include "../../OpEmitters/_Resample.h"
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

// ``interpolate_{bilinear,nearest_2d,trilinear,nearest_3d}`` — the forward's
// gathers run backwards: each axis scatters the gradient onto the samples it
// read, with the weights it read them at, in the eager adjoint's order
// (``_Resample.h``).  The scatter writes a static shape, so a symbolic batch
// declines.
template <ResampleFilter FILTER, std::size_t RANK>
class InterpolateVjp final : public VjpEmitter {
public:
    explicit InterpolateVjp(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        const bool align = FILTER == ResampleFilter::Linear && align_corners_of(node);
        return emit_unary_vjp(
            bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (node.outputs.empty() || symbolic_batch_at_dim0(x))
                    return nil;
                const Shape& recorded = node.outputs[0].shape;
                std::vector<std::int64_t> in, out;
                if (!graph_resample_extents(x, recorded, RANK, in, out))
                    return nil;
                NSMutableArray<NSNumber*>* out_shape = [NSMutableArray array];
                for (const auto d : recorded)
                    [out_shape addObject:@(d)];
                go = [g reshapeTensor:go withShape:out_shape name:nil];
                return graph_resample(g, go, in, out, FILTER, align, true);
            });
    }

private:
    std::string name_;
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
            MPSGraphTensor* inside;  // bool; nil under border padding
        };
        const auto corner = [&](MPSGraphTensor* px, MPSGraphTensor* py) -> Corner {
            const GraphPixel pixel = graph_pixel(g, px, py, H, W, padding == 0);
            return {
                [g broadcastTensor:pixel.flat toShape:@[ @(N), @(C), @(L) ] name:nil],
                    pixel.inside == nil ? real(1.0) : as_real(pixel.inside), pixel.inside
            };
        };
        // Selected, not multiplied, as the forward reads it: an infinite
        // pixel at the clamped position of an outside corner is 0 there.
        const auto read = [&](const Corner& k) -> MPSGraphTensor* {
            MPSGraphTensor* v = [g gatherAlongAxis:2
                                 withUpdatesTensor:flat_x
                                     indicesTensor:k.index
                                              name:nil];
            if (k.inside == nil)
                return v;
            return [g selectWithPredicateTensor:k.inside
                            truePredicateTensor:v
                           falsePredicateTensor:real(0.0)
                                           name:nil];
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
                ix = graph_clamp_keeping_nan(g, ix, x_lo, x_hi);
                iy = graph_clamp_keeping_nan(g, iy, y_lo, y_hi);
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
        register_vjp_emitter(
            std::make_unique<InterpolateVjp<ResampleFilter::Linear, 2>>("interpolate_bilinear"));
        register_vjp_emitter(
            std::make_unique<InterpolateVjp<ResampleFilter::Nearest, 2>>("interpolate_nearest_2d"));
        register_vjp_emitter(
            std::make_unique<InterpolateVjp<ResampleFilter::Linear, 3>>("interpolate_trilinear"));
        register_vjp_emitter(
            std::make_unique<InterpolateVjp<ResampleFilter::Nearest, 3>>("interpolate_nearest_3d"));
        register_vjp_emitter(std::make_unique<GridSampleVjp>());
    }
};

[[maybe_unused]] static const SpatialVjpRegistrar g_spatial_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
