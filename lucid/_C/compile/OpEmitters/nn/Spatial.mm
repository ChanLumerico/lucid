// lucid/_C/compile/OpEmitters/nn/Spatial.mm
//
// Spatial-domain emitters that turn 4-D (NCHW) feature maps into
// other 4-D feature maps:
//
//   - ``affine_grid``           — theta @ homogeneous-coord constants
//   - ``interpolate_{nearest_2d,bilinear,nearest_3d,trilinear}`` — per-axis
//                                  gathers and blends (``_Resample.h``)
//   - ``unfold_dim``             — sliding window via slice + concat + permute
//
//   - ``grid_sample``            — four-corner gather over flattened
//                                  spatial axes, then a blend
//
// ``rotate`` stays in :file:`../special/Stubs.mm` — not because it
// cannot be emitted (it is nearest with a compile-time constant
// grid, so it is a baked gather table) but because no Python caller
// anywhere in ``lucid/`` reaches ``engine.nn.rotate``, so an emitter
// for it could never run or be tested.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../_AttrHelpers.h"
#include "../_IndexBounds.h"
#include "../_Resample.h"

namespace lucid::compile {

namespace {

// ── affine_grid — theta (N, 2, 3) @ coords^T (3, H*W) → (N, H, W, 2).
// Build the (H*W, 3) homogeneous-coord constant at emit time from
// attrs H/W/align_corners, then matmul / reshape / permute.
class AffineGridEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "affine_grid"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId t_id = node.inputs[0];
        if (t_id < 0) return false;
        std::int64_t H = int_attr(node, "H", 0);
        std::int64_t W = int_attr(node, "W", 0);
        if (H <= 0 || W <= 0) return false;
        bool align_corners = bool_attr(node, "align_corners", false);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* theta = (__bridge MPSGraphTensor*)ctx.resolve(t_id);
        if (g == nil || theta == nil) return false;
        if (theta.shape.count != 3 || theta.shape[1].longLongValue != 2 ||
            theta.shape[2].longLongValue != 3) return false;
        std::vector<float> coords(static_cast<size_t>(H * W * 3));
        auto make_axis = [](std::int64_t n, bool ac) {
            std::vector<float> v(static_cast<size_t>(n));
            if (n == 1) { v[0] = 0.0f; return v; }
            if (ac) {
                for (std::int64_t i = 0; i < n; ++i)
                    v[i] = -1.0f + 2.0f * (float)i / (float)(n - 1);
            } else {
                for (std::int64_t i = 0; i < n; ++i)
                    v[i] = ((2.0f * (float)i + 1.0f) / (float)n) - 1.0f;
            }
            return v;
        };
        auto us = make_axis(W, align_corners);
        auto vs = make_axis(H, align_corners);
        for (std::int64_t i = 0; i < H; ++i) {
            for (std::int64_t j = 0; j < W; ++j) {
                std::int64_t row = i * W + j;
                coords[row * 3 + 0] = us[(size_t)j];
                coords[row * 3 + 1] = vs[(size_t)i];
                coords[row * 3 + 2] = 1.0f;
            }
        }
        NSData* nsd = [NSData dataWithBytes:coords.data()
                                      length:coords.size() * sizeof(float)];
        MPSGraphTensor* coords_t = [g constantWithData:nsd
                                                 shape:@[[NSNumber numberWithLongLong:H * W], @3]
                                              dataType:MPSDataTypeFloat32];
        if (coords_t.dataType != theta.dataType) {
            coords_t = [g castTensor:coords_t toType:theta.dataType name:nil];
        }
        MPSGraphTensor* coords_T =
            [g transposeTensor:coords_t dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* grid_flat =
            [g matrixMultiplicationWithPrimaryTensor:theta
                                       secondaryTensor:coords_T
                                                  name:@"affine_grid_mat"];
        NSNumber* N_n = theta.shape[0];
        NSArray<NSNumber*>* g4 = @[N_n, @2,
                                    [NSNumber numberWithLongLong:H],
                                    [NSNumber numberWithLongLong:W]];
        MPSGraphTensor* grid_nchw = [g reshapeTensor:grid_flat withShape:g4 name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g transposeTensor:grid_nchw
                                       permutation:@[@0, @2, @3, @1]
                                              name:@"affine_grid"]));
        return true;
    }
};

// ── interpolate_{nearest_2d,bilinear,nearest_3d,trilinear} — the eager
// Metal kernel's gathers and blends at the shared sample positions
// (``_Resample.h``).  Only the trailing axes are touched, so a symbolic
// batch rides through.
template <ResampleFilter FILTER, std::size_t RANK>
class InterpolateEmitterT final : public OpEmitter {
public:
    explicit InterpolateEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const UnaryInput u = resolve_unary(ctx, node);
        std::vector<std::int64_t> in, out;
        if (!u.ok || !graph_resample_extents(u.x, node.outputs[0].shape, RANK, in, out))
            return false;
        const bool align =
            FILTER == ResampleFilter::Linear && bool_attr(node, "align_corners", false);
        MPSGraphTensor* y = graph_resample(u.g, u.x, in, out, FILTER, align, false);
        if (y == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }

private:
    std::string name_;
};

// ── unfold_dim — sliding window along ``dim`` via slice + concat + permute.
// Output shape: (..., L, ..., size) where L = (dim_size - size)/step + 1
// replaces axis d and a new ``size`` axis is appended last.
class UnfoldDimEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "unfold_dim"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        std::int64_t d = int_attr(node, "dim", 0);
        std::int64_t size = int_attr(node, "size", 0);
        std::int64_t step = int_attr(node, "step", 0);
        if (size <= 0 || step <= 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSInteger nd = (NSInteger)x.shape.count;
        if (d < 0 || d >= nd) return false;
        std::int64_t dim_size = x.shape[(NSUInteger)d].longLongValue;
        std::int64_t L = (dim_size - size) / step + 1;
        if (L <= 0) return false;
        NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray array];
        for (std::int64_t l = 0; l < L; ++l) {
            MPSGraphTensor* s = [g sliceTensor:x
                                     dimension:(NSInteger)d
                                         start:(NSInteger)(l * step)
                                        length:(NSInteger)size
                                          name:nil];
            NSMutableArray<NSNumber*>* new_sh = [NSMutableArray array];
            for (NSUInteger k = 0; k < s.shape.count; ++k) {
                if ((NSInteger)k == d)
                    [new_sh addObject:@1];
                [new_sh addObject:s.shape[k]];
            }
            if ((NSInteger)new_sh.count == nd)
                [new_sh addObject:@1];
            [parts addObject:[g reshapeTensor:s withShape:new_sh name:nil]];
        }
        MPSGraphTensor* stacked = [g concatTensors:parts
                                          dimension:(NSInteger)d
                                               name:@"unfold_dim_concat"];
        NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
        NSInteger src_size_axis = d + 1;
        NSInteger total_axes = nd + 1;
        for (NSInteger k = 0; k < total_axes; ++k) {
            if (k == src_size_axis) continue;
            [perm addObject:[NSNumber numberWithLongLong:k]];
        }
        [perm addObject:[NSNumber numberWithLongLong:src_size_axis]];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g transposeTensor:stacked
                                       permutation:perm
                                              name:@"unfold_dim"]));
        return true;
    }
};

// ── grid_sample — the four-corner gather MPSGraph has no single op for.
//
// This was a stub on the grounds that "MPSGraph's ``gatherAlongAxis``
// can't express a per-pixel bilinear gather".  It can, once the spatial
// axes are flattened: the sample coordinates become one integer index
// per output position, and ``gatherAlongAxis`` takes an index *tensor*,
// so the data-dependence is not the obstacle it looks like.  What is
// left is arithmetic — denormalise, floor, clamp, four gathers, blend.
//
// Every constant here mirrors ``CpuBackend::grid_sample_forward``
// rather than being re-derived, because the two have to agree on the
// half-pixel question at every corner.
class GridSampleEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "grid_sample"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return false;
        const TensorId x_id = node.inputs[0];
        const TensorId g_id = node.inputs[1];
        if (x_id < 0 || g_id < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* grid = (__bridge MPSGraphTensor*)ctx.resolve(g_id);
        if (g == nil || x == nil || grid == nil)
            return false;
        if (x.shape.count != 4 || grid.shape.count != 4)
            return false;
        // The flatten below folds the batch into an index arithmetic that
        // a symbolic batch cannot carry.
        if (symbolic_batch_at_dim0(x) || symbolic_batch_at_dim0(grid))
            return false;

        const long long N = x.shape[0].longLongValue;
        const long long C = x.shape[1].longLongValue;
        const long long H = x.shape[2].longLongValue;
        const long long W = x.shape[3].longLongValue;
        const long long Ho = grid.shape[1].longLongValue;
        const long long Wo = grid.shape[2].longLongValue;
        if (grid.shape[3].longLongValue != 2)
            return false;
        if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || Ho <= 0 || Wo <= 0)
            return false;
        const long long L = Ho * Wo;

        // 0 = bilinear / zeros, 1 = nearest / border — the engine's own
        // encoding (see ``F.grid_sample``'s mode_map / pad_map).
        const std::int64_t mode = int_attr(node, "mode", 0);
        const std::int64_t padding = int_attr(node, "padding_mode", 0);
        const bool align = bool_attr(node, "align_corners", false);
        if (mode < 0 || mode > 1 || padding < 0 || padding > 1)
            return false;

        const MPSDataType dt = x.dataType;
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

        MPSGraphTensor* flat_x = [g reshapeTensor:x withShape:@[ @(N), @(C), @(H * W) ] name:nil];

        // grid is (N, Ho, Wo, 2) with the pair ordered (x, y).
        const auto coordinate = [&](NSInteger which) -> MPSGraphTensor* {
            MPSGraphTensor* s = [g sliceTensor:grid dimension:3 start:which length:1 name:nil];
            return [g reshapeTensor:s withShape:@[ @(N), @1, @(L) ] name:nil];
        };
        // ``align_corners`` puts the extreme coordinates on the centres of
        // the corner pixels; otherwise on their outer edges.
        const auto denormalise = [&](MPSGraphTensor* t, long long extent) -> MPSGraphTensor* {
            MPSGraphTensor* shifted = add(t, real(1.0));
            if (align)
                return mul(shifted, real(static_cast<double>(extent - 1) * 0.5));
            return sub(mul(shifted, real(static_cast<double>(extent) * 0.5)), real(0.5));
        };
        MPSGraphTensor* ix = denormalise(coordinate(0), W);
        MPSGraphTensor* iy = denormalise(coordinate(1), H);
        if (ix == nil || iy == nil)
            return false;

        MPSGraphTensor* x_lo = real(0.0);
        MPSGraphTensor* x_hi = real(static_cast<double>(W - 1));
        MPSGraphTensor* y_lo = real(0.0);
        MPSGraphTensor* y_hi = real(static_cast<double>(H - 1));

        // Reading one input pixel per output position: bring the
        // coordinates into range, flatten them to a single index, and let
        // ``gatherAlongAxis`` do the data-dependent part.  Under zero
        // padding the out-of-range positions are replaced by 0 after the
        // gather — selected, not multiplied, so an infinite pixel read at
        // the clamped position does not turn into NaN (eager's rule).
        const auto sample = [&](MPSGraphTensor* px, MPSGraphTensor* py) -> MPSGraphTensor* {
            const GraphPixel pixel = graph_pixel(g, px, py, H, W, padding == 0);
            MPSGraphTensor* spread = [g broadcastTensor:pixel.flat
                                                toShape:@[ @(N), @(C), @(L) ]
                                                   name:nil];
            MPSGraphTensor* v = [g gatherAlongAxis:2
                                 withUpdatesTensor:flat_x
                                     indicesTensor:spread
                                              name:nil];
            if (v != nil && pixel.inside != nil)
                v = [g selectWithPredicateTensor:pixel.inside
                             truePredicateTensor:v
                            falsePredicateTensor:real(0.0)
                                            name:nil];
            return v;
        };

        MPSGraphTensor* out = nil;
        if (mode == 1) {
            // Nearest: ``rint`` is round-half-to-even, matching the
            // engine's ``nearbyint`` under the default rounding mode.
            out = sample([g rintWithTensor:ix name:nil], [g rintWithTensor:iy name:nil]);
        } else {
            if (padding == 1) {
                // Border clamps the *coordinate* before the corners are
                // taken, so both corners land on the edge pixel and the
                // blend degenerates there.  Clamping the corners instead
                // would interpolate against a pixel the engine never
                // reads.
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

            MPSGraphTensor* v00 = sample(x0, y0);
            MPSGraphTensor* v01 = sample(x0, y1);
            MPSGraphTensor* v10 = sample(x1, y0);
            MPSGraphTensor* v11 = sample(x1, y1);
            if (v00 == nil || v01 == nil || v10 == nil || v11 == nil)
                return false;
            out = add(add(mul(v00, mul(wx1, wy1)), mul(v01, mul(wx1, wy0))),
                      add(mul(v10, mul(wx0, wy1)), mul(v11, mul(wx0, wy0))));
        }
        if (out == nil)
            return false;

        ctx.bind(node.outputs[0].id, (__bridge void*)([g reshapeTensor:out
                                                             withShape:@[ @(N), @(C), @(Ho), @(Wo) ]
                                                                  name:@"grid_sample"]));
        return true;
    }
};

struct SpatialRegistrar {
    SpatialRegistrar() {
        register_emitter(std::make_unique<AffineGridEmitter>());
        register_emitter(std::make_unique<InterpolateEmitterT<ResampleFilter::Linear, 2>>(
            "interpolate_bilinear"));
        register_emitter(std::make_unique<InterpolateEmitterT<ResampleFilter::Nearest, 2>>(
            "interpolate_nearest_2d"));
        register_emitter(std::make_unique<InterpolateEmitterT<ResampleFilter::Linear, 3>>(
            "interpolate_trilinear"));
        register_emitter(std::make_unique<InterpolateEmitterT<ResampleFilter::Nearest, 3>>(
            "interpolate_nearest_3d"));
        register_emitter(std::make_unique<UnfoldDimEmitter>());
        register_emitter(std::make_unique<GridSampleEmitter>());
    }
};

[[maybe_unused]] static const SpatialRegistrar g_spatial_registrar;

}  // namespace

}  // namespace lucid::compile
