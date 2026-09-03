// lucid/_C/compile/OpEmitters/nn/Spatial.mm
//
// Spatial-domain emitters that turn 4-D (NCHW) feature maps into
// other 4-D feature maps:
//
//   - ``affine_grid``           — theta @ homogeneous-coord constants
//   - ``interpolate_nearest_2d`` — MPSGraph ``resizeTensor`` (mode=nearest)
//   - ``interpolate_bilinear``   — MPSGraph ``resizeTensor`` (mode=bilinear)
//   - ``interpolate_nearest_3d`` — the plane through ``resizeTensor``,
//                                  the depth through a gather
//   - ``interpolate_trilinear``  — the same, with a two-slice blend
//   - ``unfold_dim``             — sliding window via slice + concat + permute
//
// ``grid_sample`` and ``rotate`` stay in :file:`../misc/Stubs.mm`
// because they need a per-pixel bilinear gather that MPSGraph's
// ``gatherAlongAxis`` API cannot express in one step.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../_AttrHelpers.h"

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

// ── interpolate_{nearest_2d,bilinear} — MPSGraph resizeTensor.
template <bool IS_BILINEAR>
class Interpolate2dEmitterT final : public OpEmitter {
public:
    explicit Interpolate2dEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        std::int64_t H_out = int_attr(node, "H_out", 0);
        std::int64_t W_out = int_attr(node, "W_out", 0);
        if (H_out <= 0 || W_out <= 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        if (x.shape.count != 4) return false;
        bool align_corners = bool_attr(node, "align_corners", false);
        if (IS_BILINEAR) {
            // Bilinear path: matches the reference framework with
            // centerResult=YES, alignCorners flag passed through.
            MPSShape* size_2 = @[[NSNumber numberWithLongLong:H_out],
                                  [NSNumber numberWithLongLong:W_out]];
            ctx.bind(node.outputs[0].id, (__bridge void*)([g resizeTensor:x
                                              size:size_2
                                              mode:MPSGraphResizeBilinear
                                      centerResult:YES
                                      alignCorners:align_corners ? YES : NO
                                            layout:MPSGraphTensorNamedDataLayoutNCHW
                                              name:@"interp2d_bilinear"]));
        return true;
        }
        // Nearest path: the reference framework's
        // ``F.interpolate(mode='nearest')`` uses
        // floor(dst * src_size / dst_size) → exact kron-style block
        // upsampling.  MPSGraph's plain ``resizeTensor:mode:nearest`` uses
        // ``RoundPreferCeil`` and gives a different mapping.  Use the
        // ``resizeNearestWithTensor:`` overload that lets us pin the
        // rounding mode to Floor.  size is passed as a 1-D Int32 tensor.
        std::int32_t size_data[2] = { (std::int32_t)H_out, (std::int32_t)W_out };
        NSData* size_nsd = [NSData dataWithBytes:size_data length:sizeof(size_data)];
        MPSGraphTensor* size_t =
            [g constantWithData:size_nsd shape:@[@2] dataType:MPSDataTypeInt32];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g resizeNearestWithTensor:x
                            sizeTensor:size_t
                   nearestRoundingMode:MPSGraphResizeNearestRoundingModeFloor
                          centerResult:NO
                          alignCorners:NO
                                layout:MPSGraphTensorNamedDataLayoutNCHW
                                  name:@"interp2d_nearest"]));
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

// ── interpolate_{nearest_3d,trilinear} — the plane through
// ``resizeTensor``, the depth through a gather.
//
// MPSGraph's resamplers are two-dimensional.  That is a fact about the
// rank they accept, not about the operation: resampling is separable, so
// the depth axis can ride along as channels while height and width are
// resized, and depth is then its own one-dimensional resample over the
// result.  Nearest is a gather of one slice per output plane; linear is a
// gather of two and a blend.  Both index tables are fixed by the input
// and output extents, so they are constants rather than arithmetic.
//
// The fold merges batch and channels, which a symbolic batch cannot
// express (see ``reshape_dynamic_aware``) — a dynamic-batch compile
// declines here rather than pinning the trace-time batch into the graph.
template <bool IS_LINEAR>
class Interpolate3dEmitterT final : public OpEmitter {
public:
    explicit Interpolate3dEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil)
            return false;
        if (x.shape.count != 5)
            return false;
        if (symbolic_batch_at_dim0(x))
            return false;

        const Shape& out_shape = node.outputs[0].shape;
        if (out_shape.size() != 5)
            return false;
        const long long B = x.shape[0].longLongValue;
        const long long C = x.shape[1].longLongValue;
        const long long D = x.shape[2].longLongValue;
        const long long H = x.shape[3].longLongValue;
        const long long W = x.shape[4].longLongValue;
        const long long Do = out_shape[2];
        const long long Ho = out_shape[3];
        const long long Wo = out_shape[4];
        if (B <= 0 || C <= 0 || D <= 0 || H <= 0 || W <= 0)
            return false;
        if (Do <= 0 || Ho <= 0 || Wo <= 0)
            return false;

        // Depth rides as channels while the plane is resampled.
        MPSGraphTensor* folded = [g reshapeTensor:x
                                        withShape:@[ @(B * C), @(D), @(H), @(W) ]
                                             name:nil];
        if (folded == nil)
            return false;

        MPSGraphTensor* plane = nil;
        const bool align = bool_attr(node, "align_corners", false);
        if (IS_LINEAR) {
            // Same settings as ``interpolate_bilinear``, which is what the
            // depth blend below is derived to compose with.
            plane = [g resizeTensor:folded
                               size:@[ @(Ho), @(Wo) ]
                               mode:MPSGraphResizeBilinear
                       centerResult:YES
                       alignCorners:align ? YES : NO
                             layout:MPSGraphTensorNamedDataLayoutNCHW
                               name:@"interp3d_plane"];
        } else {
            // Floor rounding, matching ``interpolate_nearest_2d`` and the
            // engine's ``floor(o * in / out)``.
            std::int32_t size_data[2] = {(std::int32_t)Ho, (std::int32_t)Wo};
            NSData* size_nsd = [NSData dataWithBytes:size_data length:sizeof(size_data)];
            MPSGraphTensor* size_t = [g constantWithData:size_nsd
                                                   shape:@[ @2 ]
                                                dataType:MPSDataTypeInt32];
            plane = [g resizeNearestWithTensor:folded
                                    sizeTensor:size_t
                           nearestRoundingMode:MPSGraphResizeNearestRoundingModeFloor
                                  centerResult:NO
                                  alignCorners:NO
                                        layout:MPSGraphTensorNamedDataLayoutNCHW
                                          name:@"interp3d_plane"];
        }
        if (plane == nil)
            return false;

        MPSGraphTensor* restored = [g reshapeTensor:plane
                                          withShape:@[ @(B), @(C), @(D), @(Ho), @(Wo) ]
                                               name:nil];
        if (restored == nil)
            return false;

        MPSGraphTensor* y = nil;
        if (Do == D && !IS_LINEAR) {
            y = restored;
        } else if (IS_LINEAR) {
            // Where output plane ``d`` reads from, in input coordinates —
            // the engine's own mapping (CpuBackend's ``src_coord_fn``).
            std::vector<std::int32_t> lower(static_cast<std::size_t>(Do));
            std::vector<std::int32_t> upper(static_cast<std::size_t>(Do));
            std::vector<float> blend(static_cast<std::size_t>(Do));
            for (long long i = 0; i < Do; ++i) {
                double p;
                if (align)
                    p = (Do <= 1) ? 0.0
                                  : static_cast<double>(i) * (D - 1) / static_cast<double>(Do - 1);
                else
                    p = (static_cast<double>(i) + 0.5) * D / static_cast<double>(Do) - 0.5;
                if (p < 0.0)
                    p = 0.0;
                if (p > static_cast<double>(D - 1))
                    p = static_cast<double>(D - 1);
                const long long lo = std::min((long long)p, D - 1);
                lower[(std::size_t)i] = (std::int32_t)lo;
                upper[(std::size_t)i] = (std::int32_t)std::min(lo + 1, D - 1);
                blend[(std::size_t)i] = (float)(p - (double)lo);
            }
            MPSGraphTensor* lo_t = index_constant(g, lower, Do);
            MPSGraphTensor* hi_t = index_constant(g, upper, Do);
            NSData* w_nsd = [NSData dataWithBytes:blend.data() length:blend.size() * sizeof(float)];
            MPSGraphTensor* w_t = [g constantWithData:w_nsd
                                                shape:@[ @1, @1, @(Do), @1, @1 ]
                                             dataType:MPSDataTypeFloat32];
            if (lo_t == nil || hi_t == nil || w_t == nil)
                return false;
            // Autocast can put the chain in half; the weights are written
            // as float32 either way.
            w_t = [g castTensor:w_t toType:restored.dataType name:nil];

            MPSGraphTensor* low = [g gatherWithUpdatesTensor:restored
                                               indicesTensor:lo_t
                                                        axis:2
                                             batchDimensions:0
                                                        name:nil];
            MPSGraphTensor* high = [g gatherWithUpdatesTensor:restored
                                                indicesTensor:hi_t
                                                         axis:2
                                              batchDimensions:0
                                                         name:nil];
            if (low == nil || high == nil)
                return false;
            MPSGraphTensor* diff = [g subtractionWithPrimaryTensor:high
                                                   secondaryTensor:low
                                                              name:nil];
            MPSGraphTensor* scaled = [g multiplicationWithPrimaryTensor:diff
                                                        secondaryTensor:w_t
                                                                   name:nil];
            y = [g additionWithPrimaryTensor:low secondaryTensor:scaled name:@"interp3d_depth"];
        } else {
            // Nearest depth: one source plane per output plane.  A gather
            // rather than a tile, so a non-integer ratio — downsampling
            // included — is the same code path.
            std::vector<std::int32_t> pick(static_cast<std::size_t>(Do));
            for (long long i = 0; i < Do; ++i) {
                long long s = (long long)std::floor(static_cast<double>(i) * D / (double)Do);
                if (s < 0)
                    s = 0;
                if (s > D - 1)
                    s = D - 1;
                pick[(std::size_t)i] = (std::int32_t)s;
            }
            MPSGraphTensor* idx = index_constant(g, pick, Do);
            if (idx == nil)
                return false;
            y = [g gatherWithUpdatesTensor:restored
                             indicesTensor:idx
                                      axis:2
                           batchDimensions:0
                                      name:@"interp3d_depth"];
        }
        if (y == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }

private:
    static MPSGraphTensor*
    index_constant(MPSGraph* g, const std::vector<std::int32_t>& v, long long n) {
        NSData* nsd = [NSData dataWithBytes:v.data() length:v.size() * sizeof(std::int32_t)];
        return [g constantWithData:nsd shape:@[ @(n) ] dataType:MPSDataTypeInt32];
    }

    std::string name_;
};

struct SpatialRegistrar {
    SpatialRegistrar() {
        register_emitter(std::make_unique<AffineGridEmitter>());
        register_emitter(std::make_unique<Interpolate2dEmitterT<true>>("interpolate_bilinear"));
        register_emitter(std::make_unique<Interpolate2dEmitterT<false>>("interpolate_nearest_2d"));
        register_emitter(std::make_unique<Interpolate3dEmitterT<true>>("interpolate_trilinear"));
        register_emitter(std::make_unique<Interpolate3dEmitterT<false>>("interpolate_nearest_3d"));
        register_emitter(std::make_unique<UnfoldDimEmitter>());
    }
};

[[maybe_unused]] static const SpatialRegistrar g_spatial_registrar;

}  // namespace

}  // namespace lucid::compile
