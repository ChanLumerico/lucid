// lucid/_C/compile/OpEmitters/_Resample.h
//
// ``interpolate`` inside a compiled graph, built step for step as the eager
// Metal kernel builds it (``backend/gpu/Interpolate.h``), so the two read
// the same samples with the same weights.
//
// The emitters used MPSGraph's resize kernels, whose sample positions are
// not the shared rule in ``backend/SamplePositions.h``: compiled against
// eager on Metal, nearest 2155 -> 5387 read 1 sample of a row elsewhere
// (91 of a 37x2155 -> 91x5387 plane), bilinear blended up to 1e-3 off, and
// half precision rounded at every step (LCD-425).  Here every position and
// weight comes from that rule on the host, as constants, and each resampled
// axis is applied in turn, the last first — a gather and, for the linear
// modes, a blend of both taps (a zero weight included, so an infinite
// neighbour gives NaN as it does in eager).  Float16 and bfloat16 blend in
// float32 and round once.  Backward scatters through the same tables in the
// eager adjoint's axis order.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../backend/SamplePositions.h"
#include "_DataMovement.h"

namespace lucid::compile {

enum class ResampleFilter { Nearest, Linear };

namespace resample_detail {

inline MPSGraphTensor* int32_constant(MPSGraph* g, const std::vector<std::int32_t>& v) {
    NSData* d = [NSData dataWithBytes:v.data() length:v.size() * sizeof(std::int32_t)];
    return [g constantWithData:d shape:@[ @(v.size()) ] dataType:MPSDataTypeInt32];
}

// ``v`` laid along ``axis`` of an ``ndim``-axis tensor, length one
// elsewhere, in ``dt``.
inline MPSGraphTensor*
along(MPSGraph* g, const std::vector<float>& v, NSUInteger axis, NSUInteger ndim, MPSDataType dt) {
    NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
    for (NSUInteger i = 0; i < ndim; ++i)
        [shape addObject:@(i == axis ? static_cast<long long>(v.size()) : 1LL)];
    NSData* d = [NSData dataWithBytes:v.data() length:v.size() * sizeof(float)];
    MPSGraphTensor* t = [g constantWithData:d shape:shape dataType:MPSDataTypeFloat32];
    return dt == MPSDataTypeFloat32 ? t : [g castTensor:t toType:dt name:nil];
}

inline MPSGraphTensor* take(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* idx, NSUInteger axis) {
    return graph_gather(g, x, idx, axis);
}

// ``x`` summed onto ``in`` positions along ``axis``: position ``d`` goes to
// ``idx[d]``.
inline MPSGraphTensor* scatter_back(
    MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* idx, NSUInteger axis, std::int64_t in) {
    NSMutableArray<NSNumber*>* shape = [x.shape mutableCopy];
    shape[axis] = @(in);
    return [g scatterWithUpdatesTensor:x
                         indicesTensor:idx
                                 shape:shape
                                  axis:static_cast<NSInteger>(axis)
                                  mode:MPSGraphScatterModeAdd
                                  name:nil];
}

// One axis of ``x`` resampled from ``in`` to ``out`` samples — or,
// ``backward``, a gradient of that taken back to ``in``.
inline MPSGraphTensor* resample_axis(MPSGraph* g,
                                     MPSGraphTensor* x,
                                     ResampleFilter filter,
                                     NSUInteger axis,
                                     std::int64_t in,
                                     std::int64_t out,
                                     bool align_corners,
                                     bool backward) {
    namespace rs = ::lucid::backend::resample;
    if (filter == ResampleFilter::Nearest) {
        const auto sources = rs::nearest_axis<float>(in, out);
        MPSGraphTensor* idx =
            int32_constant(g, std::vector<std::int32_t>(sources.begin(), sources.end()));
        return backward ? scatter_back(g, x, idx, axis, in) : take(g, x, idx, axis);
    }
    std::vector<std::int32_t> lo, hi;
    std::vector<float> keep, frac;
    for (const auto& t : rs::linear_axis<float>(in, out, align_corners)) {
        lo.push_back(static_cast<std::int32_t>(t.lo));
        hi.push_back(static_cast<std::int32_t>(t.hi));
        keep.push_back(1.0f - t.frac);
        frac.push_back(t.frac);
    }
    const NSUInteger ndim = x.shape.count;
    MPSGraphTensor* lo_t = int32_constant(g, lo);
    MPSGraphTensor* hi_t = int32_constant(g, hi);
    MPSGraphTensor* keep_t = along(g, keep, axis, ndim, x.dataType);
    MPSGraphTensor* frac_t = along(g, frac, axis, ndim, x.dataType);
    const auto mul = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
        return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    };
    if (backward)
        return [g additionWithPrimaryTensor:scatter_back(g, mul(x, keep_t), lo_t, axis, in)
                            secondaryTensor:scatter_back(g, mul(x, frac_t), hi_t, axis, in)
                                       name:nil];
    return [g additionWithPrimaryTensor:mul(take(g, x, lo_t, axis), keep_t)
                        secondaryTensor:mul(take(g, x, hi_t, axis), frac_t)
                                   name:nil];
}

}  // namespace resample_detail

// The ``rank`` spatial extents a resize of ``x`` reads (``in``) and writes
// (``out``, the recorded output's).  False — decline — unless ``x`` is
// ``(N, C, *spatial)`` with every extent known and positive.
inline bool graph_resample_extents(MPSGraphTensor* x,
                                   const std::vector<std::int64_t>& recorded_out,
                                   std::size_t rank,
                                   std::vector<std::int64_t>& in,
                                   std::vector<std::int64_t>& out) {
    if (x == nil || x.shape.count != rank + 2 || recorded_out.size() != rank + 2)
        return false;
    in.clear();
    out.clear();
    for (std::size_t j = 0; j < rank; ++j) {
        const long long i = x.shape[j + 2].longLongValue;
        const std::int64_t o = recorded_out[j + 2];
        if (i <= 0 || o <= 0)
            return false;
        in.push_back(i);
        out.push_back(o);
    }
    return true;
}

// ``x`` of ``(N, C, *in)`` resized to ``(N, C, *out)`` — or, ``backward``,
// a gradient of ``(N, C, *out)`` taken back to ``(N, C, *in)``.  ``in`` is
// the forward input's spatial extents either way.  Nil when a step fails;
// the caller declines then.
inline MPSGraphTensor* graph_resample(MPSGraph* g,
                                      MPSGraphTensor* x,
                                      const std::vector<std::int64_t>& in,
                                      const std::vector<std::int64_t>& out,
                                      ResampleFilter filter,
                                      bool align_corners,
                                      bool backward) {
    if (x == nil || in.size() != out.size() || x.shape.count < in.size())
        return nil;
    const MPSDataType dt = x.dataType;
    // A copy needs no widening; a blend or a sum does.
    const bool widen = (dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16) &&
                       (filter == ResampleFilter::Linear || backward);
    if (widen)
        x = [g castTensor:x toType:MPSDataTypeFloat32 name:nil];
    const std::size_t rank = out.size();
    const NSUInteger first = x.shape.count - rank;
    for (std::size_t k = rank; k-- > 0;) {
        const std::size_t j = backward ? rank - 1 - k : k;
        x = resample_detail::resample_axis(g, x, filter, first + j, in[j], out[j], align_corners,
                                           backward);
        if (x == nil)
            return nil;
    }
    return widen ? [g castTensor:x toType:dt name:nil] : x;
}

}  // namespace lucid::compile
