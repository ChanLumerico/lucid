// lucid/_C/compile/OpEmitters/_IndexBounds.h
//
// Where a caller's index values meet an MPSGraph gather or scatter inside a
// compiled graph, and the one place that keeps them inside the buffer.
//
// The compiled graph is a second implementation of the eager index ops, so it
// follows the eager Metal contract, policy B (LCD-228,
// ``backend/gpu/AxisIndex.h``):
//
//   * a gather answers NaN at an out-of-range position, or 0 when the result
//     is an integer or bool;
//   * a scatter drops an out-of-range update, and the base keeps its value;
//   * no index outside the axis reaches an MPSGraph gather or scatter.
//
// The graph is built from the first call's indices, and every later call
// feeds new ones that nothing on the host reads, so the check has to be in
// the graph.  Before LCD-278 the emitters handed the indices straight to
// MPSGraph, which does not compare them with the axis either.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace lucid::compile {

// What an index below zero means.
enum class NegativeIndex {
    Wrap,        // ``-k`` names ``extent - k``, once: gather, scatter
    OutOfRange,  // nothing lies below zero: a table row, a class
};

// An index that is safe to hand to an MPSGraph gather or scatter.  Both nil
// when the index cannot be checked (not an integer, or the axis length is not
// known when the graph is built), and the emitter declines to eager then.
struct GraphAxisIndex {
    // In [0, extent) everywhere, and the index itself where it is in range.
    MPSGraphTensor* safe = nil;
    // Bool, the index's shape: whether the index named a real position.
    MPSGraphTensor* in_range = nil;
};

// ``idx`` checked against an axis of ``extent`` positions, at a width that
// holds every index value and ``extent``: int64 stays int64 (narrowing first
// is how 2^32 + 1 read row 1), anything narrower is widened to int32.  The
// safe index comes back as int32 whenever ``extent`` fits, as eager's does.
inline GraphAxisIndex
graph_axis_index(MPSGraph* g, MPSGraphTensor* idx, std::int64_t extent, NegativeIndex negative) {
    const MPSDataType dt = idx.dataType;
    const bool integer = dt == MPSDataTypeInt8 || dt == MPSDataTypeInt16 ||
                         dt == MPSDataTypeInt32 || dt == MPSDataTypeInt64 || dt == MPSDataTypeBool;
    if (!integer || extent <= 0)
        return {};
    const bool fits32 = extent <= std::numeric_limits<std::int32_t>::max();
    const MPSDataType work =
        (dt == MPSDataTypeInt64 || !fits32) ? MPSDataTypeInt64 : MPSDataTypeInt32;
    MPSGraphTensor* i = dt == work ? idx : [g castTensor:idx toType:work name:nil];
    MPSGraphTensor* n = [g constantWithScalar:static_cast<double>(extent) dataType:work];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:work];
    MPSGraphTensor* negative_side = [g lessThanWithPrimaryTensor:i secondaryTensor:zero name:nil];
    MPSGraphTensor* lowest =
        negative == NegativeIndex::Wrap ? [g negativeWithTensor:n name:nil] : zero;
    MPSGraphTensor* in_range =
        [g logicalANDWithPrimaryTensor:[g greaterThanOrEqualToWithPrimaryTensor:i
                                                                secondaryTensor:lowest
                                                                           name:nil]
                       secondaryTensor:[g lessThanWithPrimaryTensor:i secondaryTensor:n name:nil]
                                  name:@"index_in_range"];
    MPSGraphTensor* wrapped = negative == NegativeIndex::Wrap
                                  ? [g selectWithPredicateTensor:negative_side
                                             truePredicateTensor:[g additionWithPrimaryTensor:i
                                                                              secondaryTensor:n
                                                                                         name:nil]
                                            falsePredicateTensor:i
                                                            name:nil]
                                  : i;
    MPSGraphTensor* safe = [g selectWithPredicateTensor:in_range
                                    truePredicateTensor:wrapped
                                   falsePredicateTensor:zero
                                                   name:@"index_safe"];
    if (fits32 && work != MPSDataTypeInt32)
        safe = [g castTensor:safe toType:MPSDataTypeInt32 name:nil];
    return {safe, in_range};
}

// ``gathered`` with every out-of-range position replaced by what policy B
// answers there: NaN for a floating result, 0 for an integer or bool one.
// ``in_range`` broadcasts against ``gathered``.  nil for a complex result,
// whose NaN this layer does not build; the emitter declines then.
inline MPSGraphTensor*
graph_fill_out_of_range(MPSGraph* g, MPSGraphTensor* gathered, MPSGraphTensor* in_range) {
    const MPSDataType dt = gathered.dataType;
    if (dt & MPSDataTypeComplexBit)
        return nil;
    const double fill = (dt & MPSDataTypeFloatBit) ? std::nan("") : 0.0;
    return [g selectWithPredicateTensor:in_range
                    truePredicateTensor:gathered
                   falsePredicateTensor:[g constantWithScalar:fill dataType:dt]
                                   name:@"index_out_of_range_fill"];
}

// The reductions an MPSGraph scatter applies.
enum class GraphScatter { Add, Max, Min, Mul };

// The update that leaves a held value as it was, which is what an
// out-of-range update becomes before it is scattered at the safe index.
// For a sum that is -0.0, not +0.0: ``x + -0.0`` is ``x`` for every ``x``,
// -0.0 included.  nil when the dtype has no such value here (complex).
inline MPSGraphTensor* graph_scatter_identity(MPSGraph* g, GraphScatter op, MPSDataType dt) {
    if (dt & MPSDataTypeComplexBit)
        return nil;
    const bool floating = (dt & MPSDataTypeFloatBit) != 0;
    double v = 0.0;
    switch (op) {
    case GraphScatter::Add:
        v = floating ? -0.0 : 0.0;
        break;
    case GraphScatter::Mul:
        v = 1.0;
        break;
    case GraphScatter::Max:
    case GraphScatter::Min: {
        const bool lowest = op == GraphScatter::Max;
        if (floating) {
            v = lowest ? -INFINITY : INFINITY;
        } else if (dt == MPSDataTypeInt32) {
            v = lowest ? std::numeric_limits<std::int32_t>::lowest()
                       : std::numeric_limits<std::int32_t>::max();
        } else if (dt == MPSDataTypeInt16) {
            v = lowest ? std::numeric_limits<std::int16_t>::lowest()
                       : std::numeric_limits<std::int16_t>::max();
        } else if (dt == MPSDataTypeInt8) {
            v = lowest ? std::numeric_limits<std::int8_t>::lowest()
                       : std::numeric_limits<std::int8_t>::max();
        } else {
            return nil;  // int64's bounds do not survive the double constant
        }
        break;
    }
    }
    return [g constantWithScalar:v dataType:dt];
}

// ``updates`` with each out-of-range entry replaced by ``op``'s identity, so
// a reducing scatter at the safe index leaves that position as it was.  nil
// when ``op`` has no identity for the dtype.
inline MPSGraphTensor* graph_drop_out_of_range(MPSGraph* g,
                                               MPSGraphTensor* updates,
                                               MPSGraphTensor* in_range,
                                               GraphScatter op) {
    MPSGraphTensor* identity = graph_scatter_identity(g, op, updates.dataType);
    if (identity == nil)
        return nil;
    return [g selectWithPredicateTensor:in_range
                    truePredicateTensor:updates
                   falsePredicateTensor:identity
                                   name:@"index_out_of_range_drop"];
}

// An overwrite scatter under policy B.  No in-range position can take a
// write and keep its value, so an out-of-range write is aimed at a one-wide
// sink slab appended past the end of ``axis`` and cut off again — eager's
// ``gpu_sink_index`` / ``with_sink_slab``.
inline MPSGraphTensor* graph_scatter_set_dropping(MPSGraph* g,
                                                  MPSGraphTensor* base,
                                                  const GraphAxisIndex& ix,
                                                  MPSGraphTensor* src,
                                                  NSInteger axis,
                                                  std::int64_t extent) {
    NSMutableArray<NSNumber*>* slab_shape = [base.shape mutableCopy];
    slab_shape[static_cast<NSUInteger>(axis)] = @1;
    MPSGraphTensor* padded = [g concatTensor:base
                                  withTensor:[g constantWithScalar:0.0
                                                             shape:slab_shape
                                                          dataType:base.dataType]
                                   dimension:axis
                                        name:nil];
    MPSGraphTensor* past_end = [g constantWithScalar:static_cast<double>(extent)
                                            dataType:ix.safe.dataType];
    MPSGraphTensor* sink = [g selectWithPredicateTensor:ix.in_range
                                    truePredicateTensor:ix.safe
                                   falsePredicateTensor:past_end
                                                   name:@"scatter_sink_index"];
    MPSGraphTensor* written = [g scatterAlongAxis:axis
                                   withDataTensor:padded
                                    updatesTensor:src
                                    indicesTensor:sink
                                             mode:MPSGraphScatterModeSet
                                             name:@"scatter_axis"];
    return [g sliceTensor:written dimension:axis start:0 length:extent name:@"scatter_cut_sink"];
}

// MPSGraph's along-axis gather and scatter want the index to match the data
// on every axis but ``skip_axis`` (pass -1 for none).  The reference — and
// Lucid's eager kernels — allow an index shorter there, and MPSGraph handed
// one aborted the process: "updates shape and indices shape must match
// except at axis".  Such a call is declined and runs eagerly.
inline bool shapes_agree_off_axis(MPSGraphTensor* a, MPSGraphTensor* b, NSInteger skip_axis) {
    NSArray<NSNumber*>* sa = a.shape;
    NSArray<NSNumber*>* sb = b.shape;
    if (sa == nil || sb == nil || sa.count != sb.count)
        return false;
    for (NSUInteger i = 0; i < sa.count; ++i) {
        if (static_cast<NSInteger>(i) == skip_axis)
            continue;
        if (sa[i].longLongValue != sb[i].longLongValue)
            return false;
    }
    return true;
}

// The static length of ``t`` along ``axis``, or -1 when it is not known when
// the graph is built (the symbolic batch).
inline std::int64_t static_extent(MPSGraphTensor* t, NSInteger axis) {
    NSArray<NSNumber*>* s = t.shape;
    if (s == nil || axis < 0 || static_cast<NSUInteger>(axis) >= s.count)
        return -1;
    return s[static_cast<NSUInteger>(axis)].longLongValue;
}

}  // namespace lucid::compile
