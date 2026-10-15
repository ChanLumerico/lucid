// lucid/_C/compile/OpEmitters/_SortOrder.h
//
// The order ``sort``, ``argsort``, ``topk``, ``argmax`` and ``argmin`` read
// a tensor in inside a compiled graph — the compiled side of
// ``backend/gpu/SortOrder.h``, so the two name the same positions.
//
// Eager orders stably: equal values keep their index order, so ``topk``
// names ties lowest index first (LCD-407), and NaN sorts as the largest
// value.  MPSGraph's ``topK`` does neither: ``topk([3, 3, 3, 3], 2)`` named
// 3 and 1, a NaN came back as 0.0 at a repeated index, and an int64 source
// was ranked through a narrower type, so 2^40 + 3 lost to 2^40 (LCD-425).
// Its ``argSort`` is stable in both directions — descending lists equal
// values lowest index first and NaN first, ascending NaN last — and exact on
// int64 (measured on macOS 26 up to 2^20 elements per axis; the parity tests
// in ``test_sort_order_compiled.py`` hold it to that).  So every ordering op
// sorts with ``argSort`` and gathers its values by the order.
//
// Two inputs still differ from eager before the sort, and the key removes
// them: MPSGraph puts -0.0 below +0.0 where eager calls them equal, and it
// has no bool sort.  The values a caller gathers come from the original
// tensor, so -0.0 keeps its sign.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "_DataMovement.h"

namespace lucid::compile {

// ``x`` as the key the order is taken on: zeros of either sign made +0.0 and
// bool widened to int32 (false < true, as eager orders it).  Half precision
// is widened to float32 first — exact, so the order is the same — because
// MPSGraph's compare-to-zero pass rejects a float16 ``equal`` with a warning
// on every compile ("ConvertBinaryCompareToZero expects the second operand
// to be zero").
inline MPSGraphTensor* graph_sort_key(MPSGraph* g, MPSGraphTensor* x) {
    MPSDataType dt = x.dataType;
    if (dt == MPSDataTypeBool)
        return [g castTensor:x toType:MPSDataTypeInt32 name:nil];
    if (dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16) {
        x = [g castTensor:x toType:MPSDataTypeFloat32 name:nil];
        dt = MPSDataTypeFloat32;
    }
    if (dt != MPSDataTypeFloat32)
        return x;
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
    MPSGraphTensor* is_zero = [g equalWithPrimaryTensor:x secondaryTensor:zero name:nil];
    return [g selectWithPredicateTensor:is_zero
                    truePredicateTensor:zero
                   falsePredicateTensor:x
                                   name:nil];
}

// The positions that order ``x`` along ``axis`` (int32), equal values in
// index order: ascending puts NaN last, descending puts it first.
inline MPSGraphTensor*
graph_sort_order(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, bool descending) {
    return [g argSortWithTensor:graph_sort_key(g, x)
                           axis:axis
                     descending:descending ? YES : NO
                           name:nil];
}

// ``x`` read at ``order`` along ``axis`` — a sort's values, every bit of
// them: MPSGraph's own gather rounded int64 through float32, so 2^40 + 3
// came back as 2^40 (LCD-435, ``_DataMovement.h``).
inline MPSGraphTensor*
graph_gather_sorted(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* order, NSInteger axis) {
    return graph_gather_along_axis(g, axis, x, order);
}

// ``argmax`` / ``argmin`` along ``axis`` (kept, length one): the first
// extreme position, and the first NaN where the slice holds one — NaN is
// both the largest and the smallest value to eager, as to the reference.
// MPSGraph's arg reductions skip NaN, so ``argmax([2, nan, 2, nan])`` named
// 0 where eager names 1.
inline MPSGraphTensor*
graph_arg_extreme(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, bool is_max) {
    MPSGraphTensor* r = is_max ? [g reductionArgMaximumWithTensor:x axis:axis name:nil]
                               : [g reductionArgMinimumWithTensor:x axis:axis name:nil];
    const MPSDataType dt = x.dataType;
    if (r == nil ||
        (dt != MPSDataTypeFloat32 && dt != MPSDataTypeFloat16 && dt != MPSDataTypeBFloat16))
        return r;
    MPSGraphTensor* nan = [g castTensor:[g isNaNWithTensor:x name:nil]
                                 toType:MPSDataTypeInt32
                                   name:nil];
    MPSGraphTensor* first_nan = [g reductionArgMaximumWithTensor:nan axis:axis name:nil];
    MPSGraphTensor* any_nan =
        [g greaterThanWithPrimaryTensor:[g reductionMaximumWithTensor:nan axis:axis name:nil]
                        secondaryTensor:[g constantWithScalar:0 dataType:MPSDataTypeInt32]
                                   name:nil];
    first_nan = [g castTensor:first_nan toType:r.dataType name:nil];
    return [g selectWithPredicateTensor:any_nan
                    truePredicateTensor:first_nan
                   falsePredicateTensor:r
                                   name:nil];
}

// ``topk``: the positions of the ``k`` largest values along ``axis``, largest
// first and equal values lowest index first, and the values at them.
// ``k`` must not exceed the axis.
struct GraphTopK {
    MPSGraphTensor* values = nil;
    MPSGraphTensor* indices = nil;  // int32
};

// Order-preserving float32 key with no NaNs:
//   -0.0  -> +0.0
//   +inf  -> FLT_MAX
//   NaN   -> +inf (so MPSGraph's topK ranks NaN above +inf without dropping it)
inline MPSGraphTensor* graph_topk_f32_key(MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* xf = graph_sort_key(g, x);
    if (xf.dataType != MPSDataTypeFloat32)
        xf = [g castTensor:xf toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* pos_inf = [g constantWithScalar:std::numeric_limits<double>::infinity()
                                           dataType:MPSDataTypeFloat32];
    MPSGraphTensor* flt_max =
        [g constantWithScalar:static_cast<double>(std::numeric_limits<float>::max())
                     dataType:MPSDataTypeFloat32];
    MPSGraphTensor* step1 = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:xf
                                                                   secondaryTensor:pos_inf
                                                                              name:nil]
                                     truePredicateTensor:flt_max
                                    falsePredicateTensor:xf
                                                    name:nil];
    return [g selectWithPredicateTensor:[g isNaNWithTensor:xf name:nil]
                    truePredicateTensor:pos_inf
                   falsePredicateTensor:step1
                                   name:nil];
}

inline GraphTopK graph_topk_full_sort(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, NSInteger k) {
    MPSGraphTensor* order = graph_sort_order(g, x, axis, true);
    MPSGraphTensor* indices = [g sliceTensor:order dimension:axis start:0 length:k name:nil];
    if (indices == nil)
        return {};
    return {graph_gather_sorted(g, x, indices, axis), indices};
}

inline GraphTopK graph_topk_small_k(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* key,
                                    NSInteger axis, NSInteger k) {
    if (k == 1) {
        MPSGraphTensor* indices = [g reductionArgMaximumWithTensor:key axis:axis name:nil];
        if (indices == nil)
            return graph_topk_full_sort(g, x, axis, k);
        return {graph_gather_sorted(g, x, indices, axis), indices};
    }
    MPSGraphTensor* pos = [g coordinateAlongAxis:axis withShape:x.shape name:nil];
    MPSGraphTensor* neg_inf = [g constantWithScalar:-std::numeric_limits<double>::infinity()
                                           dataType:MPSDataTypeFloat32];
    MPSGraphTensor* cur = key;
    NSMutableArray<MPSGraphTensor*>* picked = [NSMutableArray arrayWithCapacity:(NSUInteger)k];
    for (NSInteger s = 0; s < k; ++s) {
        MPSGraphTensor* idx_s = [g reductionArgMaximumWithTensor:cur axis:axis name:nil];
        if (idx_s == nil)
            return graph_topk_full_sort(g, x, axis, k);
        [picked addObject:idx_s];
        if (s + 1 < k) {
            MPSGraphTensor* mask = [g equalWithPrimaryTensor:pos secondaryTensor:idx_s name:nil];
            cur = [g selectWithPredicateTensor:mask
                           truePredicateTensor:neg_inf
                          falsePredicateTensor:cur
                                          name:nil];
        }
    }
    MPSGraphTensor* indices = [g concatTensors:picked dimension:axis name:nil];
    return {graph_gather_sorted(g, x, indices, axis), indices};
}

inline GraphTopK graph_topk_two_pass(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* key,
                                     NSInteger axis, NSInteger axis_len, NSInteger k) {
    NSArray<MPSGraphTensor*>* tk1 = [g topKWithSourceTensor:key axis:axis k:(NSUInteger)k name:nil];
    if (tk1 == nil || tk1.count < 2 || tk1[0] == nil)
        return graph_topk_full_sort(g, x, axis, k);

    // ``v_last`` is the k-th largest key value. Elements strictly above it
    // all belong in the top-k (score = axis_len - index >= 1); elements equal
    // to ``v_last`` get score = -index (0, -1, ..., -(N-1)) so a second topK
    // picks the lowest indices among boundary ties; elements below get -(N+1).
    MPSGraphTensor* v_last = [g sliceTensor:tk1[0] dimension:axis start:k - 1 length:1 name:nil];
    MPSGraphTensor* neg_pos = [g negativeWithTensor:[g castTensor:[g coordinateAlongAxis:axis
                                                                               withShape:x.shape
                                                                                    name:nil]
                                                           toType:MPSDataTypeFloat32
                                                             name:nil]
                                               name:nil];
    MPSGraphTensor* above = [g greaterThanWithPrimaryTensor:key secondaryTensor:v_last name:nil];
    MPSGraphTensor* tied = [g equalWithPrimaryTensor:key secondaryTensor:v_last name:nil];
    MPSGraphTensor* hi_score =
        [g additionWithPrimaryTensor:neg_pos
                     secondaryTensor:[g constantWithScalar:(double)axis_len
                                                  dataType:MPSDataTypeFloat32]
                                name:nil];
    MPSGraphTensor* lo_score = [g constantWithScalar:-(double)axis_len - 1.0
                                            dataType:MPSDataTypeFloat32];
    MPSGraphTensor* tie_key = [g selectWithPredicateTensor:above
                                       truePredicateTensor:hi_score
                                      falsePredicateTensor:[g selectWithPredicateTensor:tied
                                                                    truePredicateTensor:neg_pos
                                                                   falsePredicateTensor:lo_score
                                                                                   name:nil]
                                                      name:nil];

    NSArray<MPSGraphTensor*>* tk2 = [g topKWithSourceTensor:tie_key
                                                       axis:axis
                                                          k:(NSUInteger)k
                                                       name:nil];
    if (tk2 == nil || tk2.count < 2 || tk2[1] == nil)
        return graph_topk_full_sort(g, x, axis, k);

    // Order the k candidate indices ascending first, then stably sort their k
    // keys descending (both sorts are of length k, not axis_len).
    MPSGraphTensor* cand_idx = tk2[1];
    MPSGraphTensor* asc_order = [g argSortWithTensor:cand_idx axis:axis descending:NO name:nil];
    MPSGraphTensor* sorted_idx = graph_gather_along_axis(g, axis, cand_idx, asc_order);
    MPSGraphTensor* cand_key = graph_gather_along_axis(g, axis, key, sorted_idx);
    MPSGraphTensor* key_order = [g argSortWithTensor:cand_key axis:axis descending:YES name:nil];
    MPSGraphTensor* indices = graph_gather_along_axis(g, axis, sorted_idx, key_order);
    if (indices == nil)
        return {};
    return {graph_gather_sorted(g, x, indices, axis), indices};
}

inline bool graph_topk_can_use_fast_path(MPSGraphTensor* x, NSInteger axis, NSInteger k) {
    const MPSDataType dt = x.dataType;
    if (dt == MPSDataTypeInt64 || dt == MPSDataTypeUInt64 || dt == MPSDataTypeBool)
        return false;
    const NSInteger axis_len = [x.shape[(NSUInteger)axis] integerValue];
    if (axis_len <= 0 || k >= axis_len)
        return false;
    for (NSNumber* dim in x.shape) {
        if (dim.integerValue <= 0)
            return false;
    }
    return true;
}

inline GraphTopK graph_topk(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, NSInteger k) {
    if (g == nil || x == nil || x.shape == nil)
        return {};
    const NSInteger rank = (NSInteger)x.shape.count;
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank || k <= 0)
        return {};
    if (!graph_topk_can_use_fast_path(x, axis, k))
        return graph_topk_full_sort(g, x, axis, k);

    MPSGraphTensor* key = graph_topk_f32_key(g, x);
    if (k <= 8)
        return graph_topk_small_k(g, x, key, axis, k);
    return graph_topk_two_pass(g, x, key, axis, [x.shape[(NSUInteger)axis] integerValue], k);
}

}  // namespace lucid::compile
