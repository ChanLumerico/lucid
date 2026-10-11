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

// ``x`` read at ``order`` along ``axis`` — a sort's values.  Bool goes
// through its key, which MPSGraph can gather.
inline MPSGraphTensor*
graph_gather_sorted(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* order, NSInteger axis) {
    const bool is_bool = x.dataType == MPSDataTypeBool;
    MPSGraphTensor* src = is_bool ? [g castTensor:x toType:MPSDataTypeInt32 name:nil] : x;
    MPSGraphTensor* v = [g gatherAlongAxis:axis withUpdatesTensor:src indicesTensor:order name:nil];
    return is_bool && v != nil ? [g castTensor:v toType:MPSDataTypeBool name:nil] : v;
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

inline GraphTopK graph_topk(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, NSInteger k) {
    MPSGraphTensor* order = graph_sort_order(g, x, axis, true);
    MPSGraphTensor* indices = [g sliceTensor:order dimension:axis start:0 length:k name:nil];
    if (indices == nil)
        return {};
    return {graph_gather_sorted(g, x, indices, axis), indices};
}

}  // namespace lucid::compile
