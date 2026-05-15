// lucid/_C/ops/composite/Search.cpp
//
// ``searchsorted`` over a 1-D sorted reference, expressed as a counting
// reduction.  For each query ``v`` we compare against every element of the
// sorted array and count those that are < v (left) or ≤ v (right):
//
//     result[..., j] = sum_i [ sorted[i] < v[..., j] ]
//
// The comparison is materialised by broadcasting ``sorted`` to shape
// ``[1, ..., 1, N]`` and ``values`` to its original shape with a trailing 1.
// The bool result is cast to F32 so the existing reduction kernel (which
// covers F32/F64) can sum it; the count is then cast back to I64.
//
// This avoids needing a dedicated binary-search backend kernel and keeps the
// implementation entirely inside the existing primitive set.

#include "Search.h"

#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/Compare.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Reductions.h"
#include "../utils/Layout.h"
#include "../utils/View.h"

namespace lucid {

namespace {

// Build the per-query count of "sorted entries on the truthy side of v" via
// broadcasting + comparison + reduction.  The output has the same shape as
// ``values`` with dtype I64.
TensorImplPtr
count_compare(const TensorImplPtr& sorted_1d, const TensorImplPtr& values, bool less_or_equal) {
    const auto& vs = values->shape();
    const std::int64_t n = sorted_1d->shape()[0];

    // sorted: [N] → [1, ..., 1, N]   (rank = values.ndim + 1)
    Shape sorted_intermediate(vs.size(), 1);
    sorted_intermediate.push_back(n);
    auto sorted_b = reshape_op(sorted_1d, sorted_intermediate);

    // values: S → S + [1]
    Shape values_intermediate = vs;
    values_intermediate.push_back(1);
    auto values_b = reshape_op(values, values_intermediate);

    // Lucid's compare ops require matching shapes — broadcast both to the
    // full S+[N] grid explicitly before the elementwise comparison.
    Shape full = vs;
    full.push_back(n);
    auto sorted_full = broadcast_to_op(sorted_b, full);
    auto values_full = broadcast_to_op(values_b, full);
    auto cmp =
        less_or_equal ? less_equal_op(sorted_full, values_full) : less_op(sorted_full, values_full);

    // Reduction kernel only supports float types — cast bool → F32 for the
    // count, then back to I64 to match the standard ``searchsorted`` dtype.
    auto cmp_f = astype_op(cmp, Dtype::F32);
    int last_axis = static_cast<int>(full.size()) - 1;
    auto count_f = sum_op(cmp_f, {last_axis}, false);
    return astype_op(count_f, Dtype::I64);
}

}  // namespace

TensorImplPtr
searchsorted_op(const TensorImplPtr& sorted_1d, const TensorImplPtr& values, bool right) {
    if (!sorted_1d || !values)
        ErrorBuilder("searchsorted").fail("null input");
    if (sorted_1d->shape().size() != 1)
        ErrorBuilder("searchsorted").fail("sorted_sequence must be 1-D");
    // For ``right=true`` we want the rightmost insertion point, i.e. how
    // many elements are ≤ v; for ``right=false`` (left) we count strict
    // less-thans.
    return count_compare(sorted_1d, values, /*less_or_equal=*/right);
}

TensorImplPtr
bucketize_op(const TensorImplPtr& values, const TensorImplPtr& boundaries, bool right) {
    return searchsorted_op(boundaries, values, right);
}

}  // namespace lucid
