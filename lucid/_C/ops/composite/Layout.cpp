// lucid/_C/ops/composite/Layout.cpp
//
// ``movedim`` rebuilds a permutation: every axis listed in ``source`` is
// placed at the matching index in ``destination``; unmoved axes preserve
// their order and fill the gaps.  ``unflatten`` is a thin wrapper around
// ``reshape`` after validating the size product.

#include "Layout.h"

#include <algorithm>
#include <cstdint>

#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../ufunc/Transpose.h"
#include "../utils/View.h"

namespace lucid {

TensorImplPtr movedim_op(const TensorImplPtr& a,
                         const std::vector<int>& source,
                         const std::vector<int>& destination) {
    if (!a)
        ErrorBuilder("movedim").fail("null input");
    const int ndim = static_cast<int>(a->shape().size());
    if (source.size() != destination.size())
        ErrorBuilder("movedim").fail("source and destination size mismatch");

    auto wrap = [&](int v) {
        int w = v < 0 ? v + ndim : v;
        if (w < 0 || w >= ndim)
            ErrorBuilder("movedim").index_error("dim out of range");
        return w;
    };
    std::vector<int> src;
    std::vector<int> dst;
    src.reserve(source.size());
    dst.reserve(destination.size());
    for (int s : source)
        src.push_back(wrap(s));
    for (int d : destination)
        dst.push_back(wrap(d));

    // Reject duplicates — they would make the permutation ill-defined.
    std::vector<int> sorted_src = src;
    std::sort(sorted_src.begin(), sorted_src.end());
    if (std::adjacent_find(sorted_src.begin(), sorted_src.end()) != sorted_src.end())
        ErrorBuilder("movedim").fail("source dims must be unique");
    std::vector<int> sorted_dst = dst;
    std::sort(sorted_dst.begin(), sorted_dst.end());
    if (std::adjacent_find(sorted_dst.begin(), sorted_dst.end()) != sorted_dst.end())
        ErrorBuilder("movedim").fail("destination dims must be unique");

    // Axes not mentioned in ``source`` keep their relative order; they fill
    // the destination slots not claimed by ``dst``.
    std::vector<bool> moved(static_cast<std::size_t>(ndim), false);
    for (int s : src)
        moved[static_cast<std::size_t>(s)] = true;
    std::vector<int> rest;
    for (int i = 0; i < ndim; ++i)
        if (!moved[static_cast<std::size_t>(i)])
            rest.push_back(i);

    std::vector<int> perm(static_cast<std::size_t>(ndim), -1);
    for (std::size_t i = 0; i < src.size(); ++i)
        perm[static_cast<std::size_t>(dst[i])] = src[i];
    auto rest_it = rest.begin();
    for (int i = 0; i < ndim; ++i) {
        if (perm[static_cast<std::size_t>(i)] == -1)
            perm[static_cast<std::size_t>(i)] = *rest_it++;
    }
    return permute_op(a, perm);
}

TensorImplPtr unflatten_op(const TensorImplPtr& a, int dim, const Shape& sizes) {
    if (!a)
        ErrorBuilder("unflatten").fail("null input");
    const int ndim = static_cast<int>(a->shape().size());
    int d = dim < 0 ? dim + ndim : dim;
    if (d < 0 || d >= ndim)
        ErrorBuilder("unflatten").index_error("dim out of range");

    std::int64_t product = 1;
    for (auto s : sizes)
        product *= s;
    if (product != a->shape()[static_cast<std::size_t>(d)])
        ErrorBuilder("unflatten").fail("sizes do not multiply to dim length");

    Shape new_shape;
    new_shape.reserve(static_cast<std::size_t>(ndim) - 1 + sizes.size());
    for (int i = 0; i < d; ++i)
        new_shape.push_back(a->shape()[static_cast<std::size_t>(i)]);
    for (auto s : sizes)
        new_shape.push_back(s);
    for (int i = d + 1; i < ndim; ++i)
        new_shape.push_back(a->shape()[static_cast<std::size_t>(i)]);
    return reshape_op(a, new_shape);
}

}  // namespace lucid
