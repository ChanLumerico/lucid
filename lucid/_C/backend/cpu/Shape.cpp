// lucid/_C/backend/cpu/Shape.cpp
//
// Implements the N-D permute_copy operation for any element width.  The
// permuted tensor is the input read through reordered strides, so the copy
// is :func:`strided::pack` of that view: axes merged, the innermost run moved
// as one memcpy when it is contiguous, a merged 2-D transpose copied in
// cache-sized tiles.
//
// It used to rebuild each output element's N-D coordinate with a division
// per axis and copy one element at a time — 4-6 ns an element, single core.
// Every CPU permute is a view in the forward pass, so this copy is what the
// *backward* of permute / transpose / swapaxes / mT costs: splitting a small
// transformer's attention heads, ``(b, t, 3, h, d) -> (3, b, h, t, d)``, took
// 2.5 ms to backpropagate where the forward view costs microseconds — half
// of the whole CPU backward pass.

#include "Shape.h"

#include <cstddef>
#include <vector>

#include "../../core/Shape.h"
#include "../../core/StridedCopy.h"

namespace lucid::backend::cpu {

namespace {

// Computes C-order (row-major) element strides for a given shape.
// stride[i] = product of shape[i+1..ndim-1], with the last stride equal to 1.
std::vector<std::int64_t> elem_strides(const std::vector<std::int64_t>& shape) {
    std::vector<std::int64_t> s(shape.size());
    if (shape.empty())
        return s;
    std::int64_t acc = 1;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

}  // namespace

// The output is the input viewed with its axes in ``perm`` order (shape
// ``in_shape[perm[d]]``, stride ``in_strides[perm[d]]``), packed dense and
// row-major.  Bitwise a copy, so the result is the same as the element walk
// it replaces for every dtype, NaN payload included.
void permute_copy(const std::byte* in,
                  std::byte* out,
                  const std::vector<std::int64_t>& in_shape,
                  const std::vector<int>& perm,
                  std::size_t elem) {
    const std::size_t ndim = in_shape.size();
    const auto in_strides = elem_strides(in_shape);
    Shape view_shape(ndim);
    Stride view_stride(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        const auto axis = static_cast<std::size_t>(perm[d]);
        view_shape[d] = in_shape[axis];
        view_stride[d] = in_strides[axis] * static_cast<std::int64_t>(elem);
    }
    if (shape_numel(view_shape) == 0)
        return;
    strided::pack(in, out, view_shape, view_stride, elem);
}

}  // namespace lucid::backend::cpu
