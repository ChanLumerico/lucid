// lucid/_C/backend/cpu/Shape.cpp
//
// Implements the N-D permute_copy operation for f32, f64, i32, and i64 types.
// The generic permute_typed template uses two auxiliary stride vectors:
//   in_strides  — C-order strides of the input layout
//   out_strides — C-order strides of the output layout (permuted shape)
// For each output flat index it back-computes the N-D coordinate in the output
// shape, maps that coordinate to an input flat index via in_strides[perm[d]],
// and copies the single element.  This is O(numel * ndim) but avoids any
// additional memory allocation beyond the two stride vectors.

#include "Shape.h"

#include <cstddef>
#include <vector>

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

// Generic N-D permutation: iterates over all output positions in flat order,
// maps each back to an N-D coordinate, then reads from the corresponding
// input position by applying the inverse permutation through in_strides.
template <typename T>
void permute_typed(const T* in,
                   T* out,
                   const std::vector<std::int64_t>& in_shape,
                   const std::vector<int>& perm) {
    const std::size_t ndim = in_shape.size();
    if (ndim == 0) {
        out[0] = in[0];
        return;
    }

    std::vector<std::int64_t> out_shape(ndim);
    for (std::size_t d = 0; d < ndim; ++d)
        out_shape[d] = in_shape[static_cast<std::size_t>(perm[d])];

    const auto in_strides = elem_strides(in_shape);
    const auto out_strides = elem_strides(out_shape);

    std::size_t numel = 1;
    for (auto d : out_shape)
        numel *= static_cast<std::size_t>(d);
    if (numel == 0)
        return;

    std::vector<std::int64_t> out_idx(ndim);
    for (std::size_t flat = 0; flat < numel; ++flat) {
        std::size_t rem = flat;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::size_t s = static_cast<std::size_t>(out_strides[d]);
            const std::size_t k = (s == 0) ? 0 : (rem / s);
            out_idx[d] = static_cast<std::int64_t>(k);
            if (s)
                rem -= k * s;
        }

        std::int64_t in_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            in_flat += out_idx[d] * in_strides[static_cast<std::size_t>(perm[d])];
        }
        out[flat] = in[in_flat];
    }
}

}  // namespace

void permute_copy_f32(const float* in,
                      float* out,
                      const std::vector<std::int64_t>& in_shape,
                      const std::vector<int>& perm) {
    permute_typed<float>(in, out, in_shape, perm);
}

void permute_copy_f64(const double* in,
                      double* out,
                      const std::vector<std::int64_t>& in_shape,
                      const std::vector<int>& perm) {
    permute_typed<double>(in, out, in_shape, perm);
}

void permute_copy_i32(const std::int32_t* in,
                      std::int32_t* out,
                      const std::vector<std::int64_t>& in_shape,
                      const std::vector<int>& perm) {
    permute_typed<std::int32_t>(in, out, in_shape, perm);
}

void permute_copy_i64(const std::int64_t* in,
                      std::int64_t* out,
                      const std::vector<std::int64_t>& in_shape,
                      const std::vector<int>& perm) {
    permute_typed<std::int64_t>(in, out, in_shape, perm);
}

}  // namespace lucid::backend::cpu
