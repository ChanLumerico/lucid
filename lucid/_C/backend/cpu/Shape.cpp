#include "Shape.h"

#include <cstddef>
#include <vector>

namespace lucid::backend::cpu {

namespace {

// Compute element-stride (not byte-stride) for a contiguous shape.
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

// Multi-dimensional copy with permuted output layout.
//
// We iterate over output flat indices and unravel into output multi-index;
// the corresponding input multi-index is obtained by inverse-permutation
// (input[d] = output[perm_inv[d]]). This is one read + one write per
// element with no extra allocations on the hot path.
template <typename T>
void permute_typed(const T* in,
                   T* out,
                   const std::vector<std::int64_t>& in_shape,
                   const std::vector<int>& perm) {
    const std::size_t ndim = in_shape.size();
    if (ndim == 0) {
        // Scalar: just copy the single element.
        out[0] = in[0];
        return;
    }

    // Build out_shape from in_shape via perm.
    std::vector<std::int64_t> out_shape(ndim);
    for (std::size_t d = 0; d < ndim; ++d)
        out_shape[d] = in_shape[static_cast<std::size_t>(perm[d])];

    const auto in_strides = elem_strides(in_shape);
    const auto out_strides = elem_strides(out_shape);

    // Total elements in output = total in input (permutation preserves numel).
    std::size_t numel = 1;
    for (auto d : out_shape)
        numel *= static_cast<std::size_t>(d);
    if (numel == 0)
        return;

    std::vector<std::int64_t> out_idx(ndim);
    for (std::size_t flat = 0; flat < numel; ++flat) {
        // Unravel `flat` against out_shape.
        std::size_t rem = flat;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::size_t s = static_cast<std::size_t>(out_strides[d]);
            const std::size_t k = (s == 0) ? 0 : (rem / s);
            out_idx[d] = static_cast<std::int64_t>(k);
            if (s)
                rem -= k * s;
        }
        // Map to input flat: in_idx[perm[d]] = out_idx[d].
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
