// lucid/_C/backend/gpu/ComplexParts.h
//
// A complex64 array taken apart into float32 parts and put back together,
// for the GPU work MLX has no complex64 kernel for (its scatter-add family)
// or judges the complex value as a whole where each part counts on its own
// (``nan_to_num``).
#pragma once

#include <mlx/array.h>
#include <mlx/ops.h>

namespace lucid {
namespace backend {

// A complex64 array of ``re``'s shape from float32 parts of that shape.
// Assembling it as ``re + im * 1j`` turns an infinite part into NaN — the
// product computes ``inf * 0`` for the other lane, so a scatter-add of
// (1 + inf j) answered (nan + inf j) — so the parts are interleaved and the
// pairs reinterpreted.
inline ::mlx::core::array complex_from_parts(const ::mlx::core::array& re,
                                             const ::mlx::core::array& im) {
    namespace mx = ::mlx::core;
    return mx::reshape(mx::view(mx::stack({re, im}, -1), mx::complex64), re.shape());
}

// ``op(base, updates)`` for complex64 operands, run on the real parts and on
// the imaginary parts.  Right for an operation that acts on each part alone:
// a sum, or an overwrite.
template <class Op>
::mlx::core::array
by_complex_parts(const ::mlx::core::array& base, const ::mlx::core::array& updates, Op&& op) {
    namespace mx = ::mlx::core;
    return complex_from_parts(op(mx::real(base), mx::real(updates)),
                              op(mx::imag(base), mx::imag(updates)));
}

}  // namespace backend
}  // namespace lucid
