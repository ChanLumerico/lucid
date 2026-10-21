// lucid/_C/ops/fft/Rfftn.cpp
//
// Forward N-dimensional real DFT via mlx::core::fft::rfftn.
// Real input is rejected unless dtype is F32 / F16 (see _Detail.h).

#include "Rfftn.h"

#include "_Detail.h"

namespace lucid {

TensorImplPtr rfftn_op(const TensorImplPtr& a,
                       const std::vector<std::int64_t>& n,
                       const std::vector<int>& axes_in) {
    using ::mlx::core::fft::FFTNorm;
    return fft_detail::run_nd_fft(
        "rfftn", "rfftn.a", a, n, axes_in, Dtype::C128, fft_detail::dtype_for_rfft,
        fft_detail::rfft_out_shape,
        [&](const std::vector<int>& axes, const Shape& out_shape) {
            return fft_detail::cpu_rfftn_f64(a, n, axes, out_shape);
        },
        [&](const ::mlx::core::array& in_arr, const std::vector<int>& axes) {
            return n.empty()
                       ? ::mlx::core::fft::rfftn(in_arr, axes, FFTNorm::Backward,
                                                 fft_detail::kMlxFftStream)
                       : ::mlx::core::fft::rfftn(in_arr, fft_detail::mlx_n_from_lucid(n, "rfftn"),
                                                 axes, FFTNorm::Backward,
                                                 fft_detail::kMlxFftStream);
        });
}

}  // namespace lucid
