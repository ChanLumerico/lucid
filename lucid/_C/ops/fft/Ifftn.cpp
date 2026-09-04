// lucid/_C/ops/fft/Ifftn.cpp
//
// Forward N-dimensional inverse DFT via mlx::core::fft::ifftn.

#include "Ifftn.h"

#include <variant>

#include "../../compile/Tracer.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr ifftn_op(const TensorImplPtr& a,
                       const std::vector<std::int64_t>& n,
                       const std::vector<int>& axes_in) {
    Validator::input(a, "ifftn.a").non_null();
    OpScopeFull scope{"ifftn", a->device(), a->dtype(), a->shape()};

    const Dtype out_dtype = fft_detail::dtype_for_complex_fft(a->dtype());
    const int rank = static_cast<int>(a->shape().size());
    if (rank == 0)
        ErrorBuilder("ifftn").fail("input must be at least 1-D");

    std::vector<int> axes = axes_in;
    fft_detail::default_axes_all(axes, rank);
    fft_detail::normalise_axes(axes, rank, "ifftn");

    const Shape out_shape = fft_detail::complex_fft_out_shape(a->shape(), n, axes, "ifftn");

    auto in_arr = fft_detail::as_mlx_input(a);
    using ::mlx::core::fft::FFTNorm;
    ::mlx::core::array out_arr =
        n.empty()
            ? ::mlx::core::fft::ifftn(in_arr, axes, FFTNorm::Backward, fft_detail::kMlxFftStream)
            : ::mlx::core::fft::ifftn(in_arr, fft_detail::mlx_n_from_lucid(n), axes,
                                      FFTNorm::Backward, fft_detail::kMlxFftStream);

    Storage out =
        fft_detail::finalise_result(std::move(out_arr), out_dtype, out_shape, a->device());
    auto result = fft_detail::fresh(std::move(out), out_shape, out_dtype, a->device());
    // Record the trace I/O explicitly.  Without it the node keeps the
    // empty input list ``on_op_enter`` seeded and its result never
    // becomes a traced tensor, so the builder drops the node as dead and
    // the *consumer* sees the result as a fresh external feed — bound
    // once, at trace time.  The compiled model then reported success and
    // returned that first answer for every later input.
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a}, result);
    return result;
}

}  // namespace lucid
