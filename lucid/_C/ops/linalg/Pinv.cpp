// lucid/_C/ops/linalg/Pinv.cpp
//
// Implementation of the Moore-Penrose pseudoinverse forward op.
//
// The transposed output shape (..., n, m) is computed here and passed to
// fresh() so the TensorImpl carries the correct shape metadata.
//
// Forward:
//   CPU path: the backend calls LAPACK dgesdd to compute a full SVD
//             A = U S Vᵀ, inverts the non-negligible singular values to form
//             S⁺, and then computes A⁺ = V S⁺ Uᵀ.
//   GPU path: mlx::core::linalg::pinv() on the CPU stream.
//
// The rcond threshold (below which singular values are treated as zero) is
// chosen by the backend and typically follows the NumPy convention:
//   rcond = max(m, n) * max(S) * machine_epsilon
//
// Autograd is not wired here.  A future backward would use the pseudoinverse
// differential identity:
//   dA⁺ = -A⁺ (dA) A⁺ + A⁺ A⁺ᵀ (dAᵀ)(I - A A⁺) + (I - A⁺ A)(dAᵀ) A⁺ᵀ A⁺
// which reduces to the ordinary inverse gradient when A is square and full rank.

#include "Pinv.h"

#include <algorithm>
#include <cmath>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {
namespace {

template <typename T>
void zero_inf_only_pinv_slices(const CpuStorage& in_s,
                               CpuStorage& out_s,
                               std::int64_t batch,
                               std::size_t mn) {
    const auto* in_p = reinterpret_cast<const T*>(in_s.ptr.get());
    auto* out_p = reinterpret_cast<T*>(out_s.ptr.get());
    for (std::int64_t b = 0; b < batch; ++b) {
        const T* slice = in_p + static_cast<std::size_t>(b) * mn;
        bool has_inf = false;
        bool has_nan = false;
        for (std::size_t i = 0; i < mn; ++i) {
            has_inf |= std::isinf(slice[i]);
            has_nan |= std::isnan(slice[i]);
        }
        if (has_inf && !has_nan)
            std::fill_n(out_p + static_cast<std::size_t>(b) * mn, mn, T{0});
    }
}

}  // namespace

// Compute A⁺ (the Moore-Penrose pseudoinverse) of a.
//
// The output shape swaps the last two dimensions of the input because the
// pseudoinverse of an (m×n) matrix is (n×m).
TensorImplPtr pinv_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "pinv.a").non_null();
    require_float(a->dtype(), "pinv");
    if (a->shape().size() < 2)
        ErrorBuilder("pinv").invalid_argument("input must be at least 2-D");
    OpScopeFull scope{"pinv", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    // The pseudoinverse transposes the matrix dims: output is (..., n, m).
    Shape out_shape(sh.begin(), sh.end() - 2);
    out_shape.push_back(n);
    out_shape.push_back(m);

    // A degenerate matrix has an answer; LAPACK just will not be the one
    // to compute it — see ``empty_matrix``.
    if (empty_matrix(sh))
        return zeros_op(out_shape, a->dtype(), a->device());

    Storage out =
        backend::Dispatcher::for_device(a->device()).linalg_pinv(a->storage(), sh, a->dtype());
    if (a->device() == Device::CPU) {
        const std::int64_t batch = leading_batch_count(sh, 2);
        const std::size_t mn = static_cast<std::size_t>(m) * n;
        const auto& in_cpu = std::get<CpuStorage>(a->storage());
        auto& out_cpu = std::get<CpuStorage>(out);
        if (a->dtype() == Dtype::F32)
            zero_inf_only_pinv_slices<float>(in_cpu, out_cpu, batch, mn);
        else if (a->dtype() == Dtype::F64)
            zero_inf_only_pinv_slices<double>(in_cpu, out_cpu, batch, mn);
    }
    return fresh(std::move(out), std::move(out_shape), a->dtype(), a->device());
}

}  // namespace lucid
