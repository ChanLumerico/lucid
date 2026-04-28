#include "Pinv.h"

#include <algorithm>
#include <cmath>
#include <variant>
#include <vector>

#include <mlx/linalg.h>

#include "../../backend/cpu/Blas.h"
#include "../../backend/cpu/Lapack.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

namespace {

// Pseudoinverse via SVD: A^+ = V * diag(1/s) * U^T.
// The threshold below comes from numpy.linalg.pinv: rcond * max(s).
template <typename T>
void pinv_one(const T* A, int m, int n, T* Aplus) {
    using namespace backend::cpu;
    const int k = std::min(m, n);
    std::vector<T> U(static_cast<std::size_t>(m) * k);
    std::vector<T> S(k);
    std::vector<T> Vt(static_cast<std::size_t>(k) * n);
    int info = 0;
    if constexpr (std::is_same_v<T, float>)
        lapack_svd_f32(A, m, n, false, U.data(), S.data(), Vt.data(), &info);
    else
        lapack_svd_f64(A, m, n, false, U.data(), S.data(), Vt.data(), &info);
    if (info != 0)
        ErrorBuilder("pinv").fail("SVD did not converge");

    // rcond = eps * max(m, n).
    const T smax = (k > 0) ? *std::max_element(S.begin(), S.end()) : T{0};
    const T rcond = std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(m, n));
    const T cutoff = rcond * smax;

    // Scale U^T by 1/s: form S_inv * U^T (k × m). U is (m, k); U^T is (k, m).
    std::vector<T> S_inv_Ut(static_cast<std::size_t>(k) * m);
    for (int i = 0; i < k; ++i) {
        const T inv = (S[i] > cutoff) ? T{1} / S[i] : T{0};
        for (int j = 0; j < m; ++j)
            S_inv_Ut[i * m + j] = inv * U[j * k + i];
    }

    // A^+ (n × m) = V^T^T (n×k) * (S_inv * U^T) (k×m) = V * (S_inv U^T).
    // Use cblas_*gemm: C = V^T^T * (S_inv U^T) — V^T is (k, n), so we need
    // transA=true on V^T to get V (n,k).
    if constexpr (std::is_same_v<T, float>) {
        sgemm(/*transA=*/true, /*transB=*/false, n, m, k, 1.0f, Vt.data(), n, S_inv_Ut.data(), m,
              0.0f, Aplus, m);
    } else {
        dgemm(/*transA=*/true, /*transB=*/false, n, m, k, 1.0, Vt.data(), n, S_inv_Ut.data(), m,
              0.0, Aplus, m);
    }
}

}  // namespace

TensorImplPtr pinv_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "pinv.a").non_null();
    require_float(a->dtype_, "pinv");
    if (a->shape_.size() < 2)
        ErrorBuilder("pinv").fail("input must be at least 2-D");
    OpScopeFull scope{"pinv", a->device_, a->dtype_, a->shape_};

    if (a->device_ == Device::GPU) {
        auto in = as_mlx_array_gpu(a);
        auto out = ::mlx::core::linalg::pinv(in, kMlxLinalgStream);
        Shape sh = mlx_shape_to_lucid(out.shape());
        return fresh(wrap_gpu_result(std::move(out), a->dtype_), std::move(sh), a->dtype_,
                     a->device_);
    }

    const auto& sh = a->shape_;
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const std::int64_t batch = leading_batch_count(sh, /*mat_dims=*/2);

    Shape out_shape(sh.begin(), sh.end() - 2);
    out_shape.push_back(n);
    out_shape.push_back(m);

    auto out_cpu = allocate_cpu(out_shape, a->dtype_);
    const auto& in_cpu = std::get<CpuStorage>(a->storage_);
    const std::size_t in_per = static_cast<std::size_t>(m) * n;
    const std::size_t out_per = static_cast<std::size_t>(n) * m;

    if (a->dtype_ == Dtype::F32) {
        const auto* in_p = reinterpret_cast<const float*>(in_cpu.ptr.get());
        auto* out_p = reinterpret_cast<float*>(out_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b)
            pinv_one(in_p + b * in_per, m, n, out_p + b * out_per);
    } else {
        const auto* in_p = reinterpret_cast<const double*>(in_cpu.ptr.get());
        auto* out_p = reinterpret_cast<double*>(out_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b)
            pinv_one(in_p + b * in_per, m, n, out_p + b * out_per);
    }
    return fresh(Storage{std::move(out_cpu)}, out_shape, a->dtype_, a->device_);
}

}  // namespace lucid
