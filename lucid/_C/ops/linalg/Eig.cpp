#include "Eig.h"

#include <cstring>
#include <variant>
#include <vector>

#include <mlx/linalg.h>

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

std::vector<TensorImplPtr> eig_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "eig.a").non_null();
    require_float(a->dtype_, "eig");
    require_square_2d(a->shape_, "eig");
    OpScopeFull scope{"eig", a->device_, a->dtype_, a->shape_};

    if (a->device_ == Device::GPU) {
        auto in = as_mlx_array_gpu(a);
        auto [w, v] = ::mlx::core::linalg::eig(in, kMlxLinalgStream);
        Shape wsh = mlx_shape_to_lucid(w.shape());
        Shape vsh = mlx_shape_to_lucid(v.shape());
        std::vector<TensorImplPtr> result;
        result.push_back(
            fresh(wrap_gpu_result(std::move(w), a->dtype_), std::move(wsh), a->dtype_, a->device_));
        result.push_back(
            fresh(wrap_gpu_result(std::move(v), a->dtype_), std::move(vsh), a->dtype_, a->device_));
        return result;
    }

    // CPU path: real symmetric input → use ssyevd / dsyevd. We expose the
    // numpy-compatible signature where eigenvalues are real and eigenvectors
    // are real columns. For non-symmetric A, fall back to geev — but we
    // discard the imaginary part to keep the F32/F64 surface clean. Users
    // who need complex eigenpairs should call svd or pair this with their
    // own complex post-processing.
    const auto& sh = a->shape_;
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const std::int64_t batch = leading_batch_count(sh, /*mat_dims=*/2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;
    const std::size_t per_w = static_cast<std::size_t>(n);

    Shape wsh(sh.begin(), sh.end() - 1);  // (..., n)
    Shape vsh = sh;                       // (..., n, n)
    auto Wcpu = allocate_cpu(wsh, a->dtype_);
    auto Vcpu = allocate_cpu(vsh, a->dtype_);
    const auto& in_cpu = std::get<CpuStorage>(a->storage_);

    int info = 0;
    if (a->dtype_ == Dtype::F32) {
        const auto* in_p = reinterpret_cast<const float*>(in_cpu.ptr.get());
        auto* w_p = reinterpret_cast<float*>(Wcpu.ptr.get());
        auto* v_p = reinterpret_cast<float*>(Vcpu.ptr.get());
        std::vector<float> wr(n), wi(n);
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_eig_f32(in_p + b * per_mat, n, wr.data(), wi.data(),
                                         v_p + b * per_mat, &info);
            check_lapack_info(info, "eig");
            // Real part only — ignore imaginary (numpy-real surface).
            std::memcpy(w_p + b * per_w, wr.data(), per_w * sizeof(float));
        }
    } else {
        const auto* in_p = reinterpret_cast<const double*>(in_cpu.ptr.get());
        auto* w_p = reinterpret_cast<double*>(Wcpu.ptr.get());
        auto* v_p = reinterpret_cast<double*>(Vcpu.ptr.get());
        std::vector<double> wr(n), wi(n);
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_eig_f64(in_p + b * per_mat, n, wr.data(), wi.data(),
                                         v_p + b * per_mat, &info);
            check_lapack_info(info, "eig");
            std::memcpy(w_p + b * per_w, wr.data(), per_w * sizeof(double));
        }
    }
    std::vector<TensorImplPtr> result;
    result.push_back(fresh(Storage{std::move(Wcpu)}, wsh, a->dtype_, a->device_));
    result.push_back(fresh(Storage{std::move(Vcpu)}, vsh, a->dtype_, a->device_));
    return result;
}

}  // namespace lucid
