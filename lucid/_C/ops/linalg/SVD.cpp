#include "SVD.h"

#include <algorithm>
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

std::vector<TensorImplPtr> svd_op(const TensorImplPtr& a, bool compute_uv) {
    using namespace linalg_detail;
    Validator::input(a, "svd.a").non_null();
    require_float(a->dtype_, "svd");
    if (a->shape_.size() < 2)
        ErrorBuilder("svd").fail("input must be at least 2-D");
    OpScopeFull scope{"svd", a->device_, a->dtype_, a->shape_};

    if (a->device_ == Device::GPU) {
        auto in = as_mlx_array_gpu(a);
        auto pieces = ::mlx::core::linalg::svd(in, compute_uv, kMlxLinalgStream);
        std::vector<TensorImplPtr> out;
        out.reserve(pieces.size());
        for (auto& p : pieces) {
            Shape sh = mlx_shape_to_lucid(p.shape());
            out.push_back(fresh(wrap_gpu_result(std::move(p), a->dtype_), std::move(sh), a->dtype_,
                                a->device_));
        }
        return out;
    }

    // CPU path: reduced (or full) SVD per-batch via LAPACK gesdd.
    const auto& sh = a->shape_;
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const int k = std::min(m, n);
    const std::int64_t batch = leading_batch_count(sh, /*mat_dims=*/2);
    const std::size_t in_per = static_cast<std::size_t>(m) * n;
    const std::size_t s_per = static_cast<std::size_t>(k);

    if (!compute_uv) {
        // Singular values only.
        Shape ssh(sh.begin(), sh.end() - 2);
        ssh.push_back(k);
        auto Scpu = allocate_cpu(ssh, a->dtype_);
        const auto& in_cpu = std::get<CpuStorage>(a->storage_);

        // gesdd needs U/Vt buffers even when we discard them.
        const std::size_t u_per = static_cast<std::size_t>(m) * k;
        const std::size_t vt_per = static_cast<std::size_t>(k) * n;
        int info = 0;
        if (a->dtype_ == Dtype::F32) {
            std::vector<float> U(u_per), Vt(vt_per);
            const auto* in_p = reinterpret_cast<const float*>(in_cpu.ptr.get());
            auto* S_p = reinterpret_cast<float*>(Scpu.ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                backend::cpu::lapack_svd_f32(in_p + b * in_per, m, n,
                                             /*full_matrices=*/false, U.data(), S_p + b * s_per,
                                             Vt.data(), &info);
                check_lapack_info(info, "svd");
            }
        } else {
            std::vector<double> U(u_per), Vt(vt_per);
            const auto* in_p = reinterpret_cast<const double*>(in_cpu.ptr.get());
            auto* S_p = reinterpret_cast<double*>(Scpu.ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                backend::cpu::lapack_svd_f64(in_p + b * in_per, m, n,
                                             /*full_matrices=*/false, U.data(), S_p + b * s_per,
                                             Vt.data(), &info);
                check_lapack_info(info, "svd");
            }
        }
        std::vector<TensorImplPtr> out;
        out.push_back(fresh(Storage{std::move(Scpu)}, ssh, a->dtype_, a->device_));
        return out;
    }

    // Full triple (U, S, V^T). Match MLX's "reduced" convention: U (m,k),
    // S (k,), V^T (k,n).
    Shape ush(sh.begin(), sh.end() - 2);
    ush.push_back(m);
    ush.push_back(k);
    Shape ssh(sh.begin(), sh.end() - 2);
    ssh.push_back(k);
    Shape vsh(sh.begin(), sh.end() - 2);
    vsh.push_back(k);
    vsh.push_back(n);

    auto Ucpu = allocate_cpu(ush, a->dtype_);
    auto Scpu = allocate_cpu(ssh, a->dtype_);
    auto Vcpu = allocate_cpu(vsh, a->dtype_);
    const auto& in_cpu = std::get<CpuStorage>(a->storage_);

    const std::size_t u_per = static_cast<std::size_t>(m) * k;
    const std::size_t vt_per = static_cast<std::size_t>(k) * n;
    int info = 0;
    if (a->dtype_ == Dtype::F32) {
        const auto* in_p = reinterpret_cast<const float*>(in_cpu.ptr.get());
        auto* U_p = reinterpret_cast<float*>(Ucpu.ptr.get());
        auto* S_p = reinterpret_cast<float*>(Scpu.ptr.get());
        auto* V_p = reinterpret_cast<float*>(Vcpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_svd_f32(in_p + b * in_per, m, n,
                                         /*full_matrices=*/false, U_p + b * u_per, S_p + b * s_per,
                                         V_p + b * vt_per, &info);
            check_lapack_info(info, "svd");
        }
    } else {
        const auto* in_p = reinterpret_cast<const double*>(in_cpu.ptr.get());
        auto* U_p = reinterpret_cast<double*>(Ucpu.ptr.get());
        auto* S_p = reinterpret_cast<double*>(Scpu.ptr.get());
        auto* V_p = reinterpret_cast<double*>(Vcpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_svd_f64(in_p + b * in_per, m, n,
                                         /*full_matrices=*/false, U_p + b * u_per, S_p + b * s_per,
                                         V_p + b * vt_per, &info);
            check_lapack_info(info, "svd");
        }
    }
    std::vector<TensorImplPtr> out;
    out.push_back(fresh(Storage{std::move(Ucpu)}, ush, a->dtype_, a->device_));
    out.push_back(fresh(Storage{std::move(Scpu)}, ssh, a->dtype_, a->device_));
    out.push_back(fresh(Storage{std::move(Vcpu)}, vsh, a->dtype_, a->device_));
    return out;
}

}  // namespace lucid
