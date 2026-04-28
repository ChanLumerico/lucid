#include "QR.h"

#include <algorithm>
#include <variant>
#include <vector>

#include <mlx/linalg.h>

#include "../../backend/cpu/Lapack.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

std::pair<TensorImplPtr, TensorImplPtr> qr_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    if (!a)
        throw LucidError("qr: null input");
    require_float(a->dtype_, "qr");
    if (a->shape_.size() < 2)
        throw LucidError("qr: input must be at least 2-D");
    OpScope scope{"qr", a->device_, a->dtype_, a->shape_};

    if (a->device_ == Device::GPU) {
        auto in = as_mlx_array_gpu(a);
        auto [Q, R] = ::mlx::core::linalg::qr(in, kMlxLinalgStream);
        Shape qsh = mlx_shape_to_lucid(Q.shape());
        Shape rsh = mlx_shape_to_lucid(R.shape());
        return {
            fresh(wrap_gpu_result(std::move(Q), a->dtype_), std::move(qsh), a->dtype_, a->device_),
            fresh(wrap_gpu_result(std::move(R), a->dtype_), std::move(rsh), a->dtype_, a->device_),
        };
    }

    // CPU path: reduced QR per-batch via LAPACK geqrf + orgqr.
    const auto& sh = a->shape_;
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const int k = std::min(m, n);
    const std::int64_t batch = leading_batch_count(sh, /*mat_dims=*/2);
    const std::size_t in_per = static_cast<std::size_t>(m) * n;
    const std::size_t q_per = static_cast<std::size_t>(m) * k;
    const std::size_t r_per = static_cast<std::size_t>(k) * n;

    Shape qsh(sh.begin(), sh.end() - 2);
    qsh.push_back(m);
    qsh.push_back(k);
    Shape rsh(sh.begin(), sh.end() - 2);
    rsh.push_back(k);
    rsh.push_back(n);

    auto Qcpu = allocate_cpu(qsh, a->dtype_);
    auto Rcpu = allocate_cpu(rsh, a->dtype_);
    const auto& in_cpu = std::get<CpuStorage>(a->storage_);

    int info = 0;
    if (a->dtype_ == Dtype::F32) {
        const auto* in_p = reinterpret_cast<const float*>(in_cpu.ptr.get());
        auto* Q_p = reinterpret_cast<float*>(Qcpu.ptr.get());
        auto* R_p = reinterpret_cast<float*>(Rcpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_qr_f32(in_p + b * in_per, m, n, Q_p + b * q_per, R_p + b * r_per,
                                        &info);
            check_lapack_info(info, "qr");
        }
    } else {
        const auto* in_p = reinterpret_cast<const double*>(in_cpu.ptr.get());
        auto* Q_p = reinterpret_cast<double*>(Qcpu.ptr.get());
        auto* R_p = reinterpret_cast<double*>(Rcpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_qr_f64(in_p + b * in_per, m, n, Q_p + b * q_per, R_p + b * r_per,
                                        &info);
            check_lapack_info(info, "qr");
        }
    }
    return {
        fresh(Storage{std::move(Qcpu)}, qsh, a->dtype_, a->device_),
        fresh(Storage{std::move(Rcpu)}, rsh, a->dtype_, a->device_),
    };
}

}  // namespace lucid
