#include "Solve.h"

#include <cstring>
#include <variant>

#include <mlx/linalg.h>

#include "../../backend/cpu/Lapack.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    using namespace linalg_detail;
    if (!a || !b)
        throw LucidError("solve: null input");
    if (a->device_ != b->device_)
        throw DeviceMismatch(std::string(device_name(a->device_)),
                             std::string(device_name(b->device_)), "solve");
    if (a->dtype_ != b->dtype_)
        throw DtypeMismatch(std::string(dtype_name(a->dtype_)), std::string(dtype_name(b->dtype_)),
                            "solve");
    require_float(a->dtype_, "solve");
    require_square_2d(a->shape_, "solve");
    OpScope scope{"solve", a->device_, a->dtype_, a->shape_};

    if (a->device_ == Device::GPU) {
        auto in_a = as_mlx_array_gpu(a);
        auto in_b = as_mlx_array_gpu(b);
        auto out = ::mlx::core::linalg::solve(in_a, in_b, kMlxLinalgStream);
        Shape sh = mlx_shape_to_lucid(out.shape());
        return fresh(wrap_gpu_result(std::move(out), a->dtype_), std::move(sh), a->dtype_,
                     a->device_);
    }

    // CPU path: A (..., n, n), B (..., n, k) or (..., n).
    const auto& sh_a = a->shape_;
    const auto& sh_b = b->shape_;
    const int n = static_cast<int>(sh_a[sh_a.size() - 1]);
    const bool b_is_vec = (sh_b.size() == sh_a.size() - 1);
    const int nrhs = b_is_vec ? 1 : static_cast<int>(sh_b[sh_b.size() - 1]);
    const std::int64_t batch = leading_batch_count(sh_a, /*mat_dims=*/2);

    // Allocate result with B's shape (X has same shape as B).
    auto out_cpu = allocate_cpu(sh_b, a->dtype_);
    const auto& a_cpu = std::get<CpuStorage>(a->storage_);
    const auto& b_cpu = std::get<CpuStorage>(b->storage_);
    std::memcpy(out_cpu.ptr.get(), b_cpu.ptr.get(), b_cpu.nbytes);

    const std::size_t a_per = static_cast<std::size_t>(n) * n;
    const std::size_t b_per = static_cast<std::size_t>(n) * nrhs;

    int info = 0;
    if (a->dtype_ == Dtype::F32) {
        std::vector<float> A_local(a_per);
        const auto* a_p = reinterpret_cast<const float*>(a_cpu.ptr.get());
        auto* x_p = reinterpret_cast<float*>(out_cpu.ptr.get());
        for (std::int64_t bi = 0; bi < batch; ++bi) {
            std::memcpy(A_local.data(), a_p + bi * a_per, a_per * sizeof(float));
            backend::cpu::lapack_solve_f32(A_local.data(), x_p + bi * b_per, n, nrhs, &info);
            check_lapack_info(info, "solve");
        }
    } else {
        std::vector<double> A_local(a_per);
        const auto* a_p = reinterpret_cast<const double*>(a_cpu.ptr.get());
        auto* x_p = reinterpret_cast<double*>(out_cpu.ptr.get());
        for (std::int64_t bi = 0; bi < batch; ++bi) {
            std::memcpy(A_local.data(), a_p + bi * a_per, a_per * sizeof(double));
            backend::cpu::lapack_solve_f64(A_local.data(), x_p + bi * b_per, n, nrhs, &info);
            check_lapack_info(info, "solve");
        }
    }
    return fresh(Storage{std::move(out_cpu)}, sh_b, a->dtype_, a->device_);
}

}  // namespace lucid
