#include "MatrixPower.h"

#include <cmath>
#include <cstring>
#include <variant>
#include <vector>

#include <mlx/linalg.h>
#include <mlx/ops.h>

#include "../../backend/cpu/Blas.h"
#include "../../backend/cpu/Lapack.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

namespace {

// Set out (n×n) to identity (single matrix).
template <typename T>
inline void set_eye(T* out, int n) {
    const std::size_t total = static_cast<std::size_t>(n) * n;
    std::memset(out, 0, total * sizeof(T));
    for (int i = 0; i < n; ++i)
        out[i * n + i] = T{1};
}

// out (n×n) = a (n×n) @ b (n×n) row-major.
inline void matmul_one_f32(const float* a, const float* b, float* out, int n) {
    backend::cpu::sgemm(false, false, n, n, n, 1.0f, a, n, b, n, 0.0f, out, n);
}
inline void matmul_one_f64(const double* a, const double* b, double* out, int n) {
    backend::cpu::dgemm(false, false, n, n, n, 1.0, a, n, b, n, 0.0, out, n);
}

}  // namespace

TensorImplPtr matrix_power_op(const TensorImplPtr& a, int p) {
    using namespace linalg_detail;
    Validator::input(a, "matrix_power.a").non_null();
    require_float(a->dtype(), "matrix_power");
    require_square_2d(a->shape(), "matrix_power");
    OpScopeFull scope{"matrix_power", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const std::int64_t batch = leading_batch_count(sh, /*mat_dims=*/2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;

    if (a->device() == Device::GPU) {
        auto in = as_mlx_array_gpu(a);
        if (p == 0) {
            auto eye = ::mlx::core::eye(n, n, 0, gpu::to_mlx_dtype(a->dtype()));
            // Broadcast identity across batch dims (if any).
            if (sh.size() > 2) {
                ::mlx::core::Shape target;
                for (auto d : sh)
                    target.push_back(static_cast<int>(d));
                eye = ::mlx::core::broadcast_to(eye, target);
                eye = ::mlx::core::contiguous(eye);
            }
            return fresh(wrap_gpu_result(std::move(eye), a->dtype()), sh, a->dtype(), a->device());
        }
        const int reps = std::abs(p);
        auto base = (p < 0) ? ::mlx::core::linalg::inv(in, kMlxLinalgStream) : in;
        ::mlx::core::array result = base;
        for (int i = 1; i < reps; ++i)
            result = ::mlx::core::matmul(result, base);
        return fresh(wrap_gpu_result(std::move(result), a->dtype()), sh, a->dtype(), a->device());
    }

    // CPU path: per-batch sgemm/dgemm chain. p=0 → identity. p<0 → inv first.
    auto out_cpu = allocate_cpu(sh, a->dtype());
    const auto& in_cpu = std::get<CpuStorage>(a->storage());
    const int reps = std::abs(p);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* in_p = reinterpret_cast<const T*>(in_cpu.ptr.get());
        T* out_p = reinterpret_cast<T*>(out_cpu.ptr.get());

        std::vector<T> base(per_mat), tmp(per_mat);
        for (std::int64_t b = 0; b < batch; ++b) {
            T* result = out_p + b * per_mat;
            if (p == 0) {
                set_eye(result, n);
                continue;
            }

            // base = (p<0) ? inv(A_b) : A_b
            std::memcpy(base.data(), in_p + b * per_mat, per_mat * sizeof(T));
            if (p < 0) {
                int info = 0;
                if constexpr (std::is_same_v<T, float>)
                    backend::cpu::lapack_inv_f32(base.data(), n, &info);
                else
                    backend::cpu::lapack_inv_f64(base.data(), n, &info);
                check_lapack_info(info, "matrix_power");
            }
            // result = base; for i in 1..reps: result = result @ base.
            std::memcpy(result, base.data(), per_mat * sizeof(T));
            for (int i = 1; i < reps; ++i) {
                if constexpr (std::is_same_v<T, float>)
                    matmul_one_f32(result, base.data(), tmp.data(), n);
                else
                    matmul_one_f64(result, base.data(), tmp.data(), n);
                std::memcpy(result, tmp.data(), per_mat * sizeof(T));
            }
        }
    };

    if (a->dtype() == Dtype::F32)
        run(float{});
    else
        run(double{});
    return fresh(Storage{std::move(out_cpu)}, sh, a->dtype(), a->device());
}

}  // namespace lucid
