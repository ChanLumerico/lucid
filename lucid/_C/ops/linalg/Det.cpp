#include "Det.h"

#include <cstdint>
#include <cstring>
#include <variant>
#include <vector>

#include <mlx/linalg.h>
#include <mlx/ops.h>

#include "../../backend/cpu/Lapack.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

namespace {

// Sign of an LAPACK ipiv array (1-based row swaps): each pivot[i] != i+1
// is one swap.
inline float ipiv_sign(const int* ipiv, int n) {
    int swaps = 0;
    for (int i = 0; i < n; ++i)
        if (ipiv[i] != i + 1)
            ++swaps;
    return (swaps % 2 == 0) ? 1.0f : -1.0f;
}

// Sign of a permutation expressed as 0-based index array (MLX convention).
inline float perm_index_sign(const std::uint32_t* p, std::size_t n) {
    std::vector<bool> seen(n, false);
    std::size_t cycles = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (seen[i])
            continue;
        ++cycles;
        std::size_t j = i;
        while (!seen[j]) {
            seen[j] = true;
            j = p[j];
        }
    }
    return ((n - cycles) % 2 == 0) ? 1.0f : -1.0f;
}

}  // namespace

TensorImplPtr det_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "det.a").float_only().square_2d();
    OpScopeFull scope{"det", a->device_, a->dtype_, a->shape_};

    const auto& sh = a->shape_;
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const std::int64_t batch = leading_batch_count(sh, /*mat_dims=*/2);
    Shape out_shape(sh.begin(), sh.end() - 2);

    if (a->device_ == Device::GPU) {
        // MLX has no det; compute via LU then sign × prod(diag(U)).
        auto in = as_mlx_array_gpu(a);
        auto factors = ::mlx::core::linalg::lu(in, kMlxLinalgStream);
        if (factors.size() < 3)
            ErrorBuilder("det").fail("lu returned fewer than 3 factors");
        const auto& P = factors[0];
        const auto& U = factors[2];
        auto diag = ::mlx::core::diagonal(U, 0, -2, -1);
        auto detU = ::mlx::core::prod(diag, /*keepdims=*/false);

        auto P_eval = P;
        P_eval.eval();
        const std::size_t N = static_cast<std::size_t>(n);
        std::vector<float> signs(static_cast<std::size_t>(batch), 1.0f);
        const auto* p_data = P_eval.data<std::uint32_t>();
        for (std::int64_t b = 0; b < batch; ++b)
            signs[b] = perm_index_sign(p_data + b * N, N);

        ::mlx::core::Shape sign_shape;
        for (auto d : out_shape)
            sign_shape.push_back(d);
        ::mlx::core::array sign_arr(signs.data(), sign_shape, ::mlx::core::float32);
        if (a->dtype_ != Dtype::F32)
            sign_arr = ::mlx::core::astype(sign_arr, gpu::to_mlx_dtype(a->dtype_));
        auto detA = ::mlx::core::multiply(detU, sign_arr);
        return fresh(wrap_gpu_result(std::move(detA), a->dtype_), out_shape, a->dtype_, a->device_);
    }

    // CPU path: per-batch LAPACK getrf, sign × diag product.
    auto out_cpu = allocate_cpu(out_shape, a->dtype_);
    const auto& in_cpu = std::get<CpuStorage>(a->storage_);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;
    std::vector<int> ipiv(n);
    int info = 0;

    if (a->dtype_ == Dtype::F32) {
        std::vector<float> L(per_mat), U(per_mat);
        const auto* in_p = reinterpret_cast<const float*>(in_cpu.ptr.get());
        auto* out_p = reinterpret_cast<float*>(out_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_lu_f32(in_p + b * per_mat, n, ipiv.data(), L.data(), U.data(),
                                        &info);
            if (info < 0)
                check_lapack_info(info, "det");
            float det = ipiv_sign(ipiv.data(), n);
            if (info > 0) {
                det = 0.0f;  // singular
            } else {
                for (int i = 0; i < n; ++i)
                    det *= U[i * n + i];
            }
            out_p[b] = det;
        }
    } else {
        std::vector<double> L(per_mat), U(per_mat);
        const auto* in_p = reinterpret_cast<const double*>(in_cpu.ptr.get());
        auto* out_p = reinterpret_cast<double*>(out_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_lu_f64(in_p + b * per_mat, n, ipiv.data(), L.data(), U.data(),
                                        &info);
            if (info < 0)
                check_lapack_info(info, "det");
            double det = ipiv_sign(ipiv.data(), n);
            if (info > 0) {
                det = 0.0;
            } else {
                for (int i = 0; i < n; ++i)
                    det *= U[i * n + i];
            }
            out_p[b] = det;
        }
    }
    return fresh(Storage{std::move(out_cpu)}, out_shape, a->dtype_, a->device_);
}

}  // namespace lucid
