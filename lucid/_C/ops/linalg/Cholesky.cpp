#include "Cholesky.h"

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

TensorImplPtr cholesky_op(const TensorImplPtr& a, bool upper) {
    using namespace linalg_detail;
    if (!a)
        throw LucidError("cholesky: null input");
    require_float(a->dtype_, "cholesky");
    require_square_2d(a->shape_, "cholesky");
    OpScope scope{"cholesky", a->device_, a->dtype_, a->shape_};

    if (a->device_ == Device::GPU) {
        auto in = as_mlx_array_gpu(a);
        auto out = ::mlx::core::linalg::cholesky(in, upper, kMlxLinalgStream);
        return fresh(wrap_gpu_result(std::move(out), a->dtype_), a->shape_, a->dtype_, a->device_);
    }

    const auto& sh = a->shape_;
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const std::int64_t batch = leading_batch_count(sh, /*mat_dims=*/2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;

    auto out_cpu = allocate_cpu(sh, a->dtype_);
    const auto& in_cpu = std::get<CpuStorage>(a->storage_);
    std::memcpy(out_cpu.ptr.get(), in_cpu.ptr.get(), in_cpu.nbytes);

    const bool lower = !upper;
    int info = 0;
    if (a->dtype_ == Dtype::F32) {
        auto* p = reinterpret_cast<float*>(out_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_cholesky_f32(p + b * per_mat, n, lower, &info);
            check_lapack_info(info, "cholesky");
        }
    } else {
        auto* p = reinterpret_cast<double*>(out_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_cholesky_f64(p + b * per_mat, n, lower, &info);
            check_lapack_info(info, "cholesky");
        }
    }
    return fresh(Storage{std::move(out_cpu)}, sh, a->dtype_, a->device_);
}

}  // namespace lucid
