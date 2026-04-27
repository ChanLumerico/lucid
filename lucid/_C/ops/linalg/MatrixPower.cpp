#include "MatrixPower.h"

#include <cmath>
#include <variant>

#include <mlx/linalg.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr matrix_power_op(const TensorImplPtr& a, int n) {
    using namespace linalg_detail;
    if (!a) throw LucidError("matrix_power: null input");
    require_gpu(a, "matrix_power");
    if (a->shape_.size() < 2 ||
        a->shape_[a->shape_.size() - 1] != a->shape_[a->shape_.size() - 2])
        throw LucidError("matrix_power: last two dims must be square");
    OpScope scope{"matrix_power", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    if (n == 0) {
        auto eye = ::mlx::core::eye(static_cast<int>(a->shape_.back()),
                                     static_cast<int>(a->shape_.back()), 0,
                                     gpu::to_mlx_dtype(a->dtype_));
        return fresh(Storage{gpu::wrap_mlx_array(std::move(eye), a->dtype_)},
                     Shape{a->shape_.back(), a->shape_.back()},
                     a->dtype_, a->device_);
    }
    int reps = std::abs(n);
    auto base = (n < 0) ? ::mlx::core::linalg::inv(*ga.arr, kMlxCpu) : *ga.arr;
    ::mlx::core::array result = base;
    for (int i = 1; i < reps; ++i) {
        result = ::mlx::core::matmul(result, base);
    }
    Shape sh = mlx_shape_to_lucid(result.shape());
    return fresh(Storage{gpu::wrap_mlx_array(std::move(result), a->dtype_)},
                 std::move(sh), a->dtype_, a->device_);
}

}  // namespace lucid
