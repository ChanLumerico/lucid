#include "Solve.h"

#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    using namespace linalg_detail;
    if (!a || !b) throw LucidError("solve: null input");
    if (a->device_ != b->device_ || a->device_ != Device::GPU)
        throw NotImplementedError("solve: requires both inputs on device=\"gpu\"");
    OpScope scope{"solve", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    const auto& gb = std::get<GpuStorage>(b->storage_);
    auto out = ::mlx::core::linalg::solve(*ga.arr, *gb.arr, kMlxCpu);
    Shape sh = mlx_shape_to_lucid(out.shape());
    return fresh(Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)},
                 std::move(sh), a->dtype_, a->device_);
}

}  // namespace lucid
