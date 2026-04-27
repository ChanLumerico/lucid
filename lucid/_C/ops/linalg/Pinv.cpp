#include "Pinv.h"

#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr pinv_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    if (!a) throw LucidError("pinv: null input");
    require_gpu(a, "pinv");
    OpScope scope{"pinv", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    auto out = ::mlx::core::linalg::pinv(*ga.arr, kMlxCpu);
    Shape sh = mlx_shape_to_lucid(out.shape());
    return fresh(Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)},
                 std::move(sh), a->dtype_, a->device_);
}

}  // namespace lucid
