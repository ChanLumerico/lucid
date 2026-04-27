#include "Norm.h"

#include <optional>
#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr norm_op(const TensorImplPtr& a, double ord,
                      std::vector<int> axis, bool keepdims) {
    using namespace linalg_detail;
    if (!a) throw LucidError("norm: null input");
    require_gpu(a, "norm");
    OpScope scope{"norm", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    std::optional<std::vector<int>> axis_opt;
    if (!axis.empty()) axis_opt = std::move(axis);
    auto out = ::mlx::core::linalg::norm(*ga.arr, ord, axis_opt, keepdims,
                                          kMlxCpu);
    Shape sh = mlx_shape_to_lucid(out.shape());
    return fresh(Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)},
                 std::move(sh), a->dtype_, a->device_);
}

}  // namespace lucid
