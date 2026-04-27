#include "Inv.h"

#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr inv_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    if (!a) throw LucidError("inv: null input");
    require_gpu(a, "inv");
    OpScope scope{"inv", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    auto out = ::mlx::core::linalg::inv(*ga.arr, kMlxCpu);
    return fresh(Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)},
                 a->shape_, a->dtype_, a->device_);
}

}  // namespace lucid
