#include "Cholesky.h"

#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr cholesky_op(const TensorImplPtr& a, bool upper) {
    using namespace linalg_detail;
    if (!a) throw LucidError("cholesky: null input");
    require_gpu(a, "cholesky");
    OpScope scope{"cholesky", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    auto out = ::mlx::core::linalg::cholesky(*ga.arr, upper, kMlxCpu);
    return fresh(Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)},
                 a->shape_, a->dtype_, a->device_);
}

}  // namespace lucid
