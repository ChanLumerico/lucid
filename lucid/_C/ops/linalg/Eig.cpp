#include "Eig.h"

#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

std::pair<TensorImplPtr, TensorImplPtr> eig_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    if (!a) throw LucidError("eig: null input");
    require_gpu(a, "eig");
    OpScope scope{"eig", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    auto [w, v] = ::mlx::core::linalg::eig(*ga.arr, kMlxCpu);
    Shape wsh = mlx_shape_to_lucid(w.shape());
    Shape vsh = mlx_shape_to_lucid(v.shape());
    return {
        fresh(Storage{gpu::wrap_mlx_array(std::move(w), a->dtype_)},
              std::move(wsh), a->dtype_, a->device_),
        fresh(Storage{gpu::wrap_mlx_array(std::move(v), a->dtype_)},
              std::move(vsh), a->dtype_, a->device_),
    };
}

}  // namespace lucid
