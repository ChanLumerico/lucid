#include "SVD.h"

#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

std::vector<TensorImplPtr> svd_op(const TensorImplPtr& a, bool compute_uv) {
    using namespace linalg_detail;
    if (!a) throw LucidError("svd: null input");
    require_gpu(a, "svd");
    OpScope scope{"svd", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    auto pieces = ::mlx::core::linalg::svd(*ga.arr, compute_uv, kMlxCpu);
    std::vector<TensorImplPtr> out;
    out.reserve(pieces.size());
    for (auto& p : pieces) {
        Shape sh = mlx_shape_to_lucid(p.shape());
        out.push_back(fresh(Storage{gpu::wrap_mlx_array(std::move(p), a->dtype_)},
                            std::move(sh), a->dtype_, a->device_));
    }
    return out;
}

}  // namespace lucid
