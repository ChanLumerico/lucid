#include "QR.h"

#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

std::pair<TensorImplPtr, TensorImplPtr> qr_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    if (!a) throw LucidError("qr: null input");
    require_gpu(a, "qr");
    OpScope scope{"qr", a->device_, a->dtype_, a->shape_};
    const auto& ga = std::get<GpuStorage>(a->storage_);
    auto [Q, R] = ::mlx::core::linalg::qr(*ga.arr, kMlxCpu);
    Shape qsh = mlx_shape_to_lucid(Q.shape());
    Shape rsh = mlx_shape_to_lucid(R.shape());
    return {
        fresh(Storage{gpu::wrap_mlx_array(std::move(Q), a->dtype_)},
              std::move(qsh), a->dtype_, a->device_),
        fresh(Storage{gpu::wrap_mlx_array(std::move(R), a->dtype_)},
              std::move(rsh), a->dtype_, a->device_),
    };
}

}  // namespace lucid
