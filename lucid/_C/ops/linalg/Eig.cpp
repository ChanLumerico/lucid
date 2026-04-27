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
    OpScope scope{"eig", a->device_, a->dtype_, a->shape_};
    auto in = as_mlx_array(a);
    auto [w, v] = ::mlx::core::linalg::eig(in, kMlxCpu);
    Shape wsh = mlx_shape_to_lucid(w.shape());
    Shape vsh = mlx_shape_to_lucid(v.shape());
    return {
        fresh(wrap_result(std::move(w), a->dtype_, a->device_, wsh),
              std::move(wsh), a->dtype_, a->device_),
        fresh(wrap_result(std::move(v), a->dtype_, a->device_, vsh),
              std::move(vsh), a->dtype_, a->device_),
    };
}

}  // namespace lucid
