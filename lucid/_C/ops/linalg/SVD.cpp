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
    OpScope scope{"svd", a->device_, a->dtype_, a->shape_};
    auto in = as_mlx_array(a);
    auto pieces = ::mlx::core::linalg::svd(in, compute_uv, kMlxCpu);
    std::vector<TensorImplPtr> out;
    out.reserve(pieces.size());
    for (auto& p : pieces) {
        Shape sh = mlx_shape_to_lucid(p.shape());
        out.push_back(fresh(wrap_result(std::move(p), a->dtype_, a->device_, sh),
                            std::move(sh), a->dtype_, a->device_));
    }
    return out;
}

}  // namespace lucid
