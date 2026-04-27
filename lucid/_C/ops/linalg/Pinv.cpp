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
    OpScope scope{"pinv", a->device_, a->dtype_, a->shape_};
    auto in = as_mlx_array(a);
    auto out = ::mlx::core::linalg::pinv(in, kMlxCpu);
    Shape sh = mlx_shape_to_lucid(out.shape());
    return fresh(wrap_result(std::move(out), a->dtype_, a->device_, sh),
                 std::move(sh), a->dtype_, a->device_);
}

}  // namespace lucid
