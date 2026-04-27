#include "Solve.h"

#include <variant>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    using namespace linalg_detail;
    if (!a || !b) throw LucidError("solve: null input");
    if (a->device_ != b->device_)
        throw DeviceMismatch(std::string(device_name(a->device_)),
                              std::string(device_name(b->device_)), "solve");
    OpScope scope{"solve", a->device_, a->dtype_, a->shape_};
    auto in_a = as_mlx_array(a);
    auto in_b = as_mlx_array(b);
    auto out = ::mlx::core::linalg::solve(in_a, in_b, kMlxCpu);
    Shape sh = mlx_shape_to_lucid(out.shape());
    return fresh(wrap_result(std::move(out), a->dtype_, a->device_, sh),
                 std::move(sh), a->dtype_, a->device_);
}

}  // namespace lucid
