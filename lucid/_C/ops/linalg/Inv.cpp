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
    OpScope scope{"inv", a->device_, a->dtype_, a->shape_};
    auto in = as_mlx_array(a);
    auto out = ::mlx::core::linalg::inv(in, kMlxCpu);
    return fresh(wrap_result(std::move(out), a->dtype_, a->device_, a->shape_),
                 a->shape_, a->dtype_, a->device_);
}

}  // namespace lucid
