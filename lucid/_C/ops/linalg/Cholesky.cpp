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
    OpScope scope{"cholesky", a->device_, a->dtype_, a->shape_};
    auto in = as_mlx_array(a);
    auto out = ::mlx::core::linalg::cholesky(in, upper, kMlxCpu);
    return fresh(wrap_result(std::move(out), a->dtype_, a->device_, a->shape_),
                 a->shape_, a->dtype_, a->device_);
}

}  // namespace lucid
