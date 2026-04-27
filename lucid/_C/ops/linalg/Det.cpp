#include "Det.h"

#include <variant>

#include <mlx/linalg.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr det_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    if (!a) throw LucidError("det: null input");
    OpScope scope{"det", a->device_, a->dtype_, a->shape_};
    auto in = as_mlx_array(a);
    // MLX has no direct det. Compute via LU: det(A) = prod(diag(U)).
    auto factors = ::mlx::core::linalg::lu(in, kMlxCpu);
    if (factors.size() < 3)
        throw LucidError("det: lu returned fewer than 3 factors");
    const auto& U = factors[2];
    if (U.shape().size() < 2)
        throw LucidError("det: factorization shape unexpected");
    auto diag = ::mlx::core::diagonal(U, 0, -2, -1);
    auto detU = ::mlx::core::prod(diag, /*keepdims=*/false);
    Shape out_shape = mlx_shape_to_lucid(detU.shape());
    return fresh(wrap_result(std::move(detU), a->dtype_, a->device_, out_shape),
                 std::move(out_shape), a->dtype_, a->device_);
}

}  // namespace lucid
