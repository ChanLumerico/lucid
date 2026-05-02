#include "Eigh.h"

#include <variant>
#include <vector>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

std::vector<TensorImplPtr> eigh_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "eigh.a").non_null();
    require_float(a->dtype(), "eigh");
    require_square_2d(a->shape(), "eigh");
    OpScopeFull scope{"eigh", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    Shape wsh(sh.begin(), sh.end() - 1);  // eigenvalues: (..., n)
    Shape vsh = sh;                       // eigenvectors: (..., n, n)

    auto out = backend::Dispatcher::for_device(a->device())
                   .linalg_eigh(a->storage(), sh, wsh, vsh, a->dtype());
    return {fresh(std::move(out.first),  wsh, a->dtype(), a->device()),
            fresh(std::move(out.second), vsh, a->dtype(), a->device())};
}

}  // namespace lucid
