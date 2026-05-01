#include "Eig.h"

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

std::vector<TensorImplPtr> eig_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "eig.a").non_null();
    require_float(a->dtype(), "eig");
    require_square_2d(a->shape(), "eig");
    OpScopeFull scope{"eig", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    Shape wsh(sh.begin(), sh.end() - 1);  // (..., n)
    Shape vsh = sh;                       // (..., n, n)
    auto out = backend::Dispatcher::for_device(a->device())
                   .linalg_eig(a->storage(), sh, wsh, vsh, a->dtype());
    std::vector<TensorImplPtr> result;
    result.push_back(fresh(std::move(out.first), wsh, a->dtype(), a->device()));
    result.push_back(fresh(std::move(out.second), vsh, a->dtype(), a->device()));
    return result;
}

}  // namespace lucid
