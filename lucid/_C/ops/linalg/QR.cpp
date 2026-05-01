#include "QR.h"

#include <algorithm>
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

std::vector<TensorImplPtr> qr_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "qr.a").non_null();
    require_float(a->dtype(), "qr");
    if (a->shape().size() < 2)
        ErrorBuilder("qr").fail("input must be at least 2-D");
    OpScopeFull scope{"qr", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const int k = std::min(m, n);

    Shape qsh(sh.begin(), sh.end() - 2);
    qsh.push_back(m);
    qsh.push_back(k);
    Shape rsh(sh.begin(), sh.end() - 2);
    rsh.push_back(k);
    rsh.push_back(n);

    auto out = backend::Dispatcher::for_device(a->device())
                   .linalg_qr(a->storage(), sh, qsh, rsh, a->dtype());
    std::vector<TensorImplPtr> result;
    result.push_back(fresh(std::move(out.first), qsh, a->dtype(), a->device()));
    result.push_back(fresh(std::move(out.second), rsh, a->dtype(), a->device()));
    return result;
}

}  // namespace lucid
