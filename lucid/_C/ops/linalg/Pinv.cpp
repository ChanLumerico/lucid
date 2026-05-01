#include "Pinv.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr pinv_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "pinv.a").non_null();
    require_float(a->dtype(), "pinv");
    if (a->shape().size() < 2)
        ErrorBuilder("pinv").fail("input must be at least 2-D");
    OpScopeFull scope{"pinv", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    Shape out_shape(sh.begin(), sh.end() - 2);
    out_shape.push_back(n);
    out_shape.push_back(m);

    Storage out =
        backend::Dispatcher::for_device(a->device()).linalg_pinv(a->storage(), sh, a->dtype());
    return fresh(std::move(out), std::move(out_shape), a->dtype(), a->device());
}

}  // namespace lucid
