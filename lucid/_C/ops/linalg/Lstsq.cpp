// lucid/_C/ops/linalg/Lstsq.cpp
#include "Lstsq.h"
#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"
namespace lucid {

std::vector<TensorImplPtr> lstsq_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    using namespace linalg_detail;
    Validator::input(a, "lstsq.a").float_only().non_null();
    Validator::input(b, "lstsq.b").float_only().non_null();

    const auto& as = a->shape();
    const auto& bs = b->shape();
    const int n = static_cast<int>(as[as.size() - 1]);
    const int nrhs = (bs.size() > 1) ? static_cast<int>(bs[bs.size() - 1]) : 1;

    auto results = backend::Dispatcher::for_device(a->device()).linalg_lstsq(
        a->storage(), b->storage(), as, bs, a->dtype());

    // Solution shape: (n, nrhs) or (n,) if nrhs==1
    Shape sol_shape = {static_cast<std::int64_t>(n)};
    if (nrhs > 1) sol_shape.push_back(static_cast<std::int64_t>(nrhs));

    auto sol = fresh(std::move(results[0]), sol_shape, a->dtype(), a->device());
    return {sol};
}

}  // namespace lucid
