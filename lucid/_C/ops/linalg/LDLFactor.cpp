// lucid/_C/ops/linalg/LDLFactor.cpp
#include "LDLFactor.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"
namespace lucid {

std::vector<TensorImplPtr> ldl_factor_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "ldl_factor.a").float_only().square_2d();

    const auto& sh = a->shape();
    const int n = static_cast<int>(sh[sh.size() - 1]);

    auto [ld_storage, piv_storage] = backend::Dispatcher::for_device(a->device())
                                         .linalg_ldl_factor(a->storage(), sh, a->dtype());

    Shape piv_shape(sh.begin(), sh.end() - 2);
    piv_shape.push_back(static_cast<std::int64_t>(n));

    auto ld = fresh(std::move(ld_storage), sh, a->dtype(), a->device());
    auto piv = fresh(std::move(piv_storage), piv_shape, Dtype::I32, a->device());
    return {ld, piv};
}

}  // namespace lucid
