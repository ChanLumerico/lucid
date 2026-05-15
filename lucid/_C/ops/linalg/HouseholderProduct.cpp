// lucid/_C/ops/linalg/HouseholderProduct.cpp
#include "HouseholderProduct.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"
namespace lucid {

TensorImplPtr householder_product_op(const TensorImplPtr& H, const TensorImplPtr& tau) {
    using namespace linalg_detail;
    Validator::input(H, "householder_product.H").float_only().non_null();
    Validator::input(tau, "householder_product.tau").float_only().non_null();

    const auto& hs = H->shape();
    const int m = static_cast<int>(hs[hs.size() - 2]);
    const int n = static_cast<int>(hs[hs.size() - 1]);
    const int k = std::min(m, n);

    auto result = backend::Dispatcher::for_device(H->device())
                      .linalg_householder_product(H->storage(), tau->storage(), hs, H->dtype());

    Shape q_shape = {static_cast<std::int64_t>(m), static_cast<std::int64_t>(k)};
    return fresh(std::move(result), q_shape, H->dtype(), H->device());
}

}  // namespace lucid
