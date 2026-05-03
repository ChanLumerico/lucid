// lucid/_C/ops/linalg/LUFactor.cpp
//
// Implements lu_factor_op: LU factorisation with partial pivoting via
// IBackend::linalg_lu_factor() → LAPACK sgetrf_/dgetrf_.

#include "LUFactor.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

std::vector<TensorImplPtr> lu_factor_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "lu_factor.a").float_only().square_2d();

    const auto& sh = a->shape();
    const int n = static_cast<int>(sh[sh.size() - 1]);

    // Shape of the pivot tensor: same batch dims as A, plus length n.
    Shape pivot_shape(sh.begin(), sh.end() - 2);
    pivot_shape.push_back(n);

    auto [lu_storage, ipiv_storage] =
        backend::Dispatcher::for_device(a->device()).linalg_lu_factor(
            a->storage(), sh, a->dtype());

    auto lu = fresh(std::move(lu_storage), sh, a->dtype(), a->device());
    auto pivots = fresh(std::move(ipiv_storage), pivot_shape, Dtype::I32, a->device());
    return {lu, pivots};
}

}  // namespace lucid
