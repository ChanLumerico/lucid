// lucid/_C/ops/linalg/LUFactor.cpp
//
// Implements lu_factor_op: LU factorisation with partial pivoting via
// IBackend::linalg_lu_factor() → LAPACK sgetrf_/dgetrf_.

#include "LUFactor.h"

#include <algorithm>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {

std::vector<TensorImplPtr> lu_factor_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "lu_factor.a").float_only();
    if (a->shape().size() < 2)
        ErrorBuilder("lu_factor.a").fail("input must be at least 2-D");

    const auto& sh = a->shape();
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);

    // LU with partial pivoting is defined for any shape — ``?getrf``
    // factorises an m-by-n matrix into ``P L U`` with L unit-lower
    // m-by-min(m,n) and U upper min(m,n)-by-n.  This op required a square
    // input, which was a restriction of the wrapper rather than of the
    // factorisation: the LAPACK call underneath was passing the same
    // extent for both dimensions.
    //
    // One pivot per elimination step, and there are only as many steps as
    // the shorter side.
    Shape pivot_shape(sh.begin(), sh.end() - 2);
    pivot_shape.push_back(std::min(m, n));

    // A degenerate matrix has a decomposition; LAPACK just will not be
    // the one to compute it.  See ``empty_matrix`` for why dispatching
    // wrote to a stream Lucid does not own and then raised.
    if (empty_matrix(sh))
        return {zeros_op(sh, a->dtype(), a->device()),
                zeros_op(pivot_shape, Dtype::I32, a->device())};

    auto [lu_storage, ipiv_storage] =
        backend::Dispatcher::for_device(a->device()).linalg_lu_factor(a->storage(), sh, a->dtype());

    auto lu = fresh(std::move(lu_storage), sh, a->dtype(), a->device());
    auto pivots = fresh(std::move(ipiv_storage), pivot_shape, Dtype::I32, a->device());
    return {lu, pivots};
}

}  // namespace lucid
