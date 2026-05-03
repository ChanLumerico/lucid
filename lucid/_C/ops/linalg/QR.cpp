// lucid/_C/ops/linalg/QR.cpp
//
// Implementation of the QR decomposition forward op.
//
// Forward:
//   CPU path: LAPACK dgeqrf produces Householder vectors packed into the lower
//             triangle of the input; dorgqr then materialises Q explicitly.
//   GPU path: mlx::core::linalg::qr() on the CPU stream.
//
// Shape pre-computation: both Q and R shapes are derived before the dispatch
// call and passed to linalg_qr() so the backend can write into pre-sized
// buffers.  Specifically:
//   Q has shape (..., m, k), k = min(m, n)
//   R has shape (..., k, n)
// where the "..." prefix is any batch dimensions copied from the input shape.
//
// Autograd: not wired.  The output TensorImpls are leaf nodes in the gradient
// graph.  A future backward would use Bettale's QR gradient formula (see
// QR.h for the derivation sketch).

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

// Compute the reduced QR factorisation of a.
//
// The backend receives the pre-computed output shapes (qsh, rsh) so it can
// allocate exactly the right amount of memory for Q and R.  The shapes follow
// the "thin" convention: k = min(m, n) caps the inner dimension.
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
    // k = min(m, n): the rank of the reduced factorisation.
    const int k = std::min(m, n);

    // Batch dimensions are copied from the input shape (all dims but last two).
    Shape qsh(sh.begin(), sh.end() - 2);
    qsh.push_back(m);  // Q has m rows (same as A)
    qsh.push_back(k);  // Q has k columns (orthonormal basis)
    Shape rsh(sh.begin(), sh.end() - 2);
    rsh.push_back(k);  // R has k rows
    rsh.push_back(n);  // R has n columns (same as A)

    // linalg_qr returns a pair<Storage, Storage> for {Q, R}.
    auto out = backend::Dispatcher::for_device(a->device())
                   .linalg_qr(a->storage(), sh, qsh, rsh, a->dtype());
    std::vector<TensorImplPtr> result;
    result.push_back(fresh(std::move(out.first), qsh, a->dtype(), a->device()));
    result.push_back(fresh(std::move(out.second), rsh, a->dtype(), a->device()));
    return result;
}

}  // namespace lucid
