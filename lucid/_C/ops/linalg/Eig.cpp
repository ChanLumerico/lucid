// lucid/_C/ops/linalg/Eig.cpp
//
// Implementation of the general eigendecomposition forward op.
//
// Forward:
//   CPU path: LAPACK dgeev computes both the right eigenvectors and the
//             eigenvalues of a general (non-symmetric) real matrix.  For
//             matrices with complex eigenvalues dgeev returns the real and
//             imaginary parts in separate vectors; the current binding returns
//             only the real part, which is correct only for real-symmetric A.
//   GPU path: mlx::core::linalg::eig() on the CPU stream.
//
// Shapes for the two outputs are derived before dispatch:
//   w (eigenvalues):  (..., n)    — one eigenvalue per column of A
//   V (eigenvectors): (..., n, n) — same shape as A; columns are eigenvectors
//
// Autograd is not wired here.  The general eigenvector gradient is numerically
// sensitive near degenerate eigenvalues and requires complex arithmetic when
// eigenvalues are not real; this is deferred to a future phase.

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

// Compute eigenvalues and right eigenvectors of a.
//
// wsh drops the last dimension (the redundant n in n×n).
// vsh is identical to the input shape (the full n×n eigenvector matrix).
std::vector<TensorImplPtr> eig_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "eig.a").non_null();
    require_float(a->dtype(), "eig");
    require_square_2d(a->shape(), "eig");
    OpScopeFull scope{"eig", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    // Eigenvalue shape: remove the last dim (one eigenvalue per row/column).
    Shape wsh(sh.begin(), sh.end() - 1);
    Shape vsh = sh;
    auto out = backend::Dispatcher::for_device(a->device())
                   .linalg_eig(a->storage(), sh, wsh, vsh, a->dtype());
    std::vector<TensorImplPtr> result;
    result.push_back(fresh(std::move(out.first), wsh, a->dtype(), a->device()));
    result.push_back(fresh(std::move(out.second), vsh, a->dtype(), a->device()));
    return result;
}

}  // namespace lucid
