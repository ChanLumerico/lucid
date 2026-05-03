// lucid/_C/ops/linalg/SVD.cpp
//
// Implementation of the Singular Value Decomposition forward op.
//
// Forward:
//   CPU path: LAPACK dgesdd performs a divide-and-conquer SVD, which is
//             typically faster than the QR-iteration based dgesvd for large
//             matrices but requires more temporary workspace.
//   GPU path: mlx::core::linalg::svd() on the CPU stream.
//
// Shape pre-computation: ush, ssh, and vsh are all derived before dispatch so
// the backend receives exact target sizes.  The convention is "Vh" (V-hermitian
// conjugate / transpose), not "V", to match NumPy and PyTorch naming.
//
// compute_uv flag: passing false to the backend skips materialising U and Vh,
// which can be significantly faster when only the singular values are needed
// (e.g. for computing the condition number or rank estimation).
//
// Autograd: not wired.  A future backward would use the Papadopoulo-Lourakis
// formula; see SVD.h for the derivation sketch.

#include "SVD.h"

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

// Compute the SVD of a.
//
// Shapes (ush, ssh, vsh) are derived before the dispatch call and passed to
// linalg_svd() so both sides agree on memory layout.  The backend returns a
// vector of Storage objects: one element when compute_uv=false, three when
// compute_uv=true (order: [U, S, Vh]).
std::vector<TensorImplPtr> svd_op(const TensorImplPtr& a, bool compute_uv) {
    using namespace linalg_detail;
    Validator::input(a, "svd.a").non_null();
    require_float(a->dtype(), "svd");
    if (a->shape().size() < 2)
        ErrorBuilder("svd").fail("input must be at least 2-D");
    OpScopeFull scope{"svd", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    // k = min(m, n): number of singular values in the reduced decomposition.
    const int k = std::min(m, n);

    // Pre-compute shapes; batch dims are copied from the leading dimensions.
    Shape ush(sh.begin(), sh.end() - 2);
    ush.push_back(m);  // U: m rows
    ush.push_back(k);  // U: k columns (left singular vectors)
    Shape ssh(sh.begin(), sh.end() - 2);
    ssh.push_back(k);  // S: k singular values per matrix
    Shape vsh(sh.begin(), sh.end() - 2);
    vsh.push_back(k);  // Vh: k rows (right singular vectors, transposed)
    vsh.push_back(n);  // Vh: n columns

    auto storages = backend::Dispatcher::for_device(a->device())
                        .linalg_svd(a->storage(), sh, compute_uv, ush, ssh, vsh, a->dtype());
    std::vector<TensorImplPtr> out;
    if (!compute_uv) {
        // Values-only mode: backend returns exactly one Storage (singular values).
        out.push_back(fresh(std::move(storages[0]), ssh, a->dtype(), a->device()));
        return out;
    }
    // Full SVD mode: backend returns [U storage, S storage, Vh storage].
    out.push_back(fresh(std::move(storages[0]), ush, a->dtype(), a->device()));  // U
    out.push_back(fresh(std::move(storages[1]), ssh, a->dtype(), a->device()));  // S
    out.push_back(fresh(std::move(storages[2]), vsh, a->dtype(), a->device()));  // Vh
    return out;
}

}  // namespace lucid
