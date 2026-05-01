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
    const int k = std::min(m, n);

    Shape ush(sh.begin(), sh.end() - 2);
    ush.push_back(m);
    ush.push_back(k);
    Shape ssh(sh.begin(), sh.end() - 2);
    ssh.push_back(k);
    Shape vsh(sh.begin(), sh.end() - 2);
    vsh.push_back(k);
    vsh.push_back(n);

    auto storages = backend::Dispatcher::for_device(a->device())
                        .linalg_svd(a->storage(), sh, compute_uv, ush, ssh, vsh, a->dtype());
    std::vector<TensorImplPtr> out;
    if (!compute_uv) {
        out.push_back(fresh(std::move(storages[0]), ssh, a->dtype(), a->device()));
        return out;
    }
    out.push_back(fresh(std::move(storages[0]), ush, a->dtype(), a->device()));
    out.push_back(fresh(std::move(storages[1]), ssh, a->dtype(), a->device()));
    out.push_back(fresh(std::move(storages[2]), vsh, a->dtype(), a->device()));
    return out;
}

}  // namespace lucid
