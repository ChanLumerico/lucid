// lucid/_C/ops/utils/Flip.cpp
#include "Flip.h"
#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
namespace lucid {

TensorImplPtr flip_op(const TensorImplPtr& a, std::vector<int> dims) {
    Validator::input(a, "flip").non_null();
    const auto& sh = a->shape();
    const int ndim = static_cast<int>(sh.size());

    // Normalise negative dims and deduplicate.
    for (auto& d : dims) {
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim)
            ErrorBuilder("flip").fail("dim out of range");
    }

    OpScopeFull scope{"flip", a->device(), a->dtype(), sh};
    auto& be = backend::Dispatcher::for_device(a->device());
    Storage out = be.flip(a->storage(), sh, dims, a->dtype());
    return std::make_shared<TensorImpl>(std::move(out), sh, a->dtype(), a->device(), false);
}

}  // namespace lucid
