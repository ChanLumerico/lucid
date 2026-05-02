#include "Floordiv.h"

#include <cmath>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::fresh;
using bfunc_detail::validate_pair_eq_shape;

}  // namespace

TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair_eq_shape(a, b, "floordiv");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"floordiv", device, dt, a->shape()};

    auto out_storage = backend::Dispatcher::for_device(device).floordiv(a->storage(), b->storage(),
                                                                        a->shape(), dt);
    return fresh(std::move(out_storage), a->shape(), Dtype::I64, device);
}

}  // namespace lucid
