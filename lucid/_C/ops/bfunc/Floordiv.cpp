// lucid/_C/ops/bfunc/Floordiv.cpp
//
// Implements floordiv_op.  The backend floor-division primitive handles both
// integer and floating-point inputs; the result is always returned as I64.

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
#include "_Broadcast.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::broadcast_pair;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

}  // namespace

TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair(a, b, "floordiv");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    auto bc = broadcast_pair(a, b);
    OpScopeFull scope{"floordiv", device, dt, bc.shape};

    auto out_storage = backend::Dispatcher::for_device(device).floordiv(
        bc.a->storage(), bc.b->storage(), bc.shape, dt);
    // The output is always I64 regardless of the input dtype so that the result
    // type is consistent with Python's // operator semantics.
    return fresh(std::move(out_storage), bc.shape, Dtype::I64, device);
}

}  // namespace lucid
