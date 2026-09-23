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
#include "../ufunc/Discrete.h"  // floor_op
#include "Div.h"                // div_op
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

    // Floating-point operands keep their own dtype, as Python's ``//`` and
    // the reference framework both do — ``7.0 // 2.0`` is ``3.0``, not
    // ``3``.  The integer result this used to force was not merely a
    // different spelling of the same number: converting a float to I64
    // destroyed every non-finite value, so ``nan // 1`` came back as 0 and
    // ``inf // 1`` as INT64_MAX.  A poisoned tensor turned into ordinary
    // numbers with nothing to show for it, which is the failure mode a
    // NaN is supposed to make loud.  floor(a / b) has neither problem and
    // carries NaN and Inf through untouched.
    if (dt == Dtype::F16 || dt == Dtype::F32 || dt == Dtype::F64) {
        return floor_op(div_op(a, b));
    }

    auto bc = broadcast_pair(a, b);
    OpScopeFull scope{"floordiv", device, dt, bc.shape};

    auto out_storage = backend::Dispatcher::for_device(device).floordiv(
        bc.a->storage(), bc.b->storage(), bc.shape, dt);
    return fresh(std::move(out_storage), bc.shape, Dtype::I64, device);
}

}  // namespace lucid
