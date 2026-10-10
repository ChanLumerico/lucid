// lucid/_C/ops/bfunc/Floordiv.cpp
//
// Implements floordiv_op.  Floating-point operands are ``floor(a / b)`` in
// their own dtype; integer operands keep their dtype too.

#include "Floordiv.h"

#include <cmath>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
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
#include "_Opmath.h"

namespace lucid {

namespace {

using bfunc_detail::broadcast_pair;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

}  // namespace

TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (auto r = opmath::try_scalar<opmath::Op::FloorDiv>(a, b, &floordiv_op))
        return r;
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
    // carries NaN and Inf through untouched.  bfloat16 is one of them; left
    // off this list, it took the integer path and came back int64.
    if (dt == Dtype::F16 || dt == Dtype::BF16 || dt == Dtype::F32 || dt == Dtype::F64) {
        return floor_op(div_op(a, b));
    }
    // Refused here rather than in the backend, which the Metal stream let
    // through: it cast bool and complex to float32 and answered in int64.
    if (!is_integral(dt))
        ErrorBuilder("floordiv")
            .not_implemented(
                std::string("dtype ") + std::string(dtype_name(dt)) +
                " is not supported; floor division takes integer or real floating-point operands");

    auto bc = broadcast_pair(a, b);
    OpScopeFull scope{"floordiv", device, dt, bc.shape};

    // The backend divides int32 and int64 and always answers in int64.  An
    // integer quotient keeps its operands' dtype, as Python's ``//`` and the
    // reference do, so narrower operands are widened on the way in and the
    // quotient narrowed on the way out.  The narrowing wraps like any
    // integer cast: int8 ``-128 // -1`` is -128, as in the reference.
    auto& be = backend::Dispatcher::for_device(device);
    const bool widen = dt == Dtype::I8 || dt == Dtype::I16;
    Storage quotient = widen ? be.floordiv(be.astype(bc.a->storage(), bc.shape, dt, Dtype::I32),
                                           be.astype(bc.b->storage(), bc.shape, dt, Dtype::I32),
                                           bc.shape, Dtype::I32)
                             : be.floordiv(bc.a->storage(), bc.b->storage(), bc.shape, dt);
    if (dt != Dtype::I64)
        quotient = be.astype(quotient, bc.shape, Dtype::I64, dt);
    auto result = fresh(std::move(quotient), bc.shape, dt, device);
    // Integer floor-division is non-differentiable, so it never reaches
    // ``wire_autograd`` — which is also what records a traced op's
    // operands.  Without this the op lands in the trace with no inputs,
    // and ``lucid.compile`` refuses the whole graph rather than bake a
    // value it cannot prove constant ("did not record its trace I/O").
    // Three-axis RoPE hits exactly this: ``ids // (height * width)``
    // is how every V-JEPA 2 / V-JEPA block derives its depth index, so
    // the entire family fell back to eager.  The floating-point branch
    // above needs no such call — ``floor_op`` and ``div_op`` are kernel
    // ops that wire themselves.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a, b}, result);
    }
    return result;
}

}  // namespace lucid
