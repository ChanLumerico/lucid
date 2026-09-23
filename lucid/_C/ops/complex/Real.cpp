// lucid/_C/ops/complex/Real.cpp
//
// Forward implementation of real_op.  Validates the input is C64, then
// dispatches through ``IBackend::complex_real`` (CPU = stride-2 walk,
// GPU = ``mlx::core::real``).

#include "Real.h"

#include "../../backend/Dispatcher.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr real_op(const TensorImplPtr& a) {
    Validator::input(a, "real.a").non_null();
    complex_detail::require_complex(a->dtype(), "real");
    // The scope's dtype is the output's: this strips a lane, so it is
    // the real one, not the complex input's.
    OpScopeFull scope{"real", a->device(), real_lane_of(a->dtype()), a->shape()};

    Storage out =
        backend::Dispatcher::for_device(a->device()).complex_real(a->storage(), a->shape());
    auto result =
        complex_detail::fresh(std::move(out), a->shape(), real_lane_of(a->dtype()), a->device());

    auto bwd = std::make_shared<RealBackward>();
    bwd->shape_ = a->shape();
    bwd->lane_ = real_lane_of(a->dtype());
    bwd->device_ = a->device();
    kernel::NaryKernel<RealBackward, 1>::wire_autograd(std::move(bwd), {a}, result, false);
    return result;
}

const OpSchema RealBackward::schema_v1{"real", 1, AmpPolicy::KeepInput, true};

std::vector<Storage> RealBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    auto zero = zeros_op(shape_, lane_, device_);
    return {be.complex_combine(grad_out, zero->storage(), shape_)};
}

LUCID_REGISTER_OP(RealBackward)

}  // namespace lucid
