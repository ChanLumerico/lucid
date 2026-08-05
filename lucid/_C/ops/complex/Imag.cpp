// lucid/_C/ops/complex/Imag.cpp
//
// Forward implementation of imag_op.  Mirrors real_op but pulls the
// imaginary halves out of the interleaved C64 storage.

#include "Imag.h"

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

TensorImplPtr imag_op(const TensorImplPtr& a) {
    Validator::input(a, "imag.a").non_null();
    complex_detail::require_complex(a->dtype(), "imag");
    OpScopeFull scope{"imag", a->device(), a->dtype(), a->shape()};

    Storage out =
        backend::Dispatcher::for_device(a->device()).complex_imag(a->storage(), a->shape());
    auto result =
        complex_detail::fresh(std::move(out), a->shape(), real_lane_of(a->dtype()), a->device());

    auto bwd = std::make_shared<ImagBackward>();
    bwd->shape_ = a->shape();
    bwd->lane_ = real_lane_of(a->dtype());
    bwd->device_ = a->device();
    kernel::NaryKernel<ImagBackward, 1>::wire_autograd(std::move(bwd), {a}, result, false);
    return result;
}

const OpSchema ImagBackward::schema_v1{"imag", 1, AmpPolicy::KeepInput, true};

std::vector<Storage> ImagBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    auto zero = zeros_op(shape_, lane_, device_);
    return {be.complex_combine(zero->storage(), grad_out, shape_)};
}

LUCID_REGISTER_OP(ImagBackward)

}  // namespace lucid
