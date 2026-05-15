// lucid/_C/ops/complex/Imag.cpp
//
// Forward implementation of imag_op.  Mirrors real_op but pulls the
// imaginary halves out of the interleaved C64 storage.

#include "Imag.h"

#include "../../backend/Dispatcher.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr imag_op(const TensorImplPtr& a) {
    Validator::input(a, "imag.a").non_null();
    complex_detail::require_complex(a->dtype(), "imag");
    OpScopeFull scope{"imag", a->device(), a->dtype(), a->shape()};

    Storage out =
        backend::Dispatcher::for_device(a->device()).complex_imag(a->storage(), a->shape());
    return complex_detail::fresh(std::move(out), a->shape(), Dtype::F32, a->device());
}

}  // namespace lucid
