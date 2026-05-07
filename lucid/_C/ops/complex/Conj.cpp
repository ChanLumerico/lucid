// lucid/_C/ops/complex/Conj.cpp
//
// Forward implementation of conj_op.  The backend's ``complex_conj``
// already short-circuits real dtypes to identity; we just route through.

#include "Conj.h"

#include "../../backend/Dispatcher.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr conj_op(const TensorImplPtr& a) {
    Validator::input(a, "conj.a").non_null();
    OpScopeFull scope{"conj", a->device(), a->dtype(), a->shape()};

    Storage out = backend::Dispatcher::for_device(a->device())
                      .complex_conj(a->storage(), a->shape(), a->dtype());
    return complex_detail::fresh(std::move(out), a->shape(), a->dtype(), a->device());
}

}  // namespace lucid
