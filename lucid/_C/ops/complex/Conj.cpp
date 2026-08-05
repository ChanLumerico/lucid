// lucid/_C/ops/complex/Conj.cpp
//
// Forward implementation of conj_op.  The backend's ``complex_conj``
// already short-circuits real dtypes to identity; we just route through.

#include "Conj.h"

#include "../../backend/Dispatcher.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr conj_op(const TensorImplPtr& a) {
    Validator::input(a, "conj.a").non_null();
    OpScopeFull scope{"conj", a->device(), a->dtype(), a->shape()};

    Storage out = backend::Dispatcher::for_device(a->device())
                      .complex_conj(a->storage(), a->shape(), a->dtype());
    auto result = complex_detail::fresh(std::move(out), a->shape(), a->dtype(), a->device());

    // Wired for a real input too.
    //
    // The backend short-circuits a real dtype to the identity and hands
    // back the same storage, and skipping the node there looked harmless
    // — it is not.  ``fresh`` wraps that storage in a *new* TensorImpl
    // with no link to the old one, so the chain simply ended:
    // ``conj(x) * 2`` on a real ``x`` left ``x.grad`` at ``None``.  The
    // identity still has to carry a gradient.
    auto bwd = std::make_shared<ConjBackward>();
    bwd->shape_ = a->shape();
    bwd->dtype_ = a->dtype();
    bwd->device_ = a->device();
    kernel::NaryKernel<ConjBackward, 1>::wire_autograd(std::move(bwd), {a}, result, false);
    return result;
}

const OpSchema ConjBackward::schema_v1{"conj", 1, AmpPolicy::KeepInput, true};

std::vector<Storage> ConjBackward::apply(Storage grad_out) {
    // ``complex_conj`` is itself the identity on a real dtype, so this
    // one call covers both cases — conjugation is its own inverse.
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.complex_conj(grad_out, shape_, dtype_)};
}

LUCID_REGISTER_OP(ConjBackward)

}  // namespace lucid
