#include "Inv.h"

#include <variant>

#include "../../autograd/FuncOp.h"
#include "../../backend/Dispatcher.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Matmul.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Transpose.h"
#include "_Detail.h"

namespace lucid {

// ---------- Schema & backward ----------

const OpSchema InvBackward::schema_v1{"inv", 1, AmpPolicy::KeepInput};

std::vector<Storage> InvBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    // saved_output_ = B = inv(A)
    auto B = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);
    auto dB = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    // dA = -(B^T @ dB @ B^T)
    auto Bt = mT_op(B);
    auto dA = neg_op(matmul_op(matmul_op(Bt, dB), Bt));
    return {dA->storage()};
}

LUCID_REGISTER_OP(InvBackward)

// ---------- Forward ----------

TensorImplPtr inv_op(const TensorImplPtr& a) {
    Validator::input(a, "inv.a").float_only().square_2d();
    OpScopeFull scope{"inv", a->device(), a->dtype(), a->shape()};

    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .linalg_inv(a->storage(), a->shape(), a->dtype());
    auto out = linalg_detail::fresh(std::move(out_storage), a->shape(), a->dtype(), a->device());
    auto bwd = std::make_shared<InvBackward>();
    bwd->saved_output_ = out->storage();
    kernel::NaryKernel<InvBackward, 1>::wire_autograd(std::move(bwd), {a}, out, /*save_ins=*/false);
    return out;
}

}  // namespace lucid
