#include "Solve.h"

#include <variant>

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

const OpSchema SolveBackward::schema_v1{"solve", 1, AmpPolicy::KeepInput};

std::vector<Storage> SolveBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;

    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto dX = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    auto X = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);

    auto AT = mT_op(A);
    auto dB = solve_op(AT, dX);

    auto XT = mT_op(X);
    auto dA = neg_op(matmul_op(dB, XT));
    return {dA->storage(), dB->storage()};
}

LUCID_REGISTER_OP(SolveBackward)

TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    Validator::input(a, "solve.a").float_only().square_2d();
    Validator::pair(a, b, "solve").same_dtype().same_device();
    OpScopeFull scope{"solve", a->device(), a->dtype(), a->shape()};

    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
            .linalg_solve(a->storage(), b->storage(), a->shape(), b->shape(), a->dtype());
    auto out = linalg_detail::fresh(std::move(out_storage), b->shape(), a->dtype(), a->device());
    auto bwd = std::make_shared<SolveBackward>();
    bwd->saved_output_ = out->storage();
    kernel::NaryKernel<SolveBackward, 2>::wire_autograd(std::move(bwd), {a, b}, out, true);
    return out;
}

}  // namespace lucid
