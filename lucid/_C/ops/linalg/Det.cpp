#include "Det.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/ufunc/Transpose.h"
#include "../../ops/utils/Layout.h"
#include "Inv.h"
#include "_Detail.h"

namespace lucid {

const OpSchema DetBackward::schema_v1{"det", 1, AmpPolicy::KeepInput};

std::vector<Storage> DetBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;

    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto ddet = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    auto det_v = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);

    auto inv_A = inv_op(A);
    auto inv_AT = mT_op(inv_A);
    auto scale = mul_op(det_v, ddet);
    auto dA = mul_op(broadcast_to_op(scale, input_shapes_[0]), inv_AT);
    return {dA->storage()};
}

LUCID_REGISTER_OP(DetBackward)

TensorImplPtr det_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "det.a").float_only().square_2d();
    OpScopeFull scope{"det", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    Shape out_shape(sh.begin(), sh.end() - 2);

    Storage out_storage =
        backend::Dispatcher::for_device(a->device()).linalg_det(a->storage(), sh, a->dtype());
    auto out = fresh(std::move(out_storage), out_shape, a->dtype(), a->device());
    auto bwd = std::make_shared<DetBackward>();
    bwd->saved_output_ = out->storage();
    kernel::NaryKernel<DetBackward, 1>::wire_autograd(std::move(bwd), {a}, out, true);
    return out;
}

}  // namespace lucid
