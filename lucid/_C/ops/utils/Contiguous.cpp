#include "Contiguous.h"

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema ContiguousBackward::schema_v1{"contiguous", 1, AmpPolicy::KeepInput, true};

TensorImplPtr ContiguousBackward::forward(const TensorImplPtr& a) {
    Validator::input(a, "contiguous.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};

    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .contiguous(a->storage(), a->shape(), a->stride(),
                                          a->storage_offset(), a->is_contiguous(), a->dtype());

    auto result = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                               a->device(), false);

    kernel::NaryKernel<ContiguousBackward, 1>::wire_autograd({a}, result, false);
    return result;
}

std::vector<Storage> ContiguousBackward::apply(Storage grad_out) {
    return {clone_storage(grad_out, shape_numel(out_shape_), dtype_, device_)};
}

TensorImplPtr contiguous_op(const TensorImplPtr& a) {
    return ContiguousBackward::forward(a);
}

LUCID_REGISTER_OP(ContiguousBackward)

}  // namespace lucid
