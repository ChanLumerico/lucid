#include "Trace.h"

#include <algorithm>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::fresh;

class TraceBackward : public AutogradNode<TraceBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {backend::Dispatcher::for_device(device_).trace_backward(grad_out, input_shape_,
                                                                        dtype_)};
    }
};

const OpSchema TraceBackward::schema_v1{"trace", 1, AmpPolicy::KeepInput, true};

}  // namespace

TensorImplPtr trace_op(const TensorImplPtr& a) {
    Validator::input(a, "trace.a").non_null();
    if (a->shape().size() < 2)
        ErrorBuilder("trace").fail("input must have ndim >= 2");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    const auto& sh = a->shape();
    OpScopeFull scope{"trace", device, dt, Shape{}};

    Shape out_shape(sh.begin() + 2, sh.end());
    auto out_storage = backend::Dispatcher::for_device(device).trace(a->storage(), a->shape(), dt);
    TensorImplPtr out = fresh(std::move(out_storage), out_shape, dt, device);

    if (a->shape().size() == 2) {
        auto bwd = std::make_shared<TraceBackward>();
        bwd->input_shape_ = a->shape();
        kernel::NaryKernel<TraceBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

}  // namespace lucid
