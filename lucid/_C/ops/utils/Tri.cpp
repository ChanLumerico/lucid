#include "Tri.h"

#include <variant>

#include "../../autograd/FuncOp.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
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
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;

Storage tri_storage(const Storage& input,
                    const Shape& shape,
                    Dtype dt,
                    Device device,
                    int k,
                    bool upper,
                    const char*) {
    return backend::Dispatcher::for_device(device).tri(input, shape, dt, k, upper);
}

class TriBackward : public FuncOp<TriBackward, 1> {
public:
    static const OpSchema schema_v1;

    int k_ = 0;
    bool upper_ = false;
    const char* name_ = "tril";

    std::vector<Storage> apply(Storage grad_out) override {
        return {tri_storage(grad_out, out_shape_, dtype_, device_, k_, upper_, name_)};
    }
};

const OpSchema TriBackward::schema_v1{"tri", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, true};

TensorImplPtr
attach_tri_grad(const TensorImplPtr& a, TensorImplPtr out, int k, bool upper, const char* name) {
    auto bwd = std::make_shared<TriBackward>();
    bwd->k_ = k;
    bwd->upper_ = upper;
    bwd->name_ = name;
    kernel::NaryKernel<TriBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

TensorImplPtr tri_dispatch(const TensorImplPtr& a, int k, bool upper, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{name, device, dt, a->shape()};
    Shape sh = a->shape();
    auto out_storage = tri_storage(a->storage(), sh, dt, device, k, upper, name);
    auto out = fresh(std::move(out_storage), std::move(sh), dt, device);
    return attach_tri_grad(a, std::move(out), k, upper, name);
}

}  // namespace

TensorImplPtr tril_op(const TensorImplPtr& a, int k) {
    return tri_dispatch(a, k, false, "tril");
}
TensorImplPtr triu_op(const TensorImplPtr& a, int k) {
    return tri_dispatch(a, k, true, "triu");
}

LUCID_REGISTER_OP(TriBackward)

}  // namespace lucid
