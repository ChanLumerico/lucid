#include "Scan.h"

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

// Reverse a Storage along `axis` (used by cumsum backward).
Storage reverse_along_axis_storage(
    const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    return backend::Dispatcher::for_device(device).reverse_along_axis(s, shape, axis, dt);
}

Storage cumsum_storage_along(
    const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    return backend::Dispatcher::for_device(device).cumsum(s, shape, axis, dt);
}

class CumsumBackward : public AutogradNode<CumsumBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;

    std::vector<Storage> apply(Storage grad_out) override {
        Storage rev = reverse_along_axis_storage(grad_out, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage dx = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        return {std::move(dx)};
    }
};

// cumprod backward.
//   y_k = x_0 * x_1 * ... * x_k
//   dx_j = sum_{k >= j} (g_k * y_k) / x_j
//        = reverse_cumsum(g * y)_j / x_j
// We materialize this via:
//   1. forward y = cumprod(x, axis)            (saved at forward time)
//   2. p = g * y
//   3. q = reverse_cumsum(p, axis)
//   4. dx = q / x   (NaN at zeros — same convention as PyTorch)
class CumprodBackward : public AutogradNode<CumprodBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;
    Storage saved_x_;
    Storage saved_y_;

    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t total = shape_numel(input_shape_);
        Storage p_s = multiply_storages(grad_out, saved_y_, total, dtype_, device_);
        Storage rev = reverse_along_axis_storage(p_s, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage q = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        Storage dx = divide_storages(q, saved_x_, total, dtype_, device_);
        return {std::move(dx)};
    }
};

const OpSchema CumsumBackward::schema_v1{"cumsum", 1, AmpPolicy::KeepInput, true};
const OpSchema CumprodBackward::schema_v1{"cumprod", 1, AmpPolicy::KeepInput, true};

TensorImplPtr scan_dispatch(const TensorImplPtr& a, int axis, bool is_prod, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    auto sh = a->shape();
    if (sh.empty())
        ErrorBuilder(name).fail("input is scalar");
    int ax = axis;
    if (ax < 0)
        ax += static_cast<int>(sh.size());
    if (ax < 0 || ax >= (int)sh.size())
        ErrorBuilder(name).fail("axis out of range");
    OpScopeFull scope{name, device, dt, sh};

    Storage out_storage = is_prod
                              ? backend::Dispatcher::for_device(device).cumprod(a->storage(), sh, ax, dt)
                              : backend::Dispatcher::for_device(device).cumsum(a->storage(), sh, ax, dt);
    return fresh(std::move(out_storage), sh, dt, device);
}

}  // namespace

TensorImplPtr cumsum_op(const TensorImplPtr& a, int axis) {
    auto out = scan_dispatch(a, axis, /*is_prod=*/false, "cumsum");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumsumBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    kernel::NaryKernel<CumsumBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                         /*save_ins=*/false);
    return out;
}

TensorImplPtr cumprod_op(const TensorImplPtr& a, int axis) {
    auto out = scan_dispatch(a, axis, /*is_prod=*/true, "cumprod");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumprodBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->saved_x_ = a->storage();
    bwd->saved_y_ = out->storage();
    kernel::NaryKernel<CumprodBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                          /*save_ins=*/false);
    return out;
}

}  // namespace lucid
