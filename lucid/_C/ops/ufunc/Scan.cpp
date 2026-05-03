// lucid/_C/ops/ufunc/Scan.cpp
//
// Cumulative scan forward and backward pass implementations.
//
// Two anonymous-namespace helpers wrap the backend calls to keep apply()
// implementations readable:
//   reverse_along_axis_storage — reverses elements along one axis.
//   cumsum_storage_along       — inclusive prefix-sum along one axis.
//
// Backward for cumsum (reverse-cumsum trick):
//   Given dL/dy (upstream gradient of the output), the gradient for position i
//   is the sum of dL/dy_j for all j >= i.  This equals:
//     dx = reverse(cumsum(reverse(dy)))
//   which avoids an explicit O(n^2) summation.
//
// Backward for cumprod:
//   dx_i = sum_j [ dy_j * (prod_y_j / x_i) ]
//   where prod_y_j = saved_y_[j] (the forward cumprod output).  Reorganising:
//     dx = reverse(cumsum(reverse(dy * saved_y))) / saved_x
//   This reuses the reverse-cumsum primitive applied to (dy * saved_y),
//   then divides element-wise by the saved input x.

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

// Thin wrapper: reverse the storage contents along the given axis.
Storage reverse_along_axis_storage(
    const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    return backend::Dispatcher::for_device(device).reverse_along_axis(s, shape, axis, dt);
}

// Thin wrapper: compute the inclusive prefix sum along the given axis.
Storage
cumsum_storage_along(const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    return backend::Dispatcher::for_device(device).cumsum(s, shape, axis, dt);
}

// Private backward node for cumsum.
//
// Saved state:
//   input_shape_ — shape of the forward input (= shape of the output).
//   axis_        — normalised (non-negative) reduction axis.
class CumsumBackward : public AutogradNode<CumsumBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;

    // dx = reverse(cumsum(reverse(dy))):
    // 1. Reverse dy along axis to turn suffix-sums into prefix-sums.
    // 2. Apply cumsum to accumulate the reversed gradients.
    // 3. Reverse again to restore the original axis order.
    std::vector<Storage> apply(Storage grad_out) override {
        Storage rev = reverse_along_axis_storage(grad_out, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage dx = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        return {std::move(dx)};
    }
};

// Private backward node for cumprod.
//
// Saved state:
//   input_shape_ — shape of the forward input.
//   axis_        — normalised reduction axis.
//   saved_x_     — copy of the forward input tensor's storage.
//   saved_y_     — copy of the forward output (cumprod) tensor's storage.
class CumprodBackward : public AutogradNode<CumprodBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;
    Storage saved_x_;
    Storage saved_y_;

    // dx = reverse(cumsum(reverse(dy * saved_y))) / saved_x.
    // Step 1: weight each upstream gradient by the corresponding cumprod output.
    // Steps 2-3: apply the reverse-cumsum trick to accumulate weighted gradients.
    // Step 4: divide by the saved input x to obtain the per-element gradient.
    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t total = shape_numel(input_shape_);
        // dy * y  (element-wise product of gradient and cumprod output)
        Storage p_s = multiply_storages(grad_out, saved_y_, total, dtype_, device_);
        Storage rev = reverse_along_axis_storage(p_s, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage q = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        Storage dx = divide_storages(q, saved_x_, total, dtype_, device_);
        return {std::move(dx)};
    }
};

// KeepInput: scans are valid for integer types; no promotion needed.
const OpSchema CumsumBackward::schema_v1{"cumsum", 1, AmpPolicy::KeepInput, true};
const OpSchema CumprodBackward::schema_v1{"cumprod", 1, AmpPolicy::KeepInput, true};

// Shared forward dispatch for both scan ops.  Validates the input, normalises
// the axis, dispatches to the backend, and returns a fresh output tensor.
// Autograd wiring is left to the callers so that each can save the appropriate
// tensors (cumsum saves nothing extra; cumprod saves x and y).
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

    Storage out_storage =
        is_prod ? backend::Dispatcher::for_device(device).cumprod(a->storage(), sh, ax, dt)
                : backend::Dispatcher::for_device(device).cumsum(a->storage(), sh, ax, dt);
    return fresh(std::move(out_storage), sh, dt, device);
}

}  // namespace

// Dispatch cumsum, then wire CumsumBackward.  The axis is re-normalised here
// (after scan_dispatch validated it) so that bwd->axis_ is always non-negative.
TensorImplPtr cumsum_op(const TensorImplPtr& a, int axis) {
    auto out = scan_dispatch(a, axis, false, "cumsum");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumsumBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    kernel::NaryKernel<CumsumBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

// Dispatch cumprod, then wire CumprodBackward with both input and output saved.
TensorImplPtr cumprod_op(const TensorImplPtr& a, int axis) {
    auto out = scan_dispatch(a, axis, true, "cumprod");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumprodBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->saved_x_ = a->storage();    // needed for the final division in apply()
    bwd->saved_y_ = out->storage();  // cumprod output, used as the weight
    kernel::NaryKernel<CumprodBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

}  // namespace lucid
