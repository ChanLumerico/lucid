#include "Norm.h"

#include <algorithm>
#include <variant>
#include <vector>

#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Div.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/ScalarParam.h"
#include "../../ops/utils/Layout.h"
#include "../../ops/utils/View.h"
#include "_Detail.h"

namespace lucid {

const OpSchema NormBackward::schema_v1{"norm", 1, AmpPolicy::KeepInput};

std::vector<Storage> NormBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    const int ndim = static_cast<int>(input_shapes_[0].size());

    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto N = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);
    auto dN = fresh(std::move(grad_out), out_shape_, dtype_, device_);

    auto expand_back = [&](const TensorImplPtr& t) -> TensorImplPtr {
        if (keepdims_ || axis_.empty()) {
            return broadcast_to_op(t, input_shapes_[0]);
        }
        std::vector<int> sorted_axes;
        for (int a : axis_)
            sorted_axes.push_back(a < 0 ? a + ndim : a);
        std::sort(sorted_axes.begin(), sorted_axes.end());
        auto result = t;
        for (int i = 0; i < static_cast<int>(sorted_axes.size()); ++i)
            result = unsqueeze_op(result, sorted_axes[i]);
        return broadcast_to_op(result, input_shapes_[0]);
    };

    TensorImplPtr dA;
    if (ord_ == 2.0) {
        auto N_exp = expand_back(clip_op(N, 1e-12, 1e30));
        auto dN_exp = expand_back(dN);
        dA = mul_op(div_op(A, N_exp), dN_exp);
    } else if (ord_ == 1.0) {
        dA = mul_op(sign_op(A), expand_back(dN));
    } else {
        ErrorBuilder("norm_backward")
            .not_implemented("gradient only implemented for ord=1 and ord=2");
    }
    return {dA->storage()};
}

LUCID_REGISTER_OP(NormBackward)

namespace {

Shape reduced_shape(const Shape& sh, const std::vector<int>& axes, bool keepdims) {
    if (axes.empty()) {
        if (keepdims)
            return Shape(sh.size(), 1);
        return Shape{};
    }
    std::vector<bool> mask(sh.size(), false);
    for (int a : axes) {
        int p = a < 0 ? a + static_cast<int>(sh.size()) : a;
        mask[p] = true;
    }
    Shape out;
    for (std::size_t i = 0; i < sh.size(); ++i) {
        if (mask[i]) {
            if (keepdims)
                out.push_back(1);
        } else {
            out.push_back(sh[i]);
        }
    }
    return out;
}

}  // namespace

TensorImplPtr norm_op(const TensorImplPtr& a, double ord, std::vector<int> axis, bool keepdims) {
    using namespace linalg_detail;
    Validator::input(a, "norm.a").non_null();
    require_float(a->dtype(), "norm");
    OpScopeFull scope{"norm", a->device(), a->dtype(), a->shape()};

    Shape out_shape = reduced_shape(a->shape(), axis, keepdims);
    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
            .linalg_norm(a->storage(), a->shape(), ord, axis, keepdims, a->dtype());
    auto out = fresh(std::move(out_storage), out_shape, a->dtype(), a->device());
    auto bwd = std::make_shared<NormBackward>();
    bwd->ord_ = ord;
    bwd->axis_ = axis;
    bwd->keepdims_ = keepdims;
    bwd->saved_output_ = out->storage();
    kernel::NaryKernel<NormBackward, 1>::wire_autograd(std::move(bwd), {a}, out, true);
    return out;
}

}  // namespace lucid
