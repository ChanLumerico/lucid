// lucid/_C/nn/Fold.cpp
#include "Fold.h"
#include "../backend/Dispatcher.h"
#include "../core/ErrorBuilder.h"
#include "../core/Helpers.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
namespace lucid {

TensorImplPtr fold_op(const TensorImplPtr& x,
                       const std::vector<int>& output_size,
                       const std::vector<int>& kernel_size,
                       const std::vector<int>& stride,
                       const std::vector<int>& padding,
                       const std::vector<int>& dilation) {
    Validator::input(x, "fold.x").non_null();
    if (x->shape().size() != 3)
        ErrorBuilder("fold").fail("input must be 3-D (N, C*kH*kW, L)");

    const int kH = kernel_size[0], kW = kernel_size[1];
    const int outH = output_size[0], outW = output_size[1];
    const int N = static_cast<int>(x->shape()[0]);
    const int C = static_cast<int>(x->shape()[1]) / (kH * kW);

    Shape out_shape = {static_cast<std::int64_t>(N),
                       static_cast<std::int64_t>(C),
                       static_cast<std::int64_t>(outH),
                       static_cast<std::int64_t>(outW)};
    OpScopeFull scope{"fold", x->device(), x->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(x->device());
    Storage out = be.nn_fold(x->storage(), x->shape(), out_shape,
                              kernel_size, stride, padding, dilation, x->dtype());
    return std::make_shared<TensorImpl>(std::move(out), out_shape, x->dtype(), x->device(), false);
}

}  // namespace lucid
