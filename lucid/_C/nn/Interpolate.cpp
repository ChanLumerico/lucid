#include "Interpolate.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

// =====================================================================
// interpolate_bilinear (4-D)
// =====================================================================

const OpSchema InterpolateBilinearBackward::schema_v1{"interpolate_bilinear", 1, AmpPolicy::Promote,
                                                      true};

TensorImplPtr InterpolateBilinearBackward::forward(const TensorImplPtr& input,
                                                   int H_out,
                                                   int W_out,
                                                   bool align_corners) {
    Validator::input(input, "interpolate_bilinear.input").non_null();
    if (input->shape().size() != 4)
        throw ShapeMismatch(input->shape(), Shape{},
                            "interpolate_bilinear: input must be 4-D (N, C, H, W)");
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    const int H_in = static_cast<int>(input->shape()[2]);
    const int W_in = static_cast<int>(input->shape()[3]);
    (void)H_in;
    (void)W_in;
    Shape out_shape{N, C, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_bilinear_forward(input->storage(), input->shape(), H_out,
                                                          W_out, align_corners, input->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    {
        auto bwd = std::make_shared<InterpolateBilinearBackward>();
        bwd->H_in_ = static_cast<int>(input->shape()[2]);
        bwd->W_in_ = static_cast<int>(input->shape()[3]);
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->align_corners_ = align_corners;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateBilinearBackward, 1>::wire_autograd(std::move(bwd), {input},
                                                                          out, /*save_ins=*/false);
    }
    return out;
}

std::vector<Storage> InterpolateBilinearBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.interpolate_bilinear_backward(grad_out, orig_shape_, H_out_, W_out_, align_corners_,
                                             dtype_)};
}

TensorImplPtr interpolate_bilinear_op(const TensorImplPtr& input,
                                      int H_out,
                                      int W_out,
                                      bool align_corners) {
    return InterpolateBilinearBackward::forward(input, H_out, W_out, align_corners);
}
LUCID_REGISTER_OP(InterpolateBilinearBackward)

// =====================================================================
// interpolate_trilinear (5-D)
// =====================================================================

const OpSchema InterpolateTrilinearBackward::schema_v1{"interpolate_trilinear", 1,
                                                       AmpPolicy::Promote, true};

TensorImplPtr InterpolateTrilinearBackward::forward(
    const TensorImplPtr& input, int D_out, int H_out, int W_out, bool align_corners) {
    Validator::input(input, "interpolate_trilinear.input").non_null();
    if (input->shape().size() != 5)
        throw ShapeMismatch(input->shape(), Shape{}, "interpolate_trilinear: input must be 5-D");
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, D_out, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_trilinear_forward(
        input->storage(), input->shape(), D_out, H_out, W_out, align_corners, input->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    {
        auto bwd = std::make_shared<InterpolateTrilinearBackward>();
        bwd->D_in_ = static_cast<int>(input->shape()[2]);
        bwd->H_in_ = static_cast<int>(input->shape()[3]);
        bwd->W_in_ = static_cast<int>(input->shape()[4]);
        bwd->D_out_ = D_out;
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->align_corners_ = align_corners;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateTrilinearBackward, 1>::wire_autograd(std::move(bwd), {input},
                                                                           out, /*save_ins=*/false);
    }
    return out;
}

std::vector<Storage> InterpolateTrilinearBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.interpolate_trilinear_backward(grad_out, orig_shape_, D_out_, H_out_, W_out_,
                                              align_corners_, dtype_)};
}

TensorImplPtr interpolate_trilinear_op(
    const TensorImplPtr& input, int D_out, int H_out, int W_out, bool align_corners) {
    return InterpolateTrilinearBackward::forward(input, D_out, H_out, W_out, align_corners);
}
LUCID_REGISTER_OP(InterpolateTrilinearBackward)

// =====================================================================
// interpolate_nearest (no autograd: indices are non-differentiable).
// =====================================================================

TensorImplPtr interpolate_nearest_2d_op(const TensorImplPtr& input, int H_out, int W_out) {
    Validator::input(input, "interpolate_nearest.input").non_null();
    if (input->shape().size() != 4)
        throw ShapeMismatch(input->shape(), Shape{}, "interpolate_nearest: 4-D input required");
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, H_out, W_out};
    OpScopeFull scope{"interpolate_nearest_2d", input->device(), input->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_nearest_2d_forward(input->storage(), input->shape(), H_out,
                                                            W_out, input->dtype());
    return std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                        input->device(), false);
}

TensorImplPtr interpolate_nearest_3d_op(const TensorImplPtr& input,
                                        int D_out,
                                        int H_out,
                                        int W_out) {
    Validator::input(input, "interpolate_nearest_3d.input").non_null();
    if (input->shape().size() != 5)
        throw ShapeMismatch(input->shape(), Shape{}, "interpolate_nearest_3d: 5-D input required");
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, D_out, H_out, W_out};
    OpScopeFull scope{"interpolate_nearest_3d", input->device(), input->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_nearest_3d_forward(input->storage(), input->shape(), D_out,
                                                            H_out, W_out, input->dtype());
    return std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                        input->device(), false);
}

}  // namespace lucid
