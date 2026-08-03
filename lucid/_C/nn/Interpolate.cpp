// lucid/_C/nn/Interpolate.cpp
//
// Bilinear, trilinear, and nearest-neighbor interpolation implementations.
//
// Bilinear / trilinear: forward delegates to IBackend::interpolate_*_forward;
//   the backward node saves only the original shape and the output dimensions —
//   the backend recomputes interpolation weights during backward.
//
// Nearest 2-D / 3-D: forward delegates to IBackend::interpolate_nearest_*_forward
//   and attaches a backward node that scatter-adds each output gradient onto its
//   unique source pixel/voxel (same floor coordinate map as the forward).

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
#include "../ops/ufunc/Reductions.h"
#include "../ops/utils/Promote.h"
#include "../ops/utils/View.h"

namespace lucid {

const OpSchema InterpolateBilinearBackward::schema_v1{
    "interpolate_bilinear", 1, AmpPolicy::Promote, true, "", true};

TensorImplPtr InterpolateBilinearBackward::forward(const TensorImplPtr& input0,
                                                   int H_out,
                                                   int W_out,
                                                   bool align_corners) {
    Validator::input(input0, "interpolate_bilinear.input").non_null();
    // Resampling asks for the schema dtype the kernel templates
    // would have applied; see promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    if (input->shape().size() != 4)
        throw ShapeMismatch(input->shape(), Shape{},
                            "interpolate_bilinear: input must be 4-D (N, C, H, W)");
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    const int H_in = static_cast<int>(input->shape()[2]);
    const int W_in = static_cast<int>(input->shape()[3]);
    // The source extents were read and then explicitly discarded.  They
    // are not decoration: bilinear sampling reads the four neighbours of
    // each output coordinate, so a source with no rows or no columns has
    // nothing to read and segfaulted.  ``resized_crop`` on a degenerate
    // image reached here through ``resize``.
    if (H_in <= 0 || W_in <= 0)
        ErrorBuilder("interpolate_bilinear")
            .fail("cannot interpolate from an image with a zero spatial extent");
    if (H_out <= 0 || W_out <= 0)
        ErrorBuilder("interpolate_bilinear").fail("output extent must be positive");
    Shape out_shape{N, C, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};
    scope.set_attr("H_out", static_cast<std::int64_t>(H_out));
    scope.set_attr("W_out", static_cast<std::int64_t>(W_out));
    scope.set_attr("align_corners", align_corners);

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_bilinear_forward(input->storage(), input->shape(), H_out,
                                                          W_out, align_corners, input->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    // wire_autograd records on_op_io internally — no explicit call.
    {
        auto bwd = std::make_shared<InterpolateBilinearBackward>();
        bwd->H_in_ = static_cast<int>(input->shape()[2]);
        bwd->W_in_ = static_cast<int>(input->shape()[3]);
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->align_corners_ = align_corners;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateBilinearBackward, 1>::wire_autograd(std::move(bwd), {input},
                                                                          out, false);
    }
    return out;
}

std::vector<Storage> InterpolateBilinearBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.interpolate_bilinear_backward(grad_out, orig_shape_, H_out_, W_out_, align_corners_,
                                             dtype_)};
}

TensorImplPtr
interpolate_bilinear_op(const TensorImplPtr& input, int H_out, int W_out, bool align_corners) {
    return InterpolateBilinearBackward::forward(input, H_out, W_out, align_corners);
}
LUCID_REGISTER_OP(InterpolateBilinearBackward)

const OpSchema InterpolateTrilinearBackward::schema_v1{
    "interpolate_trilinear", 1, AmpPolicy::Promote, true, "", true};

TensorImplPtr InterpolateTrilinearBackward::forward(
    const TensorImplPtr& input0, int D_out, int H_out, int W_out, bool align_corners) {
    Validator::input(input0, "interpolate_trilinear.input").non_null();
    // Resampling asks for the schema dtype the kernel templates
    // would have applied; see promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
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
                                                                           out, false);
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

const OpSchema InterpolateNearestBackward2D::schema_v1{
    "interpolate_nearest_2d", 1, AmpPolicy::KeepInput, true, "", true};

TensorImplPtr
InterpolateNearestBackward2D::forward(const TensorImplPtr& input0, int H_out, int W_out) {
    Validator::input(input0, "interpolate_nearest.input").non_null();
    // Resampling asks for the schema dtype the kernel templates
    // would have applied; see promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    if (input->shape().size() != 4)
        throw ShapeMismatch(input->shape(), Shape{}, "interpolate_nearest: 4-D input required");
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};
    scope.set_attr("H_out", static_cast<std::int64_t>(H_out));
    scope.set_attr("W_out", static_cast<std::int64_t>(W_out));

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_nearest_2d_forward(input->storage(), input->shape(), H_out,
                                                            W_out, input->dtype());
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    {
        auto bwd = std::make_shared<InterpolateNearestBackward2D>();
        bwd->H_in_ = static_cast<int>(input->shape()[2]);
        bwd->W_in_ = static_cast<int>(input->shape()[3]);
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateNearestBackward2D, 1>::wire_autograd(std::move(bwd), {input},
                                                                           out, false);
    }
    return out;
}

std::vector<Storage> InterpolateNearestBackward2D::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.interpolate_nearest_2d_backward(grad_out, orig_shape_, H_out_, W_out_, dtype_)};
}

std::vector<TensorImplPtr>
InterpolateNearestBackward2D::apply_for_graph(const TensorImplPtr& grad_out) {
    // Nearest-neighbour upscaling by an integer factor repeats each input
    // pixel over a k x k block, so its adjoint sums each block back down.
    // That is the mirror of the expand this file's forward performs, and it
    // is expressible as reshape + sum.  A non-integer ratio maps blocks of
    // uneven size and is refused rather than approximated.
    if (H_out_ % H_in_ != 0 || W_out_ % W_in_ != 0)
        ErrorBuilder("interpolate")
            .not_implemented("create_graph=True needs an integer scale factor for nearest mode");
    const std::int64_t kh = H_out_ / H_in_;
    const std::int64_t kw = W_out_ / W_in_;

    const std::int64_t n = orig_shape_[0];
    const std::int64_t ch = orig_shape_[1];
    auto split = reshape_op(grad_out, {n, ch, H_in_, kh, W_in_, kw});
    auto summed = sum_op(split, std::vector<int>{3, 5}, /*keepdims=*/false);
    std::vector<std::int64_t> orig(orig_shape_.begin(), orig_shape_.end());
    return {reshape_op(summed, orig)};
}

TensorImplPtr interpolate_nearest_2d_op(const TensorImplPtr& input, int H_out, int W_out) {
    return InterpolateNearestBackward2D::forward(input, H_out, W_out);
}
LUCID_REGISTER_OP(InterpolateNearestBackward2D)

const OpSchema InterpolateNearestBackward3D::schema_v1{
    "interpolate_nearest_3d", 1, AmpPolicy::KeepInput, true, "", true};

TensorImplPtr InterpolateNearestBackward3D::forward(const TensorImplPtr& input0,
                                                    int D_out,
                                                    int H_out,
                                                    int W_out) {
    Validator::input(input0, "interpolate_nearest_3d.input").non_null();
    // Resampling asks for the schema dtype the kernel templates
    // would have applied; see promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    if (input->shape().size() != 5)
        throw ShapeMismatch(input->shape(), Shape{}, "interpolate_nearest_3d: 5-D input required");
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, D_out, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_nearest_3d_forward(input->storage(), input->shape(), D_out,
                                                            H_out, W_out, input->dtype());
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    {
        auto bwd = std::make_shared<InterpolateNearestBackward3D>();
        bwd->D_in_ = static_cast<int>(input->shape()[2]);
        bwd->H_in_ = static_cast<int>(input->shape()[3]);
        bwd->W_in_ = static_cast<int>(input->shape()[4]);
        bwd->D_out_ = D_out;
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateNearestBackward3D, 1>::wire_autograd(std::move(bwd), {input},
                                                                           out, false);
    }
    return out;
}

std::vector<Storage> InterpolateNearestBackward3D::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {
        be.interpolate_nearest_3d_backward(grad_out, orig_shape_, D_out_, H_out_, W_out_, dtype_)};
}

TensorImplPtr
interpolate_nearest_3d_op(const TensorImplPtr& input, int D_out, int H_out, int W_out) {
    return InterpolateNearestBackward3D::forward(input, D_out, H_out, W_out);
}
LUCID_REGISTER_OP(InterpolateNearestBackward3D)

}  // namespace lucid
