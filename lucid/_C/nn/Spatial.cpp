// lucid/_C/nn/Spatial.cpp
//
// Implementation of affine_grid and grid_sample.
//
// AffineGrid forward: IBackend::affine_grid_forward produces a (N, H, W, 2)
//   grid of normalized sample coordinates from a batch of 2x3 affine matrices.
//   Backward wiring is skipped when theta does not require a gradient.
//
// GridSample forward: IBackend::grid_sample_forward resamples the input image
//   using bilinear/nearest/bicubic interpolation at the grid positions.
//   Backward wiring is skipped when neither input nor grid requires a gradient.
//
// The backward for both ops is fully delegated to the backend; no activations
// are saved beyond the mode and shape parameters recorded in the backward node.

#include "Spatial.h"

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
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema AffineGridBackward::schema_v1{"affine_grid", 1, AmpPolicy::Promote, true};

TensorImplPtr
AffineGridBackward::forward(const TensorImplPtr& theta, int N, int H, int W, bool align_corners) {
    if (!theta)
        ErrorBuilder("affine_grid").fail("null theta");
    if (theta->shape().size() != 3 || theta->shape()[0] != N || theta->shape()[1] != 2 ||
        theta->shape()[2] != 3)
        throw ShapeMismatch(theta->shape(), Shape{static_cast<std::int64_t>(N), 2, 3},
                            "affine_grid: theta must be (N, 2, 3)");

    Shape out_shape{static_cast<std::int64_t>(N), static_cast<std::int64_t>(H),
                    static_cast<std::int64_t>(W), 2};
    OpScopeFull scope{schema_v1.name, theta->device(), theta->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(theta->device());
    Storage out_storage =
        be.affine_grid_forward(theta->storage(), N, H, W, align_corners, theta->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, theta->dtype(),
                                            theta->device(), false);

    if (!GradMode::is_enabled() || !theta->requires_grad())
        return out;

    auto bwd = std::make_shared<AffineGridBackward>();
    bwd->align_corners_ = align_corners;
    bwd->N_ = N;
    bwd->H_ = H;
    bwd->W_ = W;
    bwd->orig_theta_shape_ = theta->shape();
    kernel::NaryKernel<AffineGridBackward, 1>::wire_autograd(std::move(bwd), {theta}, out, false);
    return out;
}

std::vector<Storage> AffineGridBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.affine_grid_backward(grad_out, N_, H_, W_, align_corners_, dtype_)};
}

TensorImplPtr affine_grid_op(const TensorImplPtr& theta, int N, int H, int W, bool align_corners) {
    return AffineGridBackward::forward(theta, N, H, W, align_corners);
}
LUCID_REGISTER_OP(AffineGridBackward)

const OpSchema GridSampleBackward::schema_v1{"grid_sample", 1, AmpPolicy::Promote, true};

TensorImplPtr GridSampleBackward::forward(const TensorImplPtr& input,
                                          const TensorImplPtr& grid,
                                          int mode,
                                          int padding_mode,
                                          bool align_corners) {
    if (!input || !grid)
        ErrorBuilder("grid_sample").fail("null input");
    if (input->device() != grid->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(grid->device())), "grid_sample: input/grid");
    if (input->shape().size() != 4)
        throw ShapeMismatch(input->shape(), Shape{},
                            "grid_sample: input must be (N, C, H_in, W_in)");
    if (grid->shape().size() != 4 || grid->shape()[3] != 2)
        throw ShapeMismatch(grid->shape(), Shape{},
                            "grid_sample: grid must be (N, H_out, W_out, 2)");
    if (input->dtype() != grid->dtype())
        throw DtypeMismatch(std::string(dtype_name(input->dtype())),
                            std::string(dtype_name(grid->dtype())), "grid_sample");
    if (input->shape()[0] != grid->shape()[0])
        throw ShapeMismatch(input->shape(), grid->shape(), "grid_sample: batch size mismatch");

    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    const int H_out = static_cast<int>(grid->shape()[1]);
    const int W_out = static_cast<int>(grid->shape()[2]);
    (void)C;

    Shape out_shape{static_cast<std::int64_t>(N), static_cast<std::int64_t>(C),
                    static_cast<std::int64_t>(H_out), static_cast<std::int64_t>(W_out)};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage =
        be.grid_sample_forward(input->storage(), grid->storage(), input->shape(), grid->shape(),
                               mode, padding_mode, align_corners, input->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);

    if (!GradMode::is_enabled() || !(input->requires_grad() || grid->requires_grad()))
        return out;

    auto bwd = std::make_shared<GridSampleBackward>();
    bwd->mode_ = mode;
    bwd->padding_mode_ = padding_mode;
    bwd->align_corners_ = align_corners;
    bwd->input_shape_ = input->shape();
    bwd->grid_shape_ = grid->shape();
    kernel::NaryKernel<GridSampleBackward, 2>::wire_autograd(std::move(bwd), {input, grid}, out);
    return out;
}

std::vector<Storage> GridSampleBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return be.grid_sample_backward(grad_out, saved_inputs_[0], saved_inputs_[1], input_shape_,
                                   grid_shape_, mode_, padding_mode_, align_corners_, dtype_);
}

TensorImplPtr grid_sample_op(const TensorImplPtr& input,
                             const TensorImplPtr& grid,
                             int mode,
                             int padding_mode,
                             bool align_corners) {
    return GridSampleBackward::forward(input, grid, mode, padding_mode, align_corners);
}
LUCID_REGISTER_OP(GridSampleBackward)

}  // namespace lucid
