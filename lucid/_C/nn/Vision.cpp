#include "Vision.h"

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
// one_hot — no autograd; integer-indexed scatter.
// =====================================================================

TensorImplPtr one_hot_op(const TensorImplPtr& input, int num_classes, Dtype out_dtype) {
    Validator::input(input, "one_hot.input").non_null();
    if (num_classes <= 0)
        ErrorBuilder("one_hot").fail("num_classes must be > 0");
    Shape out_shape = input->shape();
    out_shape.push_back(num_classes);
    OpScopeFull scope{"one_hot", input->device(), out_dtype, out_shape};

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage =
        be.one_hot_forward(input->storage(), input->shape(), num_classes, out_dtype);
    return std::make_shared<TensorImpl>(std::move(out_storage), out_shape, out_dtype,
                                        input->device(), false);
}

// =====================================================================
// rotate — no autograd; nearest-neighbor sample with affine matrix.
// =====================================================================

TensorImplPtr rotate_op(const TensorImplPtr& input, double angle_deg, double cy, double cx) {
    Validator::input(input, "rotate.input").non_null();
    if (input->shape().size() != 4)
        throw ShapeMismatch(input->shape(), Shape{}, "rotate: input must be 4-D (N, C, H, W)");
    OpScopeFull scope{"rotate", input->device(), input->dtype(), input->shape()};
    const double angle_rad_neg = -angle_deg * (M_PI / 180.0);

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage =
        be.rotate_forward(input->storage(), input->shape(), angle_rad_neg, cx, cy, input->dtype());
    return std::make_shared<TensorImpl>(std::move(out_storage), input->shape(), input->dtype(),
                                        input->device(), false);
}

// =====================================================================
// bilinear (learned bilinear layer): y[..., k] = x1 W_k x2 + b_k
// =====================================================================
//
// x1: [..., D1]   x2: [..., D2]   weight: [Dout, D1, D2]   bias: [Dout]
// Forward:
//   tmp = einsum("...i, k i j -> ...k j", x1, W)            # [..., Dout, D2]
//   y   = einsum("...k j, ...j -> ...k", tmp, x2) + bias    # [..., Dout]
// Backward (broadcast over batch dims; we collapse leading dims to a single B):
//   dx1[k]  = sum_k dY[..., k] · W[k, :, :] · x2
//   dx2[j]  = sum_k dY[..., k] · W[k, i, :]^T · x1[i] (per-element); explicit form below.
//   dW[k,i,j] = sum_batch dY[..., k] · x1[i] · x2[j]
//   db[k]   = sum_batch dY[..., k]

const OpSchema BilinearLayerBackward::schema_v1{"bilinear_layer", 1, AmpPolicy::Promote, true};

namespace {

struct BilinearShape {
    std::size_t B;
    std::size_t D1;
    std::size_t D2;
    std::size_t Dout;
};

BilinearShape flatten_bilinear(const Shape& s1, const Shape& s2, const Shape& sw) {
    if (s1.empty() || s2.empty()) {
        ErrorBuilder("bilinear").fail("input must have ≥1 dim");
    }
    if (sw.size() != 3) {
        throw ShapeMismatch(sw, Shape{}, "bilinear: weight must be 3-D");
    }
    if (s1.size() != s2.size()) {
        throw ShapeMismatch(s1, s2, "bilinear: x1/x2 must have same rank");
    }
    BilinearShape r;
    r.Dout = static_cast<std::size_t>(sw[0]);
    r.D1 = static_cast<std::size_t>(sw[1]);
    r.D2 = static_cast<std::size_t>(sw[2]);
    if (static_cast<std::size_t>(s1.back()) != r.D1 ||
        static_cast<std::size_t>(s2.back()) != r.D2) {
        throw ShapeMismatch(s1, sw, "bilinear: last dims must match weight");
    }
    std::size_t b = 1;
    for (std::size_t i = 0; i + 1 < s1.size(); ++i) {
        if (s1[i] != s2[i])
            throw ShapeMismatch(s1, s2, "bilinear: batch dims must match");
        b *= static_cast<std::size_t>(s1[i]);
    }
    r.B = b;
    return r;
}

}  // namespace

TensorImplPtr BilinearLayerBackward::forward(const TensorImplPtr& x1,
                                             const TensorImplPtr& x2,
                                             const TensorImplPtr& weight,
                                             const TensorImplPtr& bias) {
    if (!x1 || !x2 || !weight)
        ErrorBuilder("bilinear_layer").fail("null input");
    if (x1->dtype() != x2->dtype() || x1->dtype() != weight->dtype())
        throw DtypeMismatch(std::string(dtype_name(x1->dtype())),
                            std::string(dtype_name(x2->dtype())), "bilinear_layer");
    if (x1->device() != x2->device() || x1->device() != weight->device())
        throw DeviceMismatch(std::string(device_name(x1->device())),
                             std::string(device_name(x2->device())), "bilinear_layer");
    const auto bs = flatten_bilinear(x1->shape(), x2->shape(), weight->shape());
    Shape out_shape = x1->shape();
    out_shape.back() = static_cast<std::int64_t>(bs.Dout);
    OpScopeFull scope{schema_v1.name, x1->device(), x1->dtype(), out_shape};

    const bool has_bias = (bias != nullptr);
    Storage bias_storage = has_bias ? bias->storage() : Storage{CpuStorage{}};

    auto& be = backend::Dispatcher::for_device(x1->device());
    Storage out_storage =
        be.bilinear_layer_forward(x1->storage(), x2->storage(), weight->storage(), bias_storage,
                                  has_bias, x1->shape(), x2->shape(), weight->shape(), x1->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, x1->dtype(),
                                            x1->device(), false);
    const bool any_grad = x1->requires_grad() || x2->requires_grad() || weight->requires_grad() ||
                          (bias && bias->requires_grad());
    if (!GradMode::is_enabled() || !any_grad)
        return out;

    auto bwd = std::make_shared<BilinearLayerBackward>();
    bwd->orig_x1_shape_ = x1->shape();
    bwd->orig_x2_shape_ = x2->shape();
    kernel::NaryKernel<BilinearLayerBackward, 4>::wire_autograd(std::move(bwd),
                                                                {x1, x2, weight, bias}, out);
    return out;
}

std::vector<Storage> BilinearLayerBackward::apply(Storage grad_out) {
    const bool has_bias = !input_shapes_[3].empty();

    auto& be = backend::Dispatcher::for_device(device_);
    return be.bilinear_layer_backward(grad_out, saved_inputs_[0], saved_inputs_[1],
                                      saved_inputs_[2], orig_x1_shape_, orig_x2_shape_,
                                      input_shapes_[2], has_bias, dtype_);
}

TensorImplPtr bilinear_layer_op(const TensorImplPtr& x1,
                                const TensorImplPtr& x2,
                                const TensorImplPtr& weight,
                                const TensorImplPtr& bias) {
    return BilinearLayerBackward::forward(x1, x2, weight, bias);
}
LUCID_REGISTER_OP(BilinearLayerBackward)

}  // namespace lucid
