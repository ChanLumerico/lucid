// lucid/_C/ops/linalg/Norm.cpp
//
// Implementation of the tensor norm op and its autograd backward node.
//
// Forward: dispatches to IBackend::linalg_norm() via Dispatcher.
//   CPU path: Apple Accelerate vDSP routines for dot products / sum-of-abs.
//   GPU path: MLX reduction ops (sum, abs, sqrt as needed).
//
// Backward:
//   ord=2: ∂L/∂A = (A / clip(N, ε, ∞)) ⊙ expand(∂L/∂N)
//          Derived from d/dA [sqrt(sum Aᵢ²)] = A / ‖A‖.
//   ord=1: ∂L/∂A = sign(A) ⊙ expand(∂L/∂N)
//          Derived from d/dA [sum |Aᵢ|] = sign(Aᵢ).
//
// The expand_back lambda inside apply() handles the general case of
// re-expanding the reduced gradient to the original shape regardless of
// whether keepdims was true/false or the reduction was partial/full.
//
// File-local reduced_shape() computes the output shape from the input shape
// and the (axes, keepdims) arguments; it is called both in norm_op (to
// construct the output TensorImpl) and implicitly drives the backward shape
// inference via out_shape_.

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

// AmpPolicy::KeepInput: preserve the input dtype so that the A/N division in
// the backward does not silently lose precision through AMP downcasting.
const OpSchema NormBackward::schema_v1{"norm", 1, AmpPolicy::KeepInput};

// Backward pass for norm_op.
//
// expand_back re-inserts the reduced axes so the scalar (batch) gradient can
// be broadcast back to the full input shape.  Two cases:
//   - keepdims_=true or a full reduction (axis_.empty()): the output already
//     has the right number of dimensions (possibly all-ones), so we broadcast
//     directly.
//   - keepdims_=false and a partial reduction: we must re-insert the missing
//     axes.  Indices are normalised (negative -> positive) and sorted ascending
//     so that each unsqueeze_op call inserts at the right logical position even
//     as previously inserted axes shift subsequent dimension indices.
std::vector<Storage> NormBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    const int ndim = static_cast<int>(input_shapes_[0].size());

    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto N = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);
    auto dN = fresh(std::move(grad_out), out_shape_, dtype_, device_);

    // Lambda: expand a reduced-shaped tensor back to the full input shape.
    auto expand_back = [&](const TensorImplPtr& t) -> TensorImplPtr {
        if (keepdims_ || axis_.empty()) {
            // All dimensions are already present (either kept as 1 or as a
            // scalar); broadcast is sufficient.
            return broadcast_to_op(t, input_shapes_[0]);
        }
        // Normalise negative axis indices and sort in ascending order.
        // Ascending sort matters: inserting axis i first preserves the correct
        // index for axis j > i on subsequent unsqueeze calls.
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
        // L2 gradient: dA = (A / clip(N, ε)) * expand(dN)
        // The clip prevents NaN when A = 0 (‖A‖ = 0 means the gradient of ‖A‖
        // is technically undefined; we follow reference framework in returning 0 there).
        auto N_exp = expand_back(clip_op(N, 1e-12, 1e30));
        auto dN_exp = expand_back(dN);
        dA = mul_op(div_op(A, N_exp), dN_exp);
    } else if (ord_ == 1.0) {
        // L1 gradient: dA = sign(A) * expand(dN)
        dA = mul_op(sign_op(A), expand_back(dN));
    } else {
        ErrorBuilder("norm_backward")
            .not_implemented("gradient only implemented for ord=1 and ord=2");
    }
    return {dA->storage()};
}

// Register NormBackward in the global op registry.
LUCID_REGISTER_OP(NormBackward)

namespace {

// Compute the output shape after reducing sh over axes with optional keepdims.
//
// If axes is empty the entire tensor collapses: to [] (scalar) without
// keepdims, or to a rank-N all-ones shape with keepdims.
// Otherwise, each reduced axis is removed or replaced by 1 depending on
// keepdims.  Negative axis values are normalised to positive here.
Shape reduced_shape(const Shape& sh, const std::vector<int>& axes, bool keepdims) {
    if (axes.empty()) {
        if (keepdims)
            return Shape(sh.size(), 1);  // keep all dims as size-1
        return Shape{};                  // global reduction to scalar
    }
    // Mark which dimensions are reduced.
    std::vector<bool> mask(sh.size(), false);
    for (int a : axes) {
        int p = a < 0 ? a + static_cast<int>(sh.size()) : a;
        mask[p] = true;
    }
    Shape out;
    for (std::size_t i = 0; i < sh.size(); ++i) {
        if (mask[i]) {
            if (keepdims)
                out.push_back(1);  // collapsed but kept as a stub
            // else: dimension is simply removed from the output shape
        } else {
            out.push_back(sh[i]);  // non-reduced dimension passes through
        }
    }
    return out;
}

}  // namespace

// Compute the p-norm of tensor a over the specified axes.
//
// The backward hyperparameters (ord, axis, keepdims) are stored on the
// NormBackward node so it can reconstruct expand_back at backward time.
// saved_output_ captures the forward norm value N so the backward can
// compute A/N without re-running the forward.
// save_inputs=true stores A in saved_inputs_[0] for the L2 numerator.
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
    // Copy the hyperparameters onto the backward node.
    bwd->ord_ = ord;
    bwd->axis_ = axis;
    bwd->keepdims_ = keepdims;
    bwd->saved_output_ = out->storage();
    // save_inputs=true: NormBackward reads A from saved_inputs_[0] for ord=2.
    kernel::NaryKernel<NormBackward, 1>::wire_autograd(std::move(bwd), {a}, out, true);
    return out;
}

}  // namespace lucid
