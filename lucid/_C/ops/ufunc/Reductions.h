// lucid/_C/ops/ufunc/Reductions.h
//
// Backward nodes and entry points for multi-axis reductions: sum, mean, prod,
// max, min.  Each node inherits from ReduceOp<Derived> (an alias for
// ReduceKernel<Derived>), which handles axis normalisation, dispatch, and the
// autograd bookkeeping (saved reduce_axes_, keepdims_, full_input_shape_).
//
// Backward strategy per op:
//   sum  — broadcast_back_for_reduce: expand the reduced gradient to the input
//           shape by broadcasting along the collapsed axes.
//   mean — same as sum, then divide by the number of reduced elements.
//   prod — broadcast both the output and the gradient back to input shape, then
//           compute grad * (prod_output / x).  Requires both input and output.
//   max/min — builds an equality mask (argmax indicator), then multiplies the
//             broadcast gradient by the mask.  Ties receive equal gradient.

#pragma once

#include <utility>
#include <vector>

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_ReduceOp.h"

namespace lucid {

// Backward node for reduction sum along arbitrary axes.
//
// Gradient rule: broadcast the upstream gradient back to the input shape.
// kSavesInput = false because sum backward needs no saved tensor.
class LUCID_API SumBackward : public ReduceOp<SumBackward> {
public:
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be,
                            const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes,
                            bool keepdims,
                            Dtype dt) {
        return be.reduce_sum(a, in_shape, {axes, keepdims}, dt);
    }

    Storage grad_formula(const Storage& grad_out);
    // sum backward has no scaling: gradient broadcasts unchanged.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g) { return g; }
};

// Backward node for reduction mean along arbitrary axes.
//
// Gradient rule: broadcast the upstream gradient back to the input shape and
// scale by 1/N, where N is the product of the reduced dimension sizes.
// kSavesInput = false.
class LUCID_API MeanBackward : public ReduceOp<MeanBackward> {
public:
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be,
                            const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes,
                            bool keepdims,
                            Dtype dt) {
        return be.reduce_mean(a, in_shape, {axes, keepdims}, dt);
    }

    Storage grad_formula(const Storage& grad_out);
    // mean backward divides by n_reduced to scale the broadcast gradient.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g);
};

// Backward node for reduction product along arbitrary axes.
//
// Gradient rule: dL/dx_i = dL/dy * (prod_y / x_i), i.e., the gradient is the
// product of all *other* elements along the reduction axes.  Both input and
// output must be saved.  Requires cpu_kernel and gpu_kernel because no single
// IBackend::dispatch overload covers prod for all back-ends.
class LUCID_API ProdBackward : public ReduceOp<ProdBackward> {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // CPU path: iterates axes in descending order (innermost last) to produce
    // a correct sequential multi-axis product using Accelerate primitives.
    static CpuStorage cpu_kernel(const CpuStorage& a,
                                 const Shape& in_shape,
                                 const std::vector<int>& axes,
                                 bool keepdims,
                                 Dtype dt);
    // GPU path: delegates to mlx::core::prod with keepdims.
    static GpuStorage gpu_kernel(const GpuStorage& a,
                                 const Shape& in_shape,
                                 const std::vector<int>& axes,
                                 bool keepdims,
                                 Dtype dt);
    Storage grad_formula(const Storage& grad_out);
};

// Backward node for reduction maximum along arbitrary axes.
//
// Gradient rule: route the gradient only to positions where x == max(x).
// Saves the *output* (the max value) to build the equality mask without
// re-running the reduction.  Ties (multiple equal maxima) split the gradient
// equally, matching NumPy/PyTorch behaviour.
class LUCID_API MaxBackward : public ReduceOp<MaxBackward> {
public:
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be,
                            const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes,
                            bool keepdims,
                            Dtype dt) {
        return be.reduce_max(a, in_shape, {axes, keepdims}, dt);
    }

    Storage grad_formula(const Storage& grad_out);
};

// Backward node for reduction minimum along arbitrary axes.
//
// Gradient rule: symmetric to MaxBackward — route gradient to positions where
// x == min(x).  The equality mask is built by composing two ge_mask calls
// (a==b iff a>=b && b>=a).
class LUCID_API MinBackward : public ReduceOp<MinBackward> {
public:
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be,
                            const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes,
                            bool keepdims,
                            Dtype dt) {
        return be.reduce_min(a, in_shape, {axes, keepdims}, dt);
    }

    Storage grad_formula(const Storage& grad_out);
};

LUCID_API TensorImplPtr sum_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

LUCID_API TensorImplPtr mean_op(const TensorImplPtr& a,
                                const std::vector<int>& axes,
                                bool keepdims);

LUCID_API TensorImplPtr prod_op(const TensorImplPtr& a,
                                const std::vector<int>& axes,
                                bool keepdims);

LUCID_API TensorImplPtr max_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

LUCID_API TensorImplPtr min_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

// Standard deviation: std(x) = sqrt(var(x, axes, keepdims)).
// Gradient flows through the sqrt and var backward nodes automatically.
LUCID_API TensorImplPtr std_op(const TensorImplPtr& a,
                               const std::vector<int>& axes,
                               bool keepdims);

}  // namespace lucid
