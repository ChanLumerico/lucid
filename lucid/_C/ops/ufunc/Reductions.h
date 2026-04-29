#pragma once

// =====================================================================
// Lucid C++ engine — collapsing reductions (Sum/Mean/Prod/Max/Min).
// =====================================================================
//
//   sum(x, axis, keepdims)   grad: broadcast(g, input_shape)
//   mean(x, axis, keepdims)  grad: broadcast(g / N, input_shape)
//   prod(x, axis, keepdims)  grad: broadcast(g * y, input_shape) / x
//                              (y = saved output, divides by saved input)
//   max(x, axis, keepdims)   grad: g flows to argmax positions only
//   min(x, axis, keepdims)   grad: g flows to argmin positions only
//
// All forward signatures: `(tensor, axes, keepdims)` where axes is a
// std::vector<int>. Empty axes = reduce all dims.

#include <utility>
#include <vector>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "../../backend/IBackend.h"
#include "_ReduceOp.h"

namespace lucid {

/// Autograd backward node for Sum.
class LUCID_API SumBackward : public ReduceOp<SumBackward> {
public:
    // grad = broadcast(g, input_shape) — no input values needed.
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes, bool keepdims, Dtype dt) {
        return be.reduce_sum(a, in_shape, {axes, keepdims}, dt);
    }

    Storage grad_formula(const Storage& grad_out);
};

/// Autograd backward node for Mean.
class LUCID_API MeanBackward : public ReduceOp<MeanBackward> {
public:
    // grad = broadcast(g / N, input_shape) — no input values needed.
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes, bool keepdims, Dtype dt) {
        return be.reduce_mean(a, in_shape, {axes, keepdims}, dt);
    }

    Storage grad_formula(const Storage& grad_out);
};

/// Autograd backward node for Prod.
class LUCID_API ProdBackward : public ReduceOp<ProdBackward> {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = true;  // backward uses both input and output
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a,
                                 const Shape& in_shape,
                                 const std::vector<int>& axes,
                                 bool keepdims,
                                 Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a,
                                 const Shape& in_shape,
                                 const std::vector<int>& axes,
                                 bool keepdims,
                                 Dtype dt);
    Storage grad_formula(const Storage& grad_out);
};

/// Autograd backward node for Max.
class LUCID_API MaxBackward : public ReduceOp<MaxBackward> {
public:
    static constexpr bool kSavesOutput = true;  // grad_formula uses saved_output_
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes, bool keepdims, Dtype dt) {
        return be.reduce_max(a, in_shape, {axes, keepdims}, dt);
    }

    Storage grad_formula(const Storage& grad_out);
};

/// Autograd backward node for Min.
class LUCID_API MinBackward : public ReduceOp<MinBackward> {
public:
    static constexpr bool kSavesOutput = true;  // grad_formula uses saved_output_
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes, bool keepdims, Dtype dt) {
        return be.reduce_min(a, in_shape, {axes, keepdims}, dt);
    }

    Storage grad_formula(const Storage& grad_out);
};

/// Sum.
LUCID_API TensorImplPtr sum_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);
/// Mean.
LUCID_API TensorImplPtr mean_op(const TensorImplPtr& a,
                                const std::vector<int>& axes,
                                bool keepdims);
/// Prod.
LUCID_API TensorImplPtr prod_op(const TensorImplPtr& a,
                                const std::vector<int>& axes,
                                bool keepdims);
/// Max.
LUCID_API TensorImplPtr max_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);
/// Min.
LUCID_API TensorImplPtr min_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

}  // namespace lucid
