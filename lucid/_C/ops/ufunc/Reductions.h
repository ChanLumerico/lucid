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
};

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
};

class LUCID_API ProdBackward : public ReduceOp<ProdBackward> {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = true;
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

}  // namespace lucid
