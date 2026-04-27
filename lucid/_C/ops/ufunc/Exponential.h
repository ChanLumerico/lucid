#pragma once

// =====================================================================
// Lucid C++ engine — element-wise exponential / logarithmic unary ops.
// =====================================================================
//
//   exp(x)   = e^x         grad: e^x * g     [saves output  — exp(x) is `result`]
//   log(x)   = ln(x)       grad: g / x       [saves input]
//   log2(x)  = log₂(x)     grad: g / (x·ln2) [saves input]
//   sqrt(x)  = √x          grad: 0.5·g / √x  [saves output]
//
// Layer: autograd/ops/unary/.

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

class LUCID_API ExpBackward : public UnaryOp<ExpBackward> {
public:
    static constexpr bool kSavesInput  = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API LogBackward : public UnaryOp<LogBackward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API Log2Backward : public UnaryOp<Log2Backward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API SqrtBackward : public UnaryOp<SqrtBackward> {
public:
    static constexpr bool kSavesInput  = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr exp_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr log_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr log2_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr sqrt_op(const TensorImplPtr& a);

}  // namespace lucid
