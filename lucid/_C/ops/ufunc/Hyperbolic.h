#pragma once

// =====================================================================
// Lucid C++ engine — hyperbolic unary ops.
// =====================================================================
//
//   sinh(x)   grad: cosh(x) * g
//   cosh(x)   grad: sinh(x) * g
//   tanh(x)   grad: (1 - tanh²(x)) * g  =  (1 - z²) * g  [z = output]

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

class LUCID_API SinhBackward : public UnaryOp<SinhBackward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API CoshBackward : public UnaryOp<CoshBackward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API TanhBackward : public UnaryOp<TanhBackward> {
public:
    static constexpr bool kSavesInput  = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr sinh_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr cosh_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr tanh_op(const TensorImplPtr& a);

}  // namespace lucid
