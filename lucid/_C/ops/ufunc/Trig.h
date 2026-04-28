#pragma once

// =====================================================================
// Lucid C++ engine — trigonometric unary ops.
// =====================================================================
//
//   sin(x)    grad: cos(x) * g
//   cos(x)    grad: -sin(x) * g
//   tan(x)    grad: g / cos²(x)
//   asin(x)   grad: g / √(1 - x²)
//   acos(x)   grad: -g / √(1 - x²)
//   atan(x)   grad: g / (1 + x²)

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

#define LUCID_DECLARE_TRIG_OP(CLASS, FN)                                                     \
    class LUCID_API CLASS##Backward : public UnaryOp<CLASS##Backward> {                      \
    public:                                                                                  \
        static const OpSchema schema_v1;                                                     \
        static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt); \
        static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt); \
        Storage grad_formula(const Storage& g);                                              \
    };                                                                                       \
    LUCID_API TensorImplPtr FN##_op(const TensorImplPtr& a);

LUCID_DECLARE_TRIG_OP(Sin, sin)
LUCID_DECLARE_TRIG_OP(Cos, cos)
LUCID_DECLARE_TRIG_OP(Tan, tan)
LUCID_DECLARE_TRIG_OP(Asin, arcsin)
LUCID_DECLARE_TRIG_OP(Acos, arccos)
LUCID_DECLARE_TRIG_OP(Atan, arctan)

#undef LUCID_DECLARE_TRIG_OP

}  // namespace lucid
