#pragma once

// =====================================================================
// Lucid C++ engine — element-wise arithmetic unary ops.
// =====================================================================
//
//   neg(x)        = -x                   grad: -g
//   abs(x)        = |x|                  grad: sign(x) * g
//   sign(x)       = sgn(x)               grad: 0  (no_grad op)
//   reciprocal(x) = 1/x                  grad: -1/x² * g     [saves input]
//   square(x)     = x²                   grad: 2x * g        [saves input]
//   cube(x)       = x³                   grad: 3x² * g       [saves input]
//
// Layer: autograd/ops/unary/. Backward of `abs` saves the input; `sign`
// produces no gradient (forward only); the rest save the input value.

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

class LUCID_API NegBackward : public UnaryOp<NegBackward> {
public:
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API AbsBackward : public UnaryOp<AbsBackward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API SignBackward : public UnaryOp<SignBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;  // sign has zero gradient everywhere
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);  // unused
};

class LUCID_API ReciprocalBackward : public UnaryOp<ReciprocalBackward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API SquareBackward : public UnaryOp<SquareBackward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

class LUCID_API CubeBackward : public UnaryOp<CubeBackward> {
public:
    static const OpSchema schema_v1;
    static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);
    static GpuStorage gpu_kernel(const GpuStorage& a, const Shape& out_shape, Dtype dt);
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr neg_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr abs_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr sign_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr reciprocal_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr square_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr cube_op(const TensorImplPtr& a);

}  // namespace lucid
