#include "Trig.h"

#include "../../backend/cpu/Vforce.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/OpRegistry.h"

namespace lucid {

namespace {

CpuStorage allocate_unary(const Shape& out_shape, Dtype dt) {
    CpuStorage out;
    out.dtype  = dt;
    out.nbytes = shape_numel(out_shape) * dtype_size(dt);
    out.ptr    = allocate_aligned_bytes(out.nbytes);
    return out;
}

template <class F32Fn, class F64Fn>
CpuStorage dispatch(const CpuStorage& a, const Shape& out_shape, Dtype dt,
                    F32Fn f32, F64Fn f64, const char* op) {
    const std::size_t numel = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32:
            f32(reinterpret_cast<const float*>(a.ptr.get()),
                reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            f64(reinterpret_cast<const double*>(a.ptr.get()),
                reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            throw NotImplementedError(std::string(op) + ": dtype not supported");
    }
    return out;
}

}  // namespace

// --------------- Sin ---------------
const OpSchema SinBackward::schema_v1{"sin", 1, AmpPolicy::Promote, true};

CpuStorage SinBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vsin_f32, backend::cpu::vsin_f64, "sin");
}

Storage SinBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage cosx = cos_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, cosx, n, dtype_, device_);
}

TensorImplPtr sin_op(const TensorImplPtr& a) { return SinBackward::forward(a); }
LUCID_REGISTER_OP(SinBackward)

// --------------- Cos ---------------
const OpSchema CosBackward::schema_v1{"cos", 1, AmpPolicy::Promote, true};

CpuStorage CosBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vcos_f32, backend::cpu::vcos_f64, "cos");
}

Storage CosBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage sinx = sin_storage(saved_inputs_[0], n, dtype_, device_);
    Storage neg_sin = negate_storage(sinx, n, dtype_, device_);
    return multiply_storages(g, neg_sin, n, dtype_, device_);
}

TensorImplPtr cos_op(const TensorImplPtr& a) { return CosBackward::forward(a); }
LUCID_REGISTER_OP(CosBackward)

// --------------- Tan ---------------
const OpSchema TanBackward::schema_v1{"tan", 1, AmpPolicy::Promote, true};

CpuStorage TanBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vtan_f32, backend::cpu::vtan_f64, "tan");
}

Storage TanBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = g / cos²(x)
    Storage cosx     = cos_storage(saved_inputs_[0], n, dtype_, device_);
    Storage cos_sq   = square_storage(cosx, n, dtype_, device_);
    return divide_storages(g, cos_sq, n, dtype_, device_);
}

TensorImplPtr tan_op(const TensorImplPtr& a) { return TanBackward::forward(a); }
LUCID_REGISTER_OP(TanBackward)

// --------------- Asin ---------------
const OpSchema AsinBackward::schema_v1{"arcsin", 1, AmpPolicy::Promote, true};

CpuStorage AsinBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vasin_f32, backend::cpu::vasin_f64, "arcsin");
}

Storage AsinBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = g / sqrt(1 - x²)
    Storage x_sq    = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage one_min = mul_scalar_storage(x_sq, -1.0, n, dtype_, device_);
    Storage radicand = add_scalar_storage(one_min, 1.0, n, dtype_, device_);
    Storage root = sqrt_storage(radicand, n, dtype_, device_);
    return divide_storages(g, root, n, dtype_, device_);
}

TensorImplPtr arcsin_op(const TensorImplPtr& a) { return AsinBackward::forward(a); }
LUCID_REGISTER_OP(AsinBackward)

// --------------- Acos ---------------
const OpSchema AcosBackward::schema_v1{"arccos", 1, AmpPolicy::Promote, true};

CpuStorage AcosBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vacos_f32, backend::cpu::vacos_f64, "arccos");
}

Storage AcosBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage x_sq    = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage one_min = mul_scalar_storage(x_sq, -1.0, n, dtype_, device_);
    Storage radicand = add_scalar_storage(one_min, 1.0, n, dtype_, device_);
    Storage root = sqrt_storage(radicand, n, dtype_, device_);
    Storage q = divide_storages(g, root, n, dtype_, device_);
    return negate_storage(q, n, dtype_, device_);
}

TensorImplPtr arccos_op(const TensorImplPtr& a) { return AcosBackward::forward(a); }
LUCID_REGISTER_OP(AcosBackward)

// --------------- Atan ---------------
const OpSchema AtanBackward::schema_v1{"arctan", 1, AmpPolicy::Promote, true};

CpuStorage AtanBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vatan_f32, backend::cpu::vatan_f64, "arctan");
}

Storage AtanBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = g / (1 + x²)
    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage denom = add_scalar_storage(x_sq, 1.0, n, dtype_, device_);
    return divide_storages(g, denom, n, dtype_, device_);
}

TensorImplPtr arctan_op(const TensorImplPtr& a) { return AtanBackward::forward(a); }
LUCID_REGISTER_OP(AtanBackward)

}  // namespace lucid
