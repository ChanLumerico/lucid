#include "Exponential.h"

#include <cmath>

#include "../../backend/cpu/Vforce.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/OpRegistry.h"

namespace lucid {

namespace {

CpuStorage allocate_unary(const Shape& out_shape, Dtype dt) {
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = shape_numel(out_shape) * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    return out;
}

template <class F32Fn, class F64Fn>
CpuStorage dispatch(
    const CpuStorage& a, const Shape& out_shape, Dtype dt, F32Fn f32, F64Fn f64, const char* op) {
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

// --------------- Exp ---------------
const OpSchema ExpBackward::schema_v1{"exp", 1, AmpPolicy::ForceFP32, true};

CpuStorage ExpBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vexp_f32, backend::cpu::vexp_f64, "exp");
}

Storage ExpBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = output * g  (cheap — output is saved, no recompute)
    return multiply_storages(g, saved_output_, n, dtype_, device_);
}

TensorImplPtr exp_op(const TensorImplPtr& a) {
    return ExpBackward::forward(a);
}
LUCID_REGISTER_OP(ExpBackward)

// --------------- Log ---------------
const OpSchema LogBackward::schema_v1{"log", 1, AmpPolicy::ForceFP32, true};

CpuStorage LogBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vlog_f32, backend::cpu::vlog_f64, "log");
}

Storage LogBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    return divide_storages(g, saved_inputs_[0], n, dtype_, device_);
}

TensorImplPtr log_op(const TensorImplPtr& a) {
    return LogBackward::forward(a);
}
LUCID_REGISTER_OP(LogBackward)

// --------------- Log2 ---------------
const OpSchema Log2Backward::schema_v1{"log2", 1, AmpPolicy::ForceFP32, true};

CpuStorage Log2Backward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vlog2_f32, backend::cpu::vlog2_f64, "log2");
}

Storage Log2Backward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = g / (x * ln2)
    constexpr double kLn2 = 0.69314718055994530941723212145817656807550013436;
    Storage x_ln2 = mul_scalar_storage(saved_inputs_[0], kLn2, n, dtype_, device_);
    return divide_storages(g, x_ln2, n, dtype_, device_);
}

TensorImplPtr log2_op(const TensorImplPtr& a) {
    return Log2Backward::forward(a);
}
LUCID_REGISTER_OP(Log2Backward)

// --------------- Sqrt ---------------
const OpSchema SqrtBackward::schema_v1{"sqrt", 1, AmpPolicy::Promote, true};

CpuStorage SqrtBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vsqrt_f32, backend::cpu::vsqrt_f64, "sqrt");
}

Storage SqrtBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = 0.5 * g / sqrt(x) = 0.5 * g / output
    Storage half_g = mul_scalar_storage(g, 0.5, n, dtype_, device_);
    return divide_storages(half_g, saved_output_, n, dtype_, device_);
}

TensorImplPtr sqrt_op(const TensorImplPtr& a) {
    return SqrtBackward::forward(a);
}
LUCID_REGISTER_OP(SqrtBackward)

}  // namespace lucid
