#include "Arith.h"

#include "../../backend/cpu/Vdsp.h"
#include "../../backend/cpu/Vforce.h"
#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
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

template <class FloatF32, class FloatF64>
CpuStorage dispatch_unary(const CpuStorage& a,
                          const Shape& out_shape,
                          Dtype dt,
                          FloatF32 f32,
                          FloatF64 f64,
                          const char* op_name) {
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
            ErrorBuilder(op_name).not_implemented("dtype not supported in Phase 3.2");
    }
    return out;
}

}  // namespace

// --------------- Neg ---------------
const OpSchema NegBackward::schema_v1{"neg", 1, AmpPolicy::Promote, true};

CpuStorage NegBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch_unary(a, out_shape, dt, backend::cpu::vneg_f32, backend::cpu::vneg_f64, "neg");
}

Storage NegBackward::grad_formula(const Storage& g) {
    return negate_storage(g, shape_numel(out_shape_), dtype_, device_);
}

TensorImplPtr neg_op(const TensorImplPtr& a) {
    return NegBackward::forward(a);
}
LUCID_REGISTER_OP(NegBackward)

// --------------- Abs ---------------
const OpSchema AbsBackward::schema_v1{"abs", 1, AmpPolicy::Promote, true};

CpuStorage AbsBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch_unary(a, out_shape, dt, backend::cpu::vabs_f32, backend::cpu::vabs_f64, "abs");
}

Storage AbsBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage s = sign_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, s, n, dtype_, device_);
}

TensorImplPtr abs_op(const TensorImplPtr& a) {
    return AbsBackward::forward(a);
}
LUCID_REGISTER_OP(AbsBackward)

// --------------- Sign (no grad) ---------------
const OpSchema SignBackward::schema_v1{"sign", 1, AmpPolicy::KeepInput, true};

CpuStorage SignBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] > 0.f) - (p[i] < 0.f);
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] > 0.0) - (p[i] < 0.0);
            break;
        }
        default:
            ErrorBuilder("sign").not_implemented("dtype not supported");
    }
    return out;
}

Storage SignBackward::grad_formula(const Storage& g) {
    // Never called — kHasGradient = false.
    (void)g;
    return Storage{CpuStorage{}};
}

TensorImplPtr sign_op(const TensorImplPtr& a) {
    return SignBackward::forward(a);
}
LUCID_REGISTER_OP(SignBackward)

// --------------- Reciprocal ---------------
const OpSchema ReciprocalBackward::schema_v1{"reciprocal", 1, AmpPolicy::Promote, true};

CpuStorage ReciprocalBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    // 1/x via vForce vrec
    const std::size_t numel = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32:
            backend::cpu::vrec_f32(reinterpret_cast<const float*>(a.ptr.get()),
                                   reinterpret_cast<float*>(out.ptr.get()), numel);
            break;
        case Dtype::F64:
            backend::cpu::vrec_f64(reinterpret_cast<const double*>(a.ptr.get()),
                                   reinterpret_cast<double*>(out.ptr.get()), numel);
            break;
        default:
            ErrorBuilder("reciprocal").not_implemented("dtype not supported");
    }
    return out;
}

Storage ReciprocalBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = -1/x² * g
    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage g_div = divide_storages(g, x_sq, n, dtype_, device_);
    return negate_storage(g_div, n, dtype_, device_);
}

TensorImplPtr reciprocal_op(const TensorImplPtr& a) {
    return ReciprocalBackward::forward(a);
}
LUCID_REGISTER_OP(ReciprocalBackward)

// --------------- Square ---------------
const OpSchema SquareBackward::schema_v1{"square", 1, AmpPolicy::Promote, true};

CpuStorage SquareBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch_unary(a, out_shape, dt, backend::cpu::vsq_f32, backend::cpu::vsq_f64, "square");
}

Storage SquareBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = 2 * x * g
    Storage two_x = mul_scalar_storage(saved_inputs_[0], 2.0, n, dtype_, device_);
    return multiply_storages(two_x, g, n, dtype_, device_);
}

TensorImplPtr square_op(const TensorImplPtr& a) {
    return SquareBackward::forward(a);
}
LUCID_REGISTER_OP(SquareBackward)

// --------------- Cube ---------------
const OpSchema CubeBackward::schema_v1{"cube", 1, AmpPolicy::Promote, true};

CpuStorage CubeBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    // x^3 = x * x^2 — two passes is fine; vDSP doesn't have a direct cube.
    const std::size_t n = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<const float*>(a.ptr.get());
            auto* q = reinterpret_cast<float*>(out.ptr.get());
            backend::cpu::vsq_f32(p, q, n);
            backend::cpu::vmul_f32(p, q, q, n);
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<const double*>(a.ptr.get());
            auto* q = reinterpret_cast<double*>(out.ptr.get());
            backend::cpu::vsq_f64(p, q, n);
            backend::cpu::vmul_f64(p, q, q, n);
            break;
        }
        default:
            ErrorBuilder("cube").not_implemented("dtype not supported");
    }
    return out;
}

Storage CubeBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = 3 * x^2 * g
    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage three_xsq = mul_scalar_storage(x_sq, 3.0, n, dtype_, device_);
    return multiply_storages(three_xsq, g, n, dtype_, device_);
}

TensorImplPtr cube_op(const TensorImplPtr& a) {
    return CubeBackward::forward(a);
}
LUCID_REGISTER_OP(CubeBackward)

}  // namespace lucid
