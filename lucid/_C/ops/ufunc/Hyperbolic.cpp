#include "Hyperbolic.h"

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
            ErrorBuilder(op).not_implemented("dtype not supported");
    }
    return out;
}
}  // namespace

// --------------- Sinh ---------------
const OpSchema SinhBackward::schema_v1{"sinh", 1, AmpPolicy::Promote, true};

CpuStorage SinhBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vsinh_f32, backend::cpu::vsinh_f64, "sinh");
}

Storage SinhBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage cx = cosh_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, cx, n, dtype_, device_);
}

TensorImplPtr sinh_op(const TensorImplPtr& a) {
    return SinhBackward::forward(a);
}
LUCID_REGISTER_OP(SinhBackward)

// --------------- Cosh ---------------
const OpSchema CoshBackward::schema_v1{"cosh", 1, AmpPolicy::Promote, true};

CpuStorage CoshBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vcosh_f32, backend::cpu::vcosh_f64, "cosh");
}

Storage CoshBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage sx = sinh_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, sx, n, dtype_, device_);
}

TensorImplPtr cosh_op(const TensorImplPtr& a) {
    return CoshBackward::forward(a);
}
LUCID_REGISTER_OP(CoshBackward)

// --------------- Tanh ---------------
const OpSchema TanhBackward::schema_v1{"tanh", 1, AmpPolicy::Promote, true};

CpuStorage TanhBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch(a, out_shape, dt, backend::cpu::vtanh_f32, backend::cpu::vtanh_f64, "tanh");
}

Storage TanhBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = (1 - z²) * g, z = output
    Storage z_sq = square_storage(saved_output_, n, dtype_, device_);
    Storage neg_zsq = mul_scalar_storage(z_sq, -1.0, n, dtype_, device_);
    Storage one_minus = add_scalar_storage(neg_zsq, 1.0, n, dtype_, device_);
    return multiply_storages(g, one_minus, n, dtype_, device_);
}

TensorImplPtr tanh_op(const TensorImplPtr& a) {
    return TanhBackward::forward(a);
}
LUCID_REGISTER_OP(TanhBackward)

}  // namespace lucid
