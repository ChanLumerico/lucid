#include "Discrete.h"

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
CpuStorage dispatch_float(const CpuStorage& a, const Shape& out_shape, Dtype dt,
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
            throw NotImplementedError(std::string(op) + ": dtype not supported (float-only)");
    }
    return out;
}
}  // namespace

// --------------- Round (banker's, half-to-even) ---------------
const OpSchema RoundBackward::schema_v1{"round", 1, AmpPolicy::KeepInput, true};

CpuStorage RoundBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch_float(a, out_shape, dt,
                          backend::cpu::vround_f32, backend::cpu::vround_f64, "round");
}
Storage RoundBackward::grad_formula(const Storage&) { return Storage{CpuStorage{}}; }
TensorImplPtr round_op(const TensorImplPtr& a) { return RoundBackward::forward(a); }
LUCID_REGISTER_OP(RoundBackward)

// --------------- Floor ---------------
const OpSchema FloorBackward::schema_v1{"floor", 1, AmpPolicy::KeepInput, true};

CpuStorage FloorBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch_float(a, out_shape, dt,
                          backend::cpu::vfloor_f32, backend::cpu::vfloor_f64, "floor");
}
Storage FloorBackward::grad_formula(const Storage&) { return Storage{CpuStorage{}}; }
TensorImplPtr floor_op(const TensorImplPtr& a) { return FloorBackward::forward(a); }
LUCID_REGISTER_OP(FloorBackward)

// --------------- Ceil ---------------
const OpSchema CeilBackward::schema_v1{"ceil", 1, AmpPolicy::KeepInput, true};

CpuStorage CeilBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    return dispatch_float(a, out_shape, dt,
                          backend::cpu::vceil_f32, backend::cpu::vceil_f64, "ceil");
}
Storage CeilBackward::grad_formula(const Storage&) { return Storage{CpuStorage{}}; }
TensorImplPtr ceil_op(const TensorImplPtr& a) { return CeilBackward::forward(a); }
LUCID_REGISTER_OP(CeilBackward)

// --------------- Invert (bitwise NOT, integer-only) ---------------
const OpSchema InvertBackward::schema_v1{"invert", 1, AmpPolicy::KeepInput, true};

CpuStorage InvertBackward::cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt) {
    const std::size_t numel = shape_numel(out_shape);
    auto out = allocate_unary(out_shape, dt);
    switch (dt) {
        case Dtype::I8: {
            auto* p = reinterpret_cast<const std::int8_t*>(a.ptr.get());
            auto* q = reinterpret_cast<std::int8_t*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) q[i] = static_cast<std::int8_t>(~p[i]);
            break;
        }
        case Dtype::I16: {
            auto* p = reinterpret_cast<const std::int16_t*>(a.ptr.get());
            auto* q = reinterpret_cast<std::int16_t*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) q[i] = static_cast<std::int16_t>(~p[i]);
            break;
        }
        case Dtype::I32: {
            auto* p = reinterpret_cast<const std::int32_t*>(a.ptr.get());
            auto* q = reinterpret_cast<std::int32_t*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) q[i] = ~p[i];
            break;
        }
        case Dtype::I64: {
            auto* p = reinterpret_cast<const std::int64_t*>(a.ptr.get());
            auto* q = reinterpret_cast<std::int64_t*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) q[i] = ~p[i];
            break;
        }
        case Dtype::Bool: {
            auto* p = reinterpret_cast<const std::uint8_t*>(a.ptr.get());
            auto* q = reinterpret_cast<std::uint8_t*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i) q[i] = !p[i];
            break;
        }
        default:
            throw NotImplementedError("invert: integer / bool dtype only");
    }
    return out;
}
Storage InvertBackward::grad_formula(const Storage&) { return Storage{CpuStorage{}}; }
TensorImplPtr invert_op(const TensorImplPtr& a) { return InvertBackward::forward(a); }
LUCID_REGISTER_OP(InvertBackward)

}  // namespace lucid
