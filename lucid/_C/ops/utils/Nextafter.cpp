// lucid/_C/ops/utils/Nextafter.cpp
//
// IEEE-754 next-representable-float computation.  Two paths:
//   * CPU (F32 or F64): per-element std::nextafter loop.
//   * GPU (F32 only): pure MLX bit-manipulation pipeline using view()
//     to reinterpret the float32 buffer as int32, applying the
//     IEEE-754 next-toward rule via where + add, and viewing back.
//     F64 is not supported on Metal (MLX limitation) — F64 GPU inputs
//     fall through the existing CPU round-trip path.
//
// Both inputs must share the same float dtype and shape.

#include "Nextafter.h"

#include <cmath>
#include <cstring>
#include <limits>

#include <mlx/ops.h>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;

CpuStorage to_cpu(const TensorImplPtr& a) {
    return backend::Dispatcher::for_device(a->device()).to_cpu(a->storage(), a->shape());
}

Storage to_device_storage(CpuStorage&& cpu, Device target_device, const Shape& shape) {
    if (target_device == Device::GPU && cpu.dtype != Dtype::F64) {
        return backend::Dispatcher::for_device(Device::GPU).from_cpu(cpu, shape);
    }
    return Storage{std::move(cpu)};
}

// Element-wise std::nextafter loop, templated over the float type.
template <typename T>
void run_nextafter(const T* a, const T* b, T* dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = std::nextafter(a[i], b[i]);
}

// Engine-side IEEE-754 next-toward-b for F32 on Metal/MLX.  The algorithm
// is a pure-elementwise bit twiddle:
//
//   * If either input is NaN → NaN.
//   * If a == b (including ±0 == ±0) → b.
//   * If a == 0 (and b is finite, non-equal): return ±smallest positive
//     subnormal with sign matching b.
//   * Otherwise: convert a to its int32 bit pattern, increment the bits
//     toward b (decrement when a is negative), bitcast back.
//
// Returns the result as a GpuStorage (F32, GPU device).
GpuStorage nextafter_gpu_f32(const GpuStorage& a, const GpuStorage& b) {
    namespace mx = ::mlx::core;
    auto& af = *a.arr;
    auto& bf = *b.arr;

    // Bit reinterpretation: f32 ↔ int32.
    auto a_bits = mx::view(af, mx::int32);
    auto b_bits = mx::view(bf, mx::int32);

    auto zero_f = mx::array(0.0f, mx::float32);
    auto one_i = mx::array(static_cast<std::int32_t>(1), mx::int32);
    auto neg_one_i = mx::array(static_cast<std::int32_t>(-1), mx::int32);

    // direction = +1 when b > a, else -1.  delta = direction when a > 0,
    // else -direction (negative-a bits move opposite the float ordering).
    auto direction = mx::where(mx::greater(bf, af), one_i, neg_one_i);
    auto a_positive = mx::greater(af, zero_f);
    auto delta = mx::where(a_positive, direction, mx::negative(direction));

    auto general = mx::add(a_bits, delta);

    // a == 0 special case: smallest +subnormal toward positive b, or
    // smallest -subnormal toward negative b.
    auto smallest_pos = mx::array(static_cast<std::int32_t>(1), mx::int32);
    auto smallest_neg = mx::array(static_cast<std::int32_t>(0x80000001u), mx::int32);
    auto a_zero = mx::equal(af, zero_f);
    auto b_pos = mx::greater(bf, zero_f);
    auto zero_branch = mx::where(b_pos, smallest_pos, smallest_neg);

    auto step = mx::where(a_zero, zero_branch, general);

    // a == b → b's bits (preserves sign-of-zero).
    auto eq = mx::equal(af, bf);
    step = mx::where(eq, b_bits, step);

    // NaN propagation: f != f.
    auto a_nan = mx::not_equal(af, af);
    auto b_nan = mx::not_equal(bf, bf);
    auto any_nan = mx::logical_or(a_nan, b_nan);
    auto nan_bits =
        mx::view(mx::array(std::numeric_limits<float>::quiet_NaN(), mx::float32), mx::int32);
    step = mx::where(any_nan, nan_bits, step);

    auto out = mx::view(step, mx::float32);
    return gpu::wrap_mlx_array(::mlx::core::contiguous(out), Dtype::F32);
}

}  // namespace

TensorImplPtr nextafter_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    Validator::input(a, "nextafter.a").non_null();
    Validator::input(b, "nextafter.b").non_null();
    if (a->shape() != b->shape())
        throw ShapeMismatch(a->shape(), b->shape(), "nextafter");
    if (a->dtype() != b->dtype())
        throw DtypeMismatch(std::string(dtype_name(a->dtype())),
                            std::string(dtype_name(b->dtype())), "nextafter");
    if (a->dtype() != Dtype::F32 && a->dtype() != Dtype::F64)
        ErrorBuilder("nextafter").fail("dtype must be F32 or F64");

    OpScopeFull scope{"nextafter", a->device(), a->dtype(), a->shape()};

    // Metal/F32 fast path: bit-twiddle on MLX, no CPU round-trip.
    if (a->device() == Device::GPU && a->dtype() == Dtype::F32) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        const auto& gb = std::get<GpuStorage>(b->storage());
        return fresh(Storage{nextafter_gpu_f32(ga, gb)}, a->shape(), a->dtype(), Device::GPU);
    }

    // CPU fallback (and F64 on either device — MLX doesn't support F64).
    const auto ca = to_cpu(a);
    const auto cb = to_cpu(b);
    const std::size_t n = shape_numel(a->shape());
    auto out = allocate_cpu(a->shape(), a->dtype());

    if (a->dtype() == Dtype::F32) {
        run_nextafter(reinterpret_cast<const float*>(ca.ptr.get()),
                      reinterpret_cast<const float*>(cb.ptr.get()),
                      reinterpret_cast<float*>(out.ptr.get()), n);
    } else {
        run_nextafter(reinterpret_cast<const double*>(ca.ptr.get()),
                      reinterpret_cast<const double*>(cb.ptr.get()),
                      reinterpret_cast<double*>(out.ptr.get()), n);
    }

    Storage final_storage = to_device_storage(std::move(out), a->device(), a->shape());
    return fresh(std::move(final_storage), a->shape(), a->dtype(), a->device());
}

}  // namespace lucid
