// lucid/_C/ops/utils/Nextafter.cpp
//
// IEEE-754 next-representable-float computation.  The implementation is
// CPU-only because MLX exposes no equivalent kernel and the per-element
// std::nextafter call is cheap; GPU inputs are round-tripped through host
// memory.  Both inputs must share the same float dtype (F32 or F64) and shape.

#include "Nextafter.h"

#include <cmath>
#include <cstring>

#include "../../backend/Dispatcher.h"
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
