#include "Gfunc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <variant>

#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"

namespace lucid {

namespace {

using helpers::allocate_cpu;

inline TensorImplPtr
finalize(Storage&& storage, Shape shape, Dtype dt, Device device, bool requires_grad) {
    return std::make_shared<TensorImpl>(std::move(storage), std::move(shape), dt, device,
                                        requires_grad);
}

}  // namespace

TensorImplPtr zeros_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"zeros", device, dt, shape};
    auto s = make_zero_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

TensorImplPtr ones_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"ones", device, dt, shape};
    auto s = make_ones_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

TensorImplPtr
full_op(const Shape& shape, double fill_value, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"full", device, dt, shape};
    auto s = backend::Dispatcher::for_device(device).full(shape, dt, fill_value);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

TensorImplPtr empty_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"empty", device, dt, shape};
    auto s = make_zero_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

TensorImplPtr eye_op(
    std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt, Device device, bool requires_grad) {
    if (M <= 0)
        M = N;
    if (N < 0 || M < 0)
        ErrorBuilder("eye").fail("N and M must be >= 0");
    Shape shape{N, M};
    OpScopeFull scope{"eye", device, dt, shape};
    auto s = backend::Dispatcher::for_device(device).eye(N, M, k, dt);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

TensorImplPtr
arange_op(double start, double stop, double step, Dtype dt, Device device, bool requires_grad) {
    if (step == 0.0)
        ErrorBuilder("arange").fail("step must be non-zero");
    const double diff = stop - start;
    const std::int64_t n =
        (diff * step <= 0) ? 0 : static_cast<std::int64_t>(std::ceil(diff / step));
    Shape shape{n};
    OpScopeFull scope{"arange", device, dt, shape};

    auto compute_cpu = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<T>(start + static_cast<double>(i) * step);
        }
    };

    auto cpu = allocate_cpu(shape, dt);
    switch (dt) {
    case Dtype::F32:
        compute_cpu(reinterpret_cast<float*>(cpu.ptr.get()));
        break;
    case Dtype::F64:
        compute_cpu(reinterpret_cast<double*>(cpu.ptr.get()));
        break;
    case Dtype::I32:
        compute_cpu(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
        break;
    case Dtype::I64:
        compute_cpu(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
        break;
    default:
        ErrorBuilder("arange").not_implemented("dtype not supported");
    }
    return finalize(backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape), shape,
                    dt, device, requires_grad);
}

TensorImplPtr linspace_op(
    double start, double stop, std::int64_t num, Dtype dt, Device device, bool requires_grad) {
    if (num < 0)
        ErrorBuilder("linspace").fail("num must be >= 0");
    Shape shape{num};
    OpScopeFull scope{"linspace", device, dt, shape};
    const double step = (num > 1) ? (stop - start) / static_cast<double>(num - 1) : 0.0;

    auto compute_cpu = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < num; ++i) {
            const double v = (num == 1) ? start : start + static_cast<double>(i) * step;
            p[i] = static_cast<T>(v);
        }

        if (num >= 2)
            p[num - 1] = static_cast<T>(stop);
    };

    auto cpu = allocate_cpu(shape, dt);
    switch (dt) {
    case Dtype::F32:
        compute_cpu(reinterpret_cast<float*>(cpu.ptr.get()));
        break;
    case Dtype::F64:
        compute_cpu(reinterpret_cast<double*>(cpu.ptr.get()));
        break;
    case Dtype::I32:
        compute_cpu(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
        break;
    case Dtype::I64:
        compute_cpu(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
        break;
    default:
        ErrorBuilder("linspace").not_implemented("dtype not supported");
    }
    return finalize(backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape), shape,
                    dt, device, requires_grad);
}

TensorImplPtr diag_op(const TensorImplPtr& v, std::int64_t k) {
    if (!v)
        ErrorBuilder("diag").fail("input is null");
    const Dtype dt = v->dtype();
    const Device device = v->device();
    const auto& sh = v->shape();
    if (sh.size() != 1 && sh.size() != 2) {
        ErrorBuilder("diag").fail("input must be 1-D or 2-D");
    }

    Shape out_shape;
    auto s = backend::Dispatcher::for_device(device).diag(v->storage(), sh, k, dt, out_shape);
    return std::make_shared<TensorImpl>(std::move(s), std::move(out_shape), dt, device, false);
}

TensorImplPtr zeros_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "zeros_like.a").non_null();
    return zeros_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

TensorImplPtr ones_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "ones_like.a").non_null();
    return ones_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

TensorImplPtr empty_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "empty_like.a").non_null();
    return empty_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

TensorImplPtr full_like_op(const TensorImplPtr& a, double fill_value, bool requires_grad) {
    Validator::input(a, "full_like.a").non_null();
    return full_op(a->shape(), fill_value, a->dtype(), a->device(), requires_grad);
}

}  // namespace lucid
