// lucid/_C/ops/gfunc/Gfunc.cpp
//
// Implementations of tensor-creation ("generator") ops.
//
// Shared implementation pattern:
//   1. Validate arguments (shape non-negative, step non-zero, etc.).
//   2. Construct an OpScopeFull for profiling and debug instrumentation.
//   3. Allocate or fill storage via the backend or Allocator:
//        - Constant fills (zeros, ones, full, empty): use make_zero_storage /
//          make_ones_storage / IBackend::full.
//        - Structured fills (eye): delegate entirely to IBackend::eye.
//        - Sequential fills (arange, linspace): fill a CPU buffer inline and
//          upload with IBackend::from_cpu so both CPU and GPU output work
//          without separate backend implementations.
//   4. Wrap the resulting Storage and metadata into a TensorImpl via finalize().
//
// The finalize() helper creates a TensorImpl from Storage + Shape + Dtype +
// Device + requires_grad.  It is used by every op in this file; factoring it
// out avoids repeating the make_shared<TensorImpl>(...) boilerplate.
//
// arange and linspace are implemented directly in C++ because expressing
// sequential fills through the backend interface would require either:
//   (a) a special-purpose IBackend::arange() that every backend must implement,
//   or (b) an MLX arange call that would require bridging to MLX array on the
//       CPU path even when the device is CPU.
// The CPU-fill-then-from_cpu approach is simpler, portable, and fast enough
// for the typical sizes involved.
//
// Note on empty_op: despite the name "empty", the current implementation
// zero-fills the buffer to prevent undefined reads.  This is a deliberate
// conservative choice; callers must not rely on the values being zero.

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

// Wrap storage and metadata into a TensorImpl.
//
// All generator ops in this file reduce to a single finalize() call at the
// end, which keeps the pattern uniform and makes it easy to spot-check that
// requires_grad, dtype, and device are forwarded correctly.
inline TensorImplPtr
finalize(Storage&& storage, Shape shape, Dtype dt, Device device, bool requires_grad) {
    return std::make_shared<TensorImpl>(std::move(storage), std::move(shape), dt, device,
                                        requires_grad);
}

}  // namespace

// Create a zero-filled tensor on the requested device.
//
// Delegates to make_zero_storage, which uses calloc-style zeroing on the CPU
// and MLX zeros on the GPU, both of which are more efficient than filling
// after allocation.
TensorImplPtr zeros_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"zeros", device, dt, shape};
    auto s = make_zero_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

// Create a ones-filled tensor on the requested device.
//
// Delegates to make_ones_storage, which on the CPU allocates then memsets,
// and on the GPU uses MLX ones or from_cpu with a pre-filled buffer.
TensorImplPtr ones_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"ones", device, dt, shape};
    auto s = make_ones_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

// Create a constant-filled tensor by delegating to IBackend::full().
//
// IBackend::full() handles dtype conversion (the fill_value is a double;
// the backend casts to the target dtype) and device placement in one call,
// avoiding a CPU alloc + upload round-trip for the GPU path.
TensorImplPtr
full_op(const Shape& shape, double fill_value, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"full", device, dt, shape};
    auto s = backend::Dispatcher::for_device(device).full(shape, dt, fill_value);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

// Create a zero-filled tensor; semantically "uninitialised" but safe.
//
// Zero-fills rather than leaving the memory uninitialised to prevent
// undefined-behaviour reads if a caller forgets to fill before use.
// The "empty" name signals contract: callers must not depend on the values.
TensorImplPtr empty_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"empty", device, dt, shape};
    auto s = make_zero_storage(shape, dt, device);
    return finalize(std::move(s), shape, dt, device, requires_grad);
}

// Create an N×M matrix with ones on diagonal k.
//
// Defaults M to N when M <= 0 to match numpy.eye behaviour.
// k=0 is the main diagonal; k>0 places ones above; k<0 places ones below.
// Both N and M must be non-negative.  IBackend::eye handles the actual fill.
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

// Create a 1-D tensor with arithmetic progression values.
//
// Computes the element count as ceil((stop - start) / step) and fills a CPU
// buffer with typed values, then routes through from_cpu so the result lands
// on the right device.  The element count formula matches numpy.arange exactly.
//
// When diff and step have opposite signs (e.g. start=5, stop=0, step=1), the
// range is empty and n=0, producing an empty 1-D tensor — not an error.
//
// The compute_cpu lambda is templated on pointer type via CTAD, avoiding
// a separate switch-inside-switch to handle the float/int cases.  Each case
// of the outer switch simply casts the raw buffer pointer to the correct type.
TensorImplPtr
arange_op(double start, double stop, double step, Dtype dt, Device device, bool requires_grad) {
    if (step == 0.0)
        ErrorBuilder("arange").fail("step must be non-zero");
    const double diff = stop - start;
    // If diff and step have opposite signs the range is empty; emit 0 elements.
    const std::int64_t n =
        (diff * step <= 0) ? 0 : static_cast<std::int64_t>(std::ceil(diff / step));
    Shape shape{n};
    OpScopeFull scope{"arange", device, dt, shape};

    // Fill p[i] = start + i * step, cast to the target element type.
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
    // from_cpu uploads the filled buffer to the requested device.
    return finalize(backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape), shape,
                    dt, device, requires_grad);
}

// Create a 1-D tensor with num evenly spaced values between start and stop
// (inclusive on both ends).
//
// The last element is explicitly pinned to stop to prevent floating-point
// drift: computing start + (num-1) * step can accumulate error that causes
// the last element to differ slightly from stop.  Pinning matches the
// numpy.linspace and torch.linspace behaviour.
//
// Special cases:
//   num == 0: returns an empty tensor (shape {0}).
//   num == 1: returns a tensor containing exactly [start]; stop is ignored
//             for the value computation and step is defined as 0.
TensorImplPtr linspace_op(
    double start, double stop, std::int64_t num, Dtype dt, Device device, bool requires_grad) {
    if (num < 0)
        ErrorBuilder("linspace").fail("num must be >= 0");
    Shape shape{num};
    OpScopeFull scope{"linspace", device, dt, shape};
    // step is only meaningful when num > 1; defined as 0 otherwise to avoid
    // division by zero when num == 1.
    const double step = (num > 1) ? (stop - start) / static_cast<double>(num - 1) : 0.0;

    auto compute_cpu = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < num; ++i) {
            // For num == 1, always emit start regardless of step.
            const double v = (num == 1) ? start : start + static_cast<double>(i) * step;
            p[i] = static_cast<T>(v);
        }
        // Pin the last element to stop to eliminate floating-point drift.
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

// Extract a diagonal from a 2-D matrix, or construct a 2-D matrix from a
// 1-D diagonal vector.
//
// Follows numpy.diag semantics:
//   - 1-D input of length m: output is (m+|k|) × (m+|k|) matrix with v on
//     diagonal k (zeros elsewhere).
//   - 2-D input of shape (r, c): output is the k-th diagonal as a 1-D vector
//     of length min(r, c, r-k, c+k) (clipped to the valid range).
//
// The actual shape computation and fill logic lives in IBackend::diag().
// out_shape is passed by reference and filled by the backend so this wrapper
// can create the TensorImpl with the correct shape without recomputing it.
// requires_grad is always false because diag is not differentiable through
// autograd in the current implementation.
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
    // The backend fills out_shape to reflect the actual output dimensions.
    auto s = backend::Dispatcher::for_device(device).diag(v->storage(), sh, k, dt, out_shape);
    return std::make_shared<TensorImpl>(std::move(s), std::move(out_shape), dt, device, false);
}

// Create a zero tensor with the same shape, dtype, and device as a.
//
// Delegates to zeros_op with the metadata extracted from a.
TensorImplPtr zeros_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "zeros_like.a").non_null();
    return zeros_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

// Create a ones tensor with the same shape, dtype, and device as a.
TensorImplPtr ones_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "ones_like.a").non_null();
    return ones_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

// Create an uninitialised (zero-filled in practice) tensor with the same
// shape, dtype, and device as a.
TensorImplPtr empty_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "empty_like.a").non_null();
    return empty_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

// Create a constant-filled tensor with the same shape, dtype, and device as a.
TensorImplPtr full_like_op(const TensorImplPtr& a, double fill_value, bool requires_grad) {
    Validator::input(a, "full_like.a").non_null();
    return full_op(a->shape(), fill_value, a->dtype(), a->device(), requires_grad);
}

}  // namespace lucid
