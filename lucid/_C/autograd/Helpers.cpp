// lucid/_C/autograd/Helpers.cpp
//
// Implements the utility functions declared in Helpers.h.  The file is split
// into three areas:
//   1. CPU in-place accumulation (typed loops + overloaded visitor).
//   2. Thin dispatcher wrappers for element-wise and reduction operations.
//   3. CPU-side random number generators (uniform, normal, Bernoulli, randint)
//      that produce CpuStorage buffers then transfer to the target device.

#include "Helpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "../backend/Dispatcher.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/Generator.h"
#include "../core/TensorImpl.h"

namespace lucid {

namespace {

// C++17 overloaded-lambda helper for std::visit.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Typed in-place addition loop: dst[i] += src[i] for numel elements of type T.
template <typename T>
void add_typed(std::byte* dst, const std::byte* src, std::size_t numel) {
    auto* td = reinterpret_cast<T*>(dst);
    const auto* ts = reinterpret_cast<const T*>(src);
    for (std::size_t i = 0; i < numel; ++i)
        td[i] = td[i] + ts[i];
}

// Perform dst += src for two CpuStorage buffers.
// Validates dtype and byte-count equality, then dispatches to add_typed<T>.
void cpu_add_inplace(CpuStorage& dst, const CpuStorage& src) {
    if (dst.dtype != src.dtype) {
        throw DtypeMismatch(std::string(dtype_name(dst.dtype)), std::string(dtype_name(src.dtype)),
                            "accumulate_into");
    }
    if (dst.nbytes != src.nbytes) {
        ErrorBuilder("accumulate_into").fail("nbytes mismatch");
    }
    const std::size_t n = dst.nbytes / dtype_size(dst.dtype);
    switch (dst.dtype) {
    case Dtype::F32:
        add_typed<float>(dst.ptr.get(), src.ptr.get(), n);
        break;
    case Dtype::F64:
        add_typed<double>(dst.ptr.get(), src.ptr.get(), n);
        break;
    case Dtype::I32:
        add_typed<std::int32_t>(dst.ptr.get(), src.ptr.get(), n);
        break;
    case Dtype::I64:
        add_typed<std::int64_t>(dst.ptr.get(), src.ptr.get(), n);
        break;
    default:
        ErrorBuilder("accumulate_into").not_implemented("dtype not yet supported in Phase 2");
    }
}

}  // namespace

Storage make_zero_storage(const Shape& shape, Dtype dtype, Device device) {
    return backend::Dispatcher::for_device(device).zeros(shape, dtype);
}

Storage make_ones_storage(const Shape& shape, Dtype dtype, Device device) {
    return backend::Dispatcher::for_device(device).ones(shape, dtype);
}

Storage reduce_grad_to_shape(const Storage& grad,
                             const Shape& grad_shape,
                             const Shape& target_shape,
                             Dtype dtype,
                             Device device) {
    return backend::Dispatcher::for_device(device).reduce_grad_to_shape(grad, grad_shape,
                                                                        target_shape, dtype);
}

// Dispatch accumulation to the appropriate device handler using std::visit.
//
// The visitor handles five cases:
//   CPU+CPU          — direct typed loop via cpu_add_inplace.
//   GPU+GPU          — MLX add, rewrap result into the destination GpuStorage.
//   SharedStorage+SharedStorage — obtain cpu_view() of both, then cpu_add_inplace.
//   SharedStorage+CpuStorage   — view the shared buffer as CPU, then add.
//   CpuStorage+SharedStorage   — view the shared source as CPU, then add.
// Any other combination (CPU+GPU, GPU+CPU, etc.) throws DeviceMismatch.
void accumulate_into(Storage& dst, const Storage& src) {
    std::visit(overloaded{
                   [&](CpuStorage& d, const CpuStorage& s) { cpu_add_inplace(d, s); },
                   [&](GpuStorage& d, const GpuStorage& s) {
                       if (!d.arr || !s.arr) {
                           ErrorBuilder("accumulate_into").fail("null GPU array");
                       }
                       if (d.dtype != s.dtype) {
                           throw DtypeMismatch(std::string(dtype_name(d.dtype)),
                                               std::string(dtype_name(s.dtype)), "accumulate_into");
                       }
                       // MLX add produces a new array; replace the destination's
                       // arr pointer in-place so callers holding a reference to
                       // d see the updated value.
                       auto next = ::mlx::core::add(*d.arr, *s.arr);
                       d.arr = gpu::wrap_mlx_array(std::move(next), d.dtype).arr;
                   },

                   [&](SharedStorage& d, const SharedStorage& s) {
                       auto dv = d.cpu_view();
                       auto sv = s.cpu_view();
                       cpu_add_inplace(dv, sv);
                   },
                   [&](SharedStorage& d, const CpuStorage& s) {
                       auto dv = d.cpu_view();
                       cpu_add_inplace(dv, s);
                   },
                   [&](CpuStorage& d, const SharedStorage& s) {
                       auto sv = s.cpu_view();
                       cpu_add_inplace(d, sv);
                   },
                   [&](auto&, auto&) {
                       throw DeviceMismatch("matching device", "mixed CPU/GPU", "accumulate_into");
                   },
               },
               dst, src);
}

// -------------------------------------------------------------------------
// Element-wise arithmetic helpers.
// Each function constructs a flat 1-D Shape from numel and delegates to the
// Dispatcher so that the correct backend (CPU/GPU) is used automatically.
// -------------------------------------------------------------------------

Storage negate_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).neg(s, flat, dt);
}

Storage
multiply_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).mul(a, b, flat, dt);
}

Storage
divide_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).div(a, b, flat, dt);
}

Storage
add_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).add(a, b, flat, dt);
}

Storage
subtract_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).sub(a, b, flat, dt);
}

Storage square_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).square(s, flat, dt);
}

Storage clone_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).clone(s, flat, dt);
}

Storage log_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).log(s, flat, dt);
}

Storage
pow_storage(const Storage& base, const Storage& expo, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).pow(base, expo, flat, dt);
}

Storage
ge_mask_storage(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).ge_mask(a, b, flat, dt);
}

Storage
lt_mask_storage(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).lt_mask(a, b, flat, dt);
}

Storage
add_scalar_storage(const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).add_scalar(s, flat, dt, scalar);
}

Storage
mul_scalar_storage(const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).mul_scalar(s, flat, dt, scalar);
}

// Macro-generated unary helpers.  Each expands to a free function that
// wraps numel into a flat Shape and calls the matching Dispatcher method.
#define LUCID_UNARY_HELPER(NAME, BACKEND_METHOD)                                                   \
    Storage NAME##_storage(const Storage& s, std::size_t n, Dtype dt, Device device) {             \
        const Shape flat{static_cast<std::int64_t>(n)};                                            \
        return backend::Dispatcher::for_device(device).BACKEND_METHOD(s, flat, dt);                \
    }

LUCID_UNARY_HELPER(exp, exp)
LUCID_UNARY_HELPER(sqrt, sqrt)
LUCID_UNARY_HELPER(abs, abs)
LUCID_UNARY_HELPER(reciprocal, reciprocal)
LUCID_UNARY_HELPER(sin, sin)
LUCID_UNARY_HELPER(cos, cos)
LUCID_UNARY_HELPER(asin, asin)
LUCID_UNARY_HELPER(acos, acos)
LUCID_UNARY_HELPER(atan, atan)
LUCID_UNARY_HELPER(sinh, sinh)
LUCID_UNARY_HELPER(cosh, cosh)
LUCID_UNARY_HELPER(tanh, tanh)

#undef LUCID_UNARY_HELPER

// tan and sign are defined outside the macro because they share the same
// dispatcher method name as other functions (avoiding collision) or because
// there is no 1-to-1 macro match.
Storage tan_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).tan(s, flat, dt);
}

Storage sign_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).sign(s, flat, dt);
}

Storage in_range_mask_storage(
    const Storage& s, double lo, double hi, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).in_range_mask(s, flat, dt, lo, hi);
}

// -------------------------------------------------------------------------
// Version counter validation.
// -------------------------------------------------------------------------

// Process-wide flag controlled by ``lucid.autograd.graph.allow_mutation_on_
// saved_tensors``.  When set, ``check_version_match`` becomes a no-op — the
// user has explicitly opted into the unsafe contract that they will not
// mutate saved tensors in a way that would corrupt the gradient.  The flag
// is intentionally a plain ``bool`` (no atomic / thread-local): autograd is
// already a single-threaded contract in Lucid and the cost of synchronisation
// would dwarf the actual check.
static bool g_allow_mutation_on_saved = false;

bool is_mutation_on_saved_allowed() {
    return g_allow_mutation_on_saved;
}

void set_mutation_on_saved_allowed(bool v) {
    g_allow_mutation_on_saved = v;
}

// Compare the version counter of the live TensorImpl against the saved value.
// A discrepancy means the tensor was mutated in-place between the forward pass
// and this backward call — which invalidates the saved activation data.
void check_version_match(const std::weak_ptr<TensorImpl>& live,
                         std::int64_t saved_version,
                         std::string_view op_name,
                         std::size_t input_idx) {
    if (g_allow_mutation_on_saved) {
        return;  // user opted into the unsafe contract
    }
    auto t = live.lock();
    if (!t)
        return;
    if (t->version() != saved_version) {
        throw VersionMismatch(saved_version, t->version(),
                              std::string(op_name) + " input " + std::to_string(input_idx));
    }
}

// -------------------------------------------------------------------------
// Shape / axis helpers for reduction backward passes.
// -------------------------------------------------------------------------

// Resolve axis indices to a sorted, deduplicated list of non-negative values.
// Negative indices are wrapped by adding ndim.  An empty input axes list is
// interpreted as "all axes".
std::vector<int> normalize_axes(const std::vector<int>& axes, int ndim) {
    std::vector<int> out;
    if (axes.empty()) {
        out.reserve(ndim);
        for (int i = 0; i < ndim; ++i)
            out.push_back(i);
        return out;
    }
    std::vector<bool> seen(ndim, false);
    for (int a : axes) {
        int wrapped = a < 0 ? a + ndim : a;
        if (wrapped < 0 || wrapped >= ndim) {
            ErrorBuilder("normalize_axes")
                .index_error(std::string("axis out of range: ") + std::to_string(a) +
                             " for ndim=" + std::to_string(ndim));
        }
        if (!seen[wrapped]) {
            seen[wrapped] = true;
            out.push_back(wrapped);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Build the output shape of a reduction, applying keepdims semantics.
// Reduced dimensions either disappear (keepdims=false) or become 1.
Shape reduce_output_shape(const Shape& input_shape, const std::vector<int>& axes, bool keepdims) {
    std::vector<bool> reduce_mask(input_shape.size(), false);
    for (int a : axes)
        reduce_mask[a] = true;
    Shape out;
    out.reserve(input_shape.size());
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (reduce_mask[i]) {
            if (keepdims)
                out.push_back(1);
            // else: dimension is dropped
        } else {
            out.push_back(input_shape[i]);
        }
    }
    return out;
}

// Expand grad back from the reduced shape to input_shape so that it can be
// added to the input's gradient buffer.
//
// The expected shape of grad is validated against reduce_output_shape() to
// catch mismatches early.  The actual broadcasting is delegated to the
// Dispatcher's broadcast_back_for_reduce(), which handles both CPU and GPU.
Storage broadcast_back_for_reduce(const Storage& grad,
                                  const Shape& grad_shape,
                                  const Shape& input_shape,
                                  const std::vector<int>& axes,
                                  bool keepdims,
                                  Dtype dt,
                                  Device device) {
    Shape expected_grad = reduce_output_shape(input_shape, axes, keepdims);
    if (grad_shape != expected_grad) {
        throw ShapeMismatch(expected_grad, grad_shape,
                            "broadcast_back_for_reduce: grad shape mismatch");
    }
    return backend::Dispatcher::for_device(device).broadcast_back_for_reduce(
        grad, grad_shape, input_shape, axes, keepdims, dt);
}

Storage
leaky_mask_storage(const Storage& s, double slope, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).leaky_mask(s, flat, dt, slope);
}

Storage sigmoid_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).sigmoid(s, flat, dt);
}

// -------------------------------------------------------------------------
// Random tensor generation — all produced on CPU then transferred to device.
// -------------------------------------------------------------------------

// Generate a Bernoulli mask with explicit shape.  Samples are drawn on CPU
// using the Generator's uniform float stream, then transferred to the target
// device via Dispatcher::from_cpu().
Storage bernoulli_mask_storage_shape(
    double keep_prob, const Shape& shape, Dtype dt, Device device, Generator& gen) {
    if (keep_prob < 0.0 || keep_prob > 1.0) {
        ErrorBuilder("bernoulli_mask").fail("keep_prob must be in [0, 1]");
    }
    std::size_t numel = 1;
    for (auto d : shape)
        numel *= static_cast<std::size_t>(d);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = numel * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    switch (dt) {
    case Dtype::F32: {
        auto* q = reinterpret_cast<float*>(out.ptr.get());
        const auto kp = static_cast<float>(keep_prob);
        for (std::size_t i = 0; i < numel; ++i) {
            q[i] = (gen.next_uniform_float() < kp) ? 1.f : 0.f;
        }
        break;
    }
    case Dtype::F64: {
        auto* q = reinterpret_cast<double*>(out.ptr.get());
        for (std::size_t i = 0; i < numel; ++i) {
            q[i] = (static_cast<double>(gen.next_uniform_float()) < keep_prob) ? 1.0 : 0.0;
        }
        break;
    }
    default:
        ErrorBuilder("bernoulli_mask").not_implemented("dtype not supported (F32/F64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(out), shape);
}

// Flat-numel convenience wrapper around bernoulli_mask_storage_shape.
Storage bernoulli_mask_storage(
    double keep_prob, std::size_t numel, Dtype dt, Device device, Generator& gen) {
    Shape flat;
    flat.push_back(static_cast<std::int64_t>(numel));
    return bernoulli_mask_storage_shape(keep_prob, flat, dt, device, gen);
}

Storage positive_mask_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).positive_mask(s, flat, dt);
}

namespace {

// Fill dst[0..numel) with uniform floats scaled to [lo, hi).
template <typename T>
void fill_uniform(T* dst, std::size_t numel, double lo, double hi, Generator& gen) {
    const T span = static_cast<T>(hi - lo);
    const T offset = static_cast<T>(lo);
    for (std::size_t i = 0; i < numel; ++i) {
        dst[i] = static_cast<T>(gen.next_uniform_float()) * span + offset;
    }
}

// Fill dst[0..numel) with samples from N(mean, std) using the Box-Muller
// transform.  Pairs of uniform samples produce two normal samples, so the
// loop processes two elements at a time with a tail case for odd numel.
// u1 is clamped away from zero to prevent log(0).
template <typename T>
void fill_normal(T* dst, std::size_t numel, double mean, double std, Generator& gen) {
    const T m = static_cast<T>(mean);
    const T s = static_cast<T>(std);
    constexpr T two_pi = static_cast<T>(6.28318530717958647692);
    constexpr T eps = static_cast<T>(1e-7);
    std::size_t i = 0;
    while (i + 1 < numel) {
        T u1 = static_cast<T>(gen.next_uniform_float());
        if (u1 < eps)
            u1 = eps;
        T u2 = static_cast<T>(gen.next_uniform_float());
        const T r = std::sqrt(static_cast<T>(-2) * std::log(u1));
        const T z0 = r * std::cos(two_pi * u2);
        const T z1 = r * std::sin(two_pi * u2);
        dst[i] = m + s * z0;
        dst[i + 1] = m + s * z1;
        i += 2;
    }
    if (i < numel) {
        // Odd element: use only z0 from the final Box-Muller pair.
        T u1 = static_cast<T>(gen.next_uniform_float());
        if (u1 < eps)
            u1 = eps;
        T u2 = static_cast<T>(gen.next_uniform_float());
        const T r = std::sqrt(static_cast<T>(-2) * std::log(u1));
        dst[i] = m + s * r * std::cos(two_pi * u2);
    }
}

// Fill dst[0..numel) with integers uniformly drawn from [low, high).
// When range fits in 32 bits, a single uint32 is used per element.
// When range exceeds 32 bits, two consecutive uint32 values are combined
// into a 64-bit value.  The Generator is queried in batches of four uint32s
// to amortise call overhead.
template <typename Int>
void fill_randint(
    Int* dst, std::size_t numel, std::int64_t low, std::int64_t high, Generator& gen) {
    const std::uint64_t range = static_cast<std::uint64_t>(high - low);
    if (range == 0) {
        for (std::size_t i = 0; i < numel; ++i)
            dst[i] = static_cast<Int>(low);
        return;
    }
    std::uint32_t buf[4];
    std::size_t i = 0;
    while (i < numel) {
        gen.next_uint32x4(buf);
        for (int k = 0; k < 4 && i < numel; ++k, ++i) {
            std::uint64_t r = buf[k];

            if (range > 0xFFFFFFFFull) {
                // Combine two 32-bit words for large ranges.
                std::uint32_t buf2[4];
                gen.next_uint32x4(buf2);
                r = (r << 32) | buf2[0];
            }
            dst[i] = static_cast<Int>(low) + static_cast<Int>(r % range);
        }
    }
}

// Allocate an uninitialized CpuStorage sized for shape and dtype.
CpuStorage allocate_for_random(const Shape& shape, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = shape_numel(shape) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

}  // namespace

Storage random_uniform_storage(
    const Shape& shape, double lo, double hi, Dtype dt, Device device, Generator& gen) {
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
    case Dtype::F32:
        fill_uniform<float>(reinterpret_cast<float*>(cpu.ptr.get()), n, lo, hi, gen);
        break;
    case Dtype::F64:
        fill_uniform<double>(reinterpret_cast<double*>(cpu.ptr.get()), n, lo, hi, gen);
        break;
    default:
        ErrorBuilder("random_uniform").not_implemented("dtype not supported (F32/F64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

Storage random_normal_storage(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator& gen) {
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
    case Dtype::F32:
        fill_normal<float>(reinterpret_cast<float*>(cpu.ptr.get()), n, mean, std, gen);
        break;
    case Dtype::F64:
        fill_normal<double>(reinterpret_cast<double*>(cpu.ptr.get()), n, mean, std, gen);
        break;
    default:
        ErrorBuilder("random_normal").not_implemented("dtype not supported (F32/F64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

Storage
random_bernoulli_storage(const Shape& shape, double p, Dtype dt, Device device, Generator& gen) {
    if (p < 0.0 || p > 1.0)
        ErrorBuilder("random_bernoulli").fail("p must be in [0, 1]");
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
    case Dtype::F32: {
        auto* q = reinterpret_cast<float*>(cpu.ptr.get());
        const auto fp = static_cast<float>(p);
        for (std::size_t i = 0; i < n; ++i) {
            q[i] = (gen.next_uniform_float() < fp) ? 1.f : 0.f;
        }
        break;
    }
    case Dtype::F64: {
        auto* q = reinterpret_cast<double*>(cpu.ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            q[i] = (static_cast<double>(gen.next_uniform_float()) < p) ? 1.0 : 0.0;
        }
        break;
    }
    default:
        ErrorBuilder("random_bernoulli").not_implemented("dtype not supported (F32/F64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

Storage random_randint_storage(const Shape& shape,
                               std::int64_t low,
                               std::int64_t high,
                               Dtype dt,
                               Device device,
                               Generator& gen) {
    if (high <= low)
        ErrorBuilder("random_randint").fail("high must be > low");
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
    case Dtype::I32:
        fill_randint<std::int32_t>(reinterpret_cast<std::int32_t*>(cpu.ptr.get()), n, low, high,
                                   gen);
        break;
    case Dtype::I64:
        fill_randint<std::int64_t>(reinterpret_cast<std::int64_t*>(cpu.ptr.get()), n, low, high,
                                   gen);
        break;
    default:
        ErrorBuilder("random_randint").not_implemented("dtype not supported (I32/I64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

}  // namespace lucid
