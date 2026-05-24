// lucid/_C/random/Random.cpp
//
// Thin wrappers that route each random operation through the Storage-layer
// random helpers and wrap the result in a non-differentiable TensorImpl.
// All distribution-specific logic lives in core/Storage (CPU) or is
// delegated to mlx::core::random (GPU); this file is concerned only with
// profiling scope management, argument validation, and TensorImpl construction.

#include "Random.h"

#include <utility>

#include "../autograd/Helpers.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/Generator.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"

namespace lucid {

namespace {

// Fall back to the process-global default Generator when none is supplied.
inline Generator& resolve_gen(Generator* gen) {
    return gen ? *gen : default_generator();
}

// Construct a leaf TensorImpl from a freshly allocated Storage.
// requires_grad is always false for random tensors.
inline TensorImplPtr finalize(Storage&& storage, Shape shape, Dtype dt, Device device) {
    return std::make_shared<TensorImpl>(std::move(storage), std::move(shape), dt, device, false);
}

}  // namespace

// Tracer hook helper.  Without it the RNG output impl never enters
// the trace's ``impl_to_id_`` map, so downstream consumers see a
// fresh feed id instead of the trace id minted by ``OpScopeFull`` —
// breaks the compile-path connection.  Same pattern as factories
// (zeros / ones / full).
namespace {
inline void register_rng_output(const TensorImplPtr& out) {
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({}, out);
    }
}
}  // namespace

// Fill with U(0, 1) uniform samples.
TensorImplPtr rand_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"rand", device, dt, shape};
    // The compile-path emitter reads the seed from the active
    // generator's state so the compiled executable is reproducible
    // call-to-call (mirroring the eager generator's seed value at
    // trace time — stateless re-emit, no Philox advancement).
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    auto s = random_uniform_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output(out);
    return out;
}

// Fill with U(low, high) uniform samples; validate that high > low first.
TensorImplPtr
uniform_op(const Shape& shape, double low, double high, Dtype dt, Device device, Generator* gen) {
    if (high <= low)
        ErrorBuilder("uniform").fail("high must be > low");
    OpScopeFull scope{"uniform", device, dt, shape};
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("low", low);
    scope.set_attr("high", high);
    auto s = random_uniform_storage(shape, low, high, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output(out);
    return out;
}

// Fill with N(0, 1) standard-normal samples.
TensorImplPtr randn_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"randn", device, dt, shape};
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    auto s = random_normal_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output(out);
    return out;
}

// Fill with N(mean, std^2) samples; validate non-negative std.
TensorImplPtr
normal_op(const Shape& shape, double mean, double std, Dtype dt, Device device, Generator* gen) {
    if (std < 0.0)
        ErrorBuilder("normal").fail("std must be >= 0");
    OpScopeFull scope{"normal", device, dt, shape};
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("mean", mean);
    scope.set_attr("std", std);
    auto s = random_normal_storage(shape, mean, std, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output(out);
    return out;
}

// Fill with integer samples drawn uniformly from [low, high).
TensorImplPtr randint_op(const Shape& shape,
                         std::int64_t low,
                         std::int64_t high,
                         Dtype dt,
                         Device device,
                         Generator* gen) {
    OpScopeFull scope{"randint", device, dt, shape};
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("low", low);
    scope.set_attr("high", high);
    auto s = random_randint_storage(shape, low, high, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output(out);
    return out;
}

// Fill with independent Bernoulli(p) samples.
TensorImplPtr bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"bernoulli", device, dt, shape};
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("p", p);
    auto s = random_bernoulli_storage(shape, p, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output(out);
    return out;
}

}  // namespace lucid
