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

// Fill with U(0, 1) uniform samples.
TensorImplPtr rand_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"rand", device, dt, shape};
    auto s = random_uniform_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

// Fill with U(low, high) uniform samples; validate that high > low first.
TensorImplPtr
uniform_op(const Shape& shape, double low, double high, Dtype dt, Device device, Generator* gen) {
    if (high <= low)
        ErrorBuilder("uniform").fail("high must be > low");
    OpScopeFull scope{"uniform", device, dt, shape};
    auto s = random_uniform_storage(shape, low, high, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

// Fill with N(0, 1) standard-normal samples.
TensorImplPtr randn_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"randn", device, dt, shape};
    auto s = random_normal_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

// Fill with N(mean, std^2) samples; validate non-negative std.
TensorImplPtr
normal_op(const Shape& shape, double mean, double std, Dtype dt, Device device, Generator* gen) {
    if (std < 0.0)
        ErrorBuilder("normal").fail("std must be >= 0");
    OpScopeFull scope{"normal", device, dt, shape};
    auto s = random_normal_storage(shape, mean, std, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

// Fill with integer samples drawn uniformly from [low, high).
TensorImplPtr randint_op(const Shape& shape,
                         std::int64_t low,
                         std::int64_t high,
                         Dtype dt,
                         Device device,
                         Generator* gen) {
    OpScopeFull scope{"randint", device, dt, shape};
    auto s = random_randint_storage(shape, low, high, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

// Fill with independent Bernoulli(p) samples.
TensorImplPtr bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"bernoulli", device, dt, shape};
    auto s = random_bernoulli_storage(shape, p, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

}  // namespace lucid
