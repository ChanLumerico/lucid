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

inline Generator& resolve_gen(Generator* gen) {
    return gen ? *gen : default_generator();
}

inline TensorImplPtr finalize(Storage&& storage, Shape shape, Dtype dt, Device device) {
    return std::make_shared<TensorImpl>(std::move(storage), std::move(shape), dt, device, false);
}

}  // namespace

TensorImplPtr rand_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"rand", device, dt, shape};
    auto s = random_uniform_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

TensorImplPtr
uniform_op(const Shape& shape, double low, double high, Dtype dt, Device device, Generator* gen) {
    if (high <= low)
        ErrorBuilder("uniform").fail("high must be > low");
    OpScopeFull scope{"uniform", device, dt, shape};
    auto s = random_uniform_storage(shape, low, high, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

TensorImplPtr randn_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"randn", device, dt, shape};
    auto s = random_normal_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

TensorImplPtr
normal_op(const Shape& shape, double mean, double std, Dtype dt, Device device, Generator* gen) {
    if (std < 0.0)
        ErrorBuilder("normal").fail("std must be >= 0");
    OpScopeFull scope{"normal", device, dt, shape};
    auto s = random_normal_storage(shape, mean, std, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

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

TensorImplPtr bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"bernoulli", device, dt, shape};
    auto s = random_bernoulli_storage(shape, p, dt, device, resolve_gen(gen));
    return finalize(std::move(s), shape, dt, device);
}

}  // namespace lucid
