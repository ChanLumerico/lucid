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
#include "../compile/RngFeeds.h"
#include "../compile/Tracer.h"
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

// Tracer hook.  A draw from the default generator becomes a feed that
// ``run_executable`` draws again on every call — see
// :file:`compile/RngFeeds.h` for why a seeded op in the graph is not
// enough.  A draw from a caller's own generator stays an op, tagged so
// the compile path declines it: the executable could not advance that
// generator, and an eager fallback is the only way to keep its stream.
namespace {
inline void register_rng_output(const char* name,
                                const TensorImplPtr& out,
                                Generator* gen,
                                compile::RngRecipe recipe) {
    auto* trc = ::lucid::compile::current_tracer();
    if (trc == nullptr)
        return;
    const bool default_gen =
        gen == nullptr || gen == &default_generator() || gen == &current_default_generator();
    if (default_gen && !trc->redraw_rng()) {
        // Kept as an op: the consumer of this trace wants the draw itself.
        trc->on_op_io({}, out);
        return;
    }
    if (default_gen) {
        recipe.shape = out->shape();
        recipe.dtype = out->dtype();
        recipe.device = out->device();
        trc->on_rng_feed(name, out);
        compile::note_rng_feed(out, std::move(recipe));
        return;
    }
    trc->on_op_attr("own_generator", true);
    trc->on_op_io({}, out);
}

compile::RngRecipe recipe_of(compile::RngRecipe::Kind kind, double a, double b) {
    compile::RngRecipe r;
    r.kind = kind;
    r.a = a;
    r.b = b;
    return r;
}
}  // namespace

// Fill with U(0, 1) uniform samples.
TensorImplPtr rand_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"rand", device, dt, shape};
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    auto s = random_uniform_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output("rand", out, gen, recipe_of(compile::RngRecipe::Kind::Uniform, 0.0, 1.0));
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
    register_rng_output("uniform", out, gen,
                        recipe_of(compile::RngRecipe::Kind::Uniform, low, high));
    return out;
}

// Fill with N(0, 1) standard-normal samples.
TensorImplPtr randn_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"randn", device, dt, shape};
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    auto s = random_normal_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output("randn", out, gen, recipe_of(compile::RngRecipe::Kind::Normal, 0.0, 1.0));
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
    register_rng_output("normal", out, gen, recipe_of(compile::RngRecipe::Kind::Normal, mean, std));
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
    auto r = recipe_of(compile::RngRecipe::Kind::Randint, 0.0, 0.0);
    r.lo = low;
    r.hi = high;
    register_rng_output("randint", out, gen, std::move(r));
    return out;
}

// Fill with independent Bernoulli(p) samples.
TensorImplPtr bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"bernoulli", device, dt, shape};
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("p", p);
    auto s = random_bernoulli_storage(shape, p, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output("bernoulli", out, gen,
                        recipe_of(compile::RngRecipe::Kind::Bernoulli, p, 0.0));
    return out;
}

}  // namespace lucid
