// lucid/_C/core/Generator.h
//
// Counter-based pseudo-random number generator using the Philox-4x32-10
// algorithm.  Philox is a block cipher used as a PRNG: given a (counter,
// seed) pair it deterministically produces 4 independent 32-bit outputs in a
// single call, with no state dependencies between successive calls beyond the
// counter increment.  This makes it well-suited for parallel sampling — two
// threads can independently advance the counter without lock contention.
//
// Each Generator instance owns an independent (seed, counter) pair and a
// mutex for thread-safe access from Python.  The global default generator
// (default_generator()) is a process-wide singleton with seed = 0.
//
// Thread safety: next_uint32x4() and next_uniform_float() are NOT thread-safe
// without external locking.  Call mutex().lock() / unlock() or hold a
// std::lock_guard<std::mutex> around sampling calls when sharing a Generator
// across threads.  The mutex() accessor is exposed so the Python binding can
// acquire it during batch generation.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include "../api.h"

namespace lucid {

// Philox-4x32-10 counter-based PRNG.
//
// Invariants:
//   - counter_ is monotonically increasing; it resets to 0 on set_seed().
//   - seed_ and counter_ together uniquely identify each call's output.
//   - mu_ guards counter_ and seed_ for multi-threaded sampling.
class LUCID_API Generator {
public:
    explicit Generator(std::uint64_t seed = 0);

    // Sets the seed and resets the counter to 0, restarting the sequence.
    void set_seed(std::uint64_t seed);
    std::uint64_t seed() const { return seed_; }

    // Generates 4 independent 32-bit uniform random values using
    // Philox-4x32-10 with the current (counter_, seed_) state, then
    // increments counter_ by 1.  out must point to an array of at least 4
    // uint32_t values.
    void next_uint32x4(std::uint32_t out[4]);

    // Generates a single float in [0, 1) using the upper 24 bits of one
    // Philox output word (discarding the lower 8 bits for uniformity).
    float next_uniform_float();

    // Provides access to the internal mutex so callers can hold it across
    // sequences of sampling calls without repeated lock/unlock overhead.
    std::mutex& mutex() { return mu_; }

    std::uint64_t counter() const { return counter_; }
    // Direct counter override — useful for checkpointing/restoring RNG state.
    void set_counter(std::uint64_t c) { counter_ = c; }

private:
    std::uint64_t seed_;
    std::uint64_t counter_;
    std::mutex mu_;
};

// Returns the process-wide default Generator (seed = 0, lazily constructed).
// All random ops that do not receive an explicit Generator argument use this.
LUCID_API Generator& default_generator();

}  // namespace lucid
