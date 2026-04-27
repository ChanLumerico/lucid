#pragma once

// =====================================================================
// Lucid C++ engine — Generator (Philox-4x32-10 counter-based RNG).
// =====================================================================
//
// Why counter-based: given the same (key, counter) you always get the same
// output. That makes parallel determinism trivial — each worker can
// independently advance from a known counter offset and recompose into a
// reproducible sequence. Same family PyTorch uses for CPU/CUDA RNGs.
//
// Threading:      One Generator per thread is safe. To share, hold mutex().
// Layer:          core/.

#include <cstdint>
#include <memory>
#include <mutex>

#include "../api.h"

namespace lucid {

// Counter-based RNG (Philox-4x32-10), the same family PyTorch uses for its
// CPU/CUDA generators. Counter-based means: given the same (key, counter),
// you always get the same output — perfect for parallel determinism. We
// expose a single 64-bit counter plus a 64-bit key (seed); each `next_uint32`
// advances the counter monotonically.
//
// Threading: a single Generator is NOT thread-safe by default; use one per
// thread, or hold the embedded mutex via `lock()`. The Phase 3 op layer
// expects each op call to take its own Generator& and the caller is
// responsible for not sharing across threads without locking.
class LUCID_API Generator {
public:
    explicit Generator(std::uint64_t seed = 0);

    void set_seed(std::uint64_t seed);
    std::uint64_t seed() const { return seed_; }

    // Next 4 random uint32s in `out`. Advances the internal counter by 1.
    // (Philox-4x32 produces 4 outputs per round; we expose all four.)
    void next_uint32x4(std::uint32_t out[4]);

    // Convenience: single uniform float in [0, 1).
    float next_uniform_float();

    // For thread-safe sharing (rare path).
    std::mutex& mutex() { return mu_; }

    // Snapshot for reproducibility tests / checkpointing.
    std::uint64_t counter() const { return counter_; }
    void set_counter(std::uint64_t c) { counter_ = c; }

private:
    std::uint64_t seed_;
    std::uint64_t counter_;
    std::mutex mu_;
};

// Process-default generator. Provided so `lucid.random.seed(n)` has somewhere
// to write to. Phase 3 op forwards take an explicit Generator& parameter that
// defaults to this one.
LUCID_API Generator& default_generator();

}  // namespace lucid
