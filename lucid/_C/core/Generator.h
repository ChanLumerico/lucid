#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include "../api.h"

namespace lucid {

class LUCID_API Generator {
public:
    explicit Generator(std::uint64_t seed = 0);

    void set_seed(std::uint64_t seed);
    std::uint64_t seed() const { return seed_; }

    void next_uint32x4(std::uint32_t out[4]);

    float next_uniform_float();

    std::mutex& mutex() { return mu_; }

    std::uint64_t counter() const { return counter_; }
    void set_counter(std::uint64_t c) { counter_ = c; }

private:
    std::uint64_t seed_;
    std::uint64_t counter_;
    std::mutex mu_;
};

LUCID_API Generator& default_generator();

}  // namespace lucid
