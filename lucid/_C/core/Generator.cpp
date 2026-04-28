#include "Generator.h"

#include <cstring>

namespace lucid {

namespace {

// Philox-4x32-10 — D. E. Shaw / Salmon, Moraes, Dror, Shaw 2011.
// "Parallel Random Numbers: As Easy as 1, 2, 3" — the same construction
// PyTorch uses. We split the 64-bit key into two 32-bit lanes and the
// 64-bit counter into four 32-bit input words.
constexpr std::uint32_t PHILOX_M0 = 0xD2511F53u;
constexpr std::uint32_t PHILOX_M1 = 0xCD9E8D57u;
constexpr std::uint32_t PHILOX_W0 = 0x9E3779B9u;  // golden ratio
constexpr std::uint32_t PHILOX_W1 = 0xBB67AE85u;

inline void mulhilo32(std::uint32_t a, std::uint32_t b, std::uint32_t& hi, std::uint32_t& lo) {
    const std::uint64_t product = static_cast<std::uint64_t>(a) * b;
    hi = static_cast<std::uint32_t>(product >> 32);
    lo = static_cast<std::uint32_t>(product);
}

inline void philox_round(std::uint32_t ctr[4], std::uint32_t key[2]) {
    std::uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(PHILOX_M0, ctr[0], hi0, lo0);
    mulhilo32(PHILOX_M1, ctr[2], hi1, lo1);
    const std::uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    const std::uint32_t new1 = lo1;
    const std::uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    const std::uint32_t new3 = lo0;
    ctr[0] = new0;
    ctr[1] = new1;
    ctr[2] = new2;
    ctr[3] = new3;
}

inline void philox_bumpkey(std::uint32_t key[2]) {
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

void philox_4x32_10(std::uint64_t counter, std::uint64_t seed, std::uint32_t out[4]) {
    std::uint32_t ctr[4];
    ctr[0] = static_cast<std::uint32_t>(counter);
    ctr[1] = static_cast<std::uint32_t>(counter >> 32);
    ctr[2] = 0;
    ctr[3] = 0;

    std::uint32_t key[2];
    key[0] = static_cast<std::uint32_t>(seed);
    key[1] = static_cast<std::uint32_t>(seed >> 32);

    for (int i = 0; i < 10; ++i) {
        philox_round(ctr, key);
        if (i != 9)
            philox_bumpkey(key);
    }

    out[0] = ctr[0];
    out[1] = ctr[1];
    out[2] = ctr[2];
    out[3] = ctr[3];
}

}  // namespace

Generator::Generator(std::uint64_t seed) : seed_(seed), counter_(0) {}

void Generator::set_seed(std::uint64_t seed) {
    seed_ = seed;
    counter_ = 0;
}

void Generator::next_uint32x4(std::uint32_t out[4]) {
    philox_4x32_10(counter_, seed_, out);
    counter_ += 1;
}

float Generator::next_uniform_float() {
    std::uint32_t buf[4];
    next_uint32x4(buf);
    // Use top 24 bits of buf[0] for fp32 mantissa precision.
    const std::uint32_t mantissa = buf[0] >> 8;  // 24 bits
    return static_cast<float>(mantissa) * (1.0f / static_cast<float>(1u << 24));
}

Generator& default_generator() {
    static Generator g{0};
    return g;
}

}  // namespace lucid
