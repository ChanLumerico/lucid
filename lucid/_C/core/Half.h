// IEEE-754 binary16 conversion, shared by every backend-side path that has
// to touch half precision without a host ``_Float16``.
//
// Extracted from ``backend/cpu/CpuBackend.h`` when the CPU reduction
// kernels needed it too: reductions live in ``ops/`` and no op source
// includes the backend header, so the choice was to duplicate an IEEE
// conversion or to move it somewhere both can see.  Duplicating rounding
// code is how two copies quietly disagree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace lucid {
namespace backend {

namespace detail {

// bfloat16 <-> float32.
//
// A bfloat16 is the top 16 bits of an IEEE-754 binary32 float (same sign,
// same 8-bit exponent, mantissa truncated from 23 bits to 7).  Widening is a
// 16-bit left shift; narrowing is a branchless round-to-nearest-even shift
// that preserves quiet NaN payloads so auto-vectorization (NEON) can emit
// wide SIMD shifts and selects without branches.
inline float bfloat16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float out;
    __builtin_memcpy(&out, &widened, sizeof(out));
    return out;
}

inline float bfloat_bits_to_float(std::uint16_t bits) {
    return bfloat16_bits_to_float(bits);
}

inline std::uint16_t float_to_bfloat16_bits(float value) {
    std::uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t rounded =
        static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    return ((bits & 0x7fffffffu) > 0x7f800000u) ? static_cast<std::uint16_t>((bits >> 16) | 0x0040u)
                                                : rounded;
}

inline std::uint16_t float_to_bfloat_bits(float value) {
    return float_to_bfloat16_bits(value);
}

// IEEE-754 binary16 <-> float32.
//
// On Apple Silicon (__ARM_FP16_FORMAT_IEEE), scalar conversion uses the
// hardware ``fcvt`` instruction via ``__fp16``, which is branchless and
// bit-exact with IEEE-754 round-to-nearest-even (including subnormals,
// signed zero, infinities, and quieted NaNs).
inline float half_bits_to_float(std::uint16_t bits) {
#if defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 h;
    __builtin_memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
#else
    const std::uint32_t sign = static_cast<std::uint32_t>(bits >> 15) & 0x1u;
    const std::uint32_t exp = static_cast<std::uint32_t>(bits >> 10) & 0x1fu;
    const std::uint32_t mant = static_cast<std::uint32_t>(bits) & 0x3ffu;
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            std::uint32_t e = 1;
            std::uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            f = (sign << 31) | ((e + 112u) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
#endif
}

inline std::uint16_t float_to_half_bits(float value) {
#if defined(__ARM_FP16_FORMAT_IEEE)
    const __fp16 h = static_cast<__fp16>(value);
    std::uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits));
    return bits;
#else
    std::uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((f >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant = f & 0x7fffffu;

    if (((f >> 23) & 0xffu) == 0xffu)
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1f)
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10)
            return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        const std::uint32_t half = 1u << (shift - 1);
        const std::uint32_t rounded = (mant + half - 1u + ((mant >> shift) & 1u)) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t half = 0x1000u;
    const std::uint32_t rounded = mant + half - 1u + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) {
        ++exp;
        if (exp >= 0x1f)
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                      (rounded >> 13));
#endif
}

// Bulk contiguous conversions: NEON-vectorizable branchless loops over
// hardware ``__fp16`` (IEEE binary16) and bit-shift (bfloat16) helpers.
inline void
half_bits_to_float_bulk(const std::uint16_t* __restrict src, float* __restrict dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = half_bits_to_float(src[i]);
}

inline void
float_to_half_bits_bulk(const float* __restrict src, std::uint16_t* __restrict dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = float_to_half_bits(src[i]);
}

inline void bfloat16_bits_to_float_bulk(const std::uint16_t* __restrict src,
                                        float* __restrict dst,
                                        std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = bfloat16_bits_to_float(src[i]);
}

inline void float_to_bfloat16_bits_bulk(const float* __restrict src,
                                        std::uint16_t* __restrict dst,
                                        std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = float_to_bfloat16_bits(src[i]);
}

inline void half_or_bfloat16_to_float_bulk(const std::uint16_t* __restrict src,
                                           float* __restrict dst,
                                           std::size_t n,
                                           bool is_bfloat16) {
    if (is_bfloat16)
        bfloat16_bits_to_float_bulk(src, dst, n);
    else
        half_bits_to_float_bulk(src, dst, n);
}

inline void float_to_half_or_bfloat16_bulk(const float* __restrict src,
                                           std::uint16_t* __restrict dst,
                                           std::size_t n,
                                           bool is_bfloat16) {
    if (is_bfloat16)
        float_to_bfloat16_bits_bulk(src, dst, n);
    else
        float_to_half_bits_bulk(src, dst, n);
}

}  // namespace detail

}  // namespace backend
}  // namespace lucid
