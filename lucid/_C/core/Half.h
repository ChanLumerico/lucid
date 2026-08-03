// IEEE-754 binary16 conversion, shared by every backend-side path that has
// to touch half precision without a host ``_Float16``.
//
// Extracted from ``backend/cpu/CpuBackend.h`` when the CPU reduction
// kernels needed it too: reductions live in ``ops/`` and no op source
// includes the backend header, so the choice was to duplicate an IEEE
// conversion or to move it somewhere both can see.  Duplicating rounding
// code is how two copies quietly disagree.

#pragma once

#include <cstdint>

namespace lucid {
namespace backend {

namespace detail {

// IEEE-754 binary16 <-> float, done by hand so this header does not depend on
// ``__fp16`` / ``_Float16`` availability.  Used only by ``CpuBackend::astype``:
// F16 has no ``static_cast``-able host scalar, so it cannot ride the generic
// cast table.
// bfloat16 <-> float.
//
// Far simpler than binary16: a bfloat16 *is* the top 16 bits of a float32,
// same sign, same 8-bit exponent, mantissa truncated from 23 bits to 7.  So
// widening is a shift and narrowing is a rounded shift — no exponent
// rebasing, no subnormal handling, and the whole float32 range survives,
// which is the reason to use the format at all.
inline float bfloat_bits_to_float(std::uint16_t bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float out;
    __builtin_memcpy(&out, &widened, sizeof(out));
    return out;
}

inline std::uint16_t float_to_bfloat_bits(float value) {
    std::uint32_t raw;
    __builtin_memcpy(&raw, &value, sizeof(raw));
    // NaN must stay NaN: rounding a quiet NaN's payload can carry into the
    // exponent and turn it into an infinity.
    if ((raw & 0x7f800000u) == 0x7f800000u && (raw & 0x007fffffu) != 0u)
        return static_cast<std::uint16_t>((raw >> 16) | 0x0040u);
    // Round to nearest, ties to even, on the bit that is about to be lost.
    const std::uint32_t lsb = (raw >> 16) & 1u;
    const std::uint32_t bias = 0x7fffu + lsb;
    return static_cast<std::uint16_t>((raw + bias) >> 16);
}

inline float half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits >> 15) & 0x1u;
    const std::uint32_t exp = static_cast<std::uint32_t>(bits >> 10) & 0x1fu;
    const std::uint32_t mant = static_cast<std::uint32_t>(bits) & 0x3ffu;
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;  // +/- zero
        } else {
            // Subnormal half: renormalise into a float's exponent range.
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
        f = (sign << 31) | (0xffu << 23) | (mant << 13);  // inf / NaN
    } else {
        f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Round-to-nearest-even, with overflow to inf and graceful subnormal handling.
inline std::uint16_t float_to_half_bits(float value) {
    std::uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((f >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant = f & 0x7fffffu;

    if (((f >> 23) & 0xffu) == 0xffu)  // inf / NaN
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1f)  // overflow -> inf
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10)  // underflow -> signed zero
            return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;  // restore the implicit bit, then shift into place
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        const std::uint32_t half = 1u << (shift - 1);
        const std::uint32_t rounded = (mant + half - 1u + ((mant >> shift) & 1u)) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t half = 0x1000u;
    const std::uint32_t rounded = mant + half - 1u + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) {  // rounding carried into the exponent
        ++exp;
        if (exp >= 0x1f)
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                      (rounded >> 13));
}

}  // namespace detail

}  // namespace backend
}  // namespace lucid
