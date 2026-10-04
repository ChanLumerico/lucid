// lucid/_C/backend/cpu/NonFinite.h
//
// isnan / isinf / isfinite and nan_to_num for every CPU float format,
// answered from the bits.
//
// The CPU kernels used to test only float32 and float64.  Every other dtype
// fell through to the integer answer — nothing NaN, nothing infinite,
// everything finite — so a float16 or bfloat16 tensor full of infinities
// passed as clean, nan_to_num handed it back unchanged, and GradScaler,
// which asks isfinite of each gradient, never saw a half-precision overflow
// on the CPU (CHA-34).  Metal answered correctly all along.
//
// No conversion is needed to answer.  With the sign bit cleared, an IEEE
// value is NaN when its bits exceed the all-ones exponent, infinite when they
// equal it, and finite when they are below it.  That holds for float16
// (exponent 0x7C00), bfloat16 (0x7F80), float32 and float64 alike, so one
// template over the unsigned word of the right width covers all four.  It
// also covers the parts of a complex number.  The loops are integer AND
// and compare, which vectorise.  Converting a half to float first would
// cost a widening pass and give the same answer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "Parallel.h"

namespace lucid::backend::cpu {

// What a probe asks of each element.
enum class NonFiniteProbe { IsNan, IsInf, IsFinite };

// The bit layout of one IEEE binary format.
//
// Template Parameters
// -------------------
// Bits : unsigned integer type
//     The word one value occupies — 16, 32 or 64 bits.
// kExponent : Bits
//     The exponent field with every bit set: the bits of +inf.
template <typename Bits, Bits kExponent>
struct FloatFormat {
    using Word = Bits;
    static constexpr Word exponent = kExponent;
    // Every bit except the sign: masking with it gives |x|'s bits.
    static constexpr Word magnitude = std::numeric_limits<Word>::max() >> 1;
};

using HalfFormat = FloatFormat<std::uint16_t, 0x7C00u>;
using BrainFormat = FloatFormat<std::uint16_t, 0x7F80u>;
using SingleFormat = FloatFormat<std::uint32_t, 0x7F800000u>;
using DoubleFormat = FloatFormat<std::uint64_t, 0x7FF0000000000000ull>;

// Answer probe ``P`` for one value, given its magnitude bits.
template <NonFiniteProbe P, typename Format>
constexpr bool probe_magnitude(typename Format::Word m) {
    if constexpr (P == NonFiniteProbe::IsNan)
        return m > Format::exponent;
    else if constexpr (P == NonFiniteProbe::IsInf)
        return m == Format::exponent;
    else
        return m < Format::exponent;
}

// ``out[i] = P(in[i])`` as a 0 / 1 byte, over ``n`` elements.
//
// Template Parameters
// -------------------
// P : NonFiniteProbe
//     The question.  A template argument, not a runtime switch, so the loop
//     body holds a single comparison and vectorises.
// Format : FloatFormat
//     The layout of each value.
// kComplex : bool
//     Each element is two values, real then imaginary.  A complex number is
//     NaN, or infinite, when either part is, and finite only when both are
//     (the reference framework's rule).
//
// Parameters
// ----------
// in : const void*
//     ``n`` contiguous elements.
// out : std::uint8_t*
//     ``n`` bytes.
// n : std::size_t
//     Element count.
// grain : std::size_t
//     Smallest chunk handed to another core (cpu::parallel_for).
template <NonFiniteProbe P, typename Format, bool kComplex = false>
void probe_nonfinite(const void* in, std::uint8_t* out, std::size_t n, std::size_t grain) {
    using Word = typename Format::Word;
    parallel_for(n, grain, [&](std::size_t lo, std::size_t hi) {
        // Locals, not the closure's references: a byte store may alias
        // anything, which would keep the loop scalar.
        const Word* __restrict x = static_cast<const Word*>(in);
        std::uint8_t* __restrict y = out;
        for (std::size_t i = lo; i < hi; ++i) {
            if constexpr (kComplex) {
                const bool re = probe_magnitude<P, Format>(x[2 * i] & Format::magnitude);
                const bool im = probe_magnitude<P, Format>(x[2 * i + 1] & Format::magnitude);
                // Bitwise, not short-circuit: both parts are read anyway,
                // and a branch would keep the loop scalar.
                const bool both = P == NonFiniteProbe::IsFinite;
                y[i] = static_cast<std::uint8_t>(both ? (re & im) : (re | im));
            } else {
                y[i] = probe_magnitude<P, Format>(x[i] & Format::magnitude) ? 1u : 0u;
            }
        }
    });
}

// ``out[i] = in[i]`` with NaN, +inf and -inf replaced, over ``n`` words.
//
// The replacements arrive already in the format's own bits.  The caller
// rounds each double the way a cast to the dtype would (through float for
// the 16-bit formats, as the reference framework's half types do), so a
// value the format cannot hold becomes its infinity, as the reference
// framework's cast makes it.  Every other word is copied bit for bit:
// -0, subnormals and NaN payloads are untouched where they are not
// replaced.
//
// A complex tensor is passed as its ``2 n`` interleaved parts.  Each part
// is replaced on its own, which is the reference framework's rule for
// complex input.
//
// Parameters
// ----------
// in : const void*
//     ``n`` contiguous words.
// out : void*
//     ``n`` words.
// n : std::size_t
//     Word count.
// nan_bits, posinf_bits, neginf_bits : Format::Word
//     The replacements.
// grain : std::size_t
//     Smallest chunk handed to another core (cpu::parallel_for).
template <typename Format>
void replace_nonfinite(const void* in,
                       void* out,
                       std::size_t n,
                       typename Format::Word nan_bits,
                       typename Format::Word posinf_bits,
                       typename Format::Word neginf_bits,
                       std::size_t grain) {
    using Word = typename Format::Word;
    parallel_for(n, grain, [&](std::size_t lo, std::size_t hi) {
        const Word* __restrict x = static_cast<const Word*>(in);
        Word* __restrict y = static_cast<Word*>(out);
        const Word nan_r = nan_bits;
        const Word pos_r = posinf_bits;
        const Word neg_r = neginf_bits;
        for (std::size_t i = lo; i < hi; ++i) {
            const Word v = x[i];
            const Word m = v & Format::magnitude;
            // Clearing the sign changed nothing: a positive value.
            const Word inf_r = v == m ? pos_r : neg_r;
            const Word kept = m == Format::exponent ? inf_r : v;
            y[i] = m > Format::exponent ? nan_r : kept;
        }
    });
}

}  // namespace lucid::backend::cpu
