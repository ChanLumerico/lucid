// lucid/_C/backend/DivMod.h
//
// The rule for the floor-division family — ``//``, ``%`` (remainder) and
// ``fmod`` — shared by both streams.
//
// All three used to go through float.  Integer ``//`` divided in double on
// the CPU and float32 on Metal, so ``(2**53 + 1) // 1`` came back 2**53 and
// Metal's ``100000001 // 3`` came back 33333334.  ``fmod`` and ``remainder``
// were ``a - trunc(a / b) * b`` and ``a - floor(a / b) * b``, which round
// the quotient: ``2**62 + 3 % 7`` was 259.  Now every element is computed
// exactly, by the reference's formulas, which are written here once.  The
// CPU kernels (cpu/DivMod.h) call these per element; the Metal kernels
// (gpu/DivMod.h) spell the same steps in MLX ops.
//
// Integers divide in their own type.  The quotient truncates in C++, so the
// floored quotient subtracts one when the remainder is non-zero and its
// sign differs from the divisor's, and the floored remainder adds the
// divisor back in the same case.  ``min / -1`` overflows in C++ (undefined
// behaviour), so a divisor of -1 is answered as negation, which wraps:
// ``int8 -128 // -1`` is -128, as in the reference.
//
// An integer zero divisor has no answer.  The CPU refuses it before any
// kernel runs (ops/bfunc/Floordiv.cpp), as the reference's CPU does.  Metal
// does not look: reading ``b`` back would stall every integer ``//`` on the
// host, and the reference's accelerator does not check either.  There a
// zero divisor gives ``a // 0 == -1`` and ``a % 0 == fmod(a, 0) == a``,
// with no error; the pair keeps ``(a // b) * b + a % b == a``.  The scalar
// rule below answers the same values without dividing by zero.
//
// Floats start from the exact ``fmod``.  ``remainder`` adds the divisor
// back when the signs disagree, so ``-5 % inf`` is inf and ``5 % inf`` is 5.
// The floored quotient is ``(a - fmod) / b``, which is within rounding of a
// whole number, corrected by the same sign rule and then snapped to the
// nearest whole number from below; ``-5 // inf`` is -1, ``1.0 // 0.1`` is
// 9, and a zero quotient keeps the sign of ``a / b``.  Division by zero is
// the IEEE ``a / b``.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace lucid::backend {

enum class DivMod : std::uint8_t { FloorDiv, Remainder, Fmod };

template <DivMod M, std::signed_integral T>
constexpr T divmod_integer(T a, T b) {
    using U = std::make_unsigned_t<T>;
    if (b == 0)
        return M == DivMod::FloorDiv ? T{-1} : a;
    if (b == -1) {
        if constexpr (M == DivMod::FloorDiv)
            return static_cast<T>(static_cast<U>(U{0} - static_cast<U>(a)));
        return T{0};
    }
    const T q = static_cast<T>(a / b);
    const T r = static_cast<T>(a % b);
    if constexpr (M == DivMod::Fmod)
        return r;
    const bool adjust = r != 0 && ((r < 0) != (b < 0));
    if constexpr (M == DivMod::FloorDiv)
        return adjust ? static_cast<T>(q - 1) : q;
    return adjust ? static_cast<T>(r + b) : r;
}

// ``mod`` is ``fmod(a, b)``, which the caller computes (vectorised on the
// CPU stream).
template <DivMod M, std::floating_point T>
T divmod_floating(T a, T b, T mod) {
    if constexpr (M == DivMod::Fmod)
        return mod;
    const bool adjust = mod != 0 && ((b < 0) != (mod < 0));
    if constexpr (M == DivMod::Remainder)
        return adjust ? mod + b : mod;
    if (b == 0)
        return a / b;
    T div = (a - mod) / b;
    if (adjust)
        div -= T(1);
    if (div == 0)
        return std::copysign(T(0), a / b);
    T whole = std::floor(div);
    if (div - whole > T(0.5))
        whole += T(1);
    return whole;
}

}  // namespace lucid::backend
