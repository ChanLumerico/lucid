// lucid/_C/backend/cpu/IntPow.h
//
// Exact integer exponentiation by squaring for CPU binary ``pow``.
//
// ``std::pow(ap[i], bp[i])`` converts ``int64_t`` operands through ``double``,
// losing precision above $2^{53}$, and signed integer overflow is undefined
// behaviour in C++.  Accumulating in the matching unsigned type wraps modulo
// $2^w$, matching Metal's ``mlx::core::power`` integer kernel and the
// reference framework's integer power across ``int8``, ``int16``, ``int32``
// and ``int64``.  Negative integer exponents are rejected on the CPU stream
// with the reference framework's error message.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../../core/ErrorBuilder.h"

namespace lucid::backend::cpu {

// Exact ``base ** exp`` modulo $2^w$ for non-negative ``exp``.
template <typename T>
inline T ipow_scalar(T base, T exp) {
    if (exp < 0)
        ErrorBuilder("pow").fail("Integers to negative integer powers are not allowed.");
    using U = std::make_unsigned_t<T>;
    U ubase = static_cast<U>(base);
    U uexp = static_cast<U>(exp);
    U res = 1;
    while (uexp != 0) {
        if (uexp & 1)
            res *= ubase;
        uexp >>= 1;
        ubase *= ubase;
    }
    return static_cast<T>(res);
}

// Element-wise integer power ``op[i] = ap[i] ** bp[i]`` over ``n`` elements.
template <typename T>
inline void ipow_vec(const T* ap, const T* bp, T* op, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        op[i] = ipow_scalar(ap[i], bp[i]);
}

}  // namespace lucid::backend::cpu
