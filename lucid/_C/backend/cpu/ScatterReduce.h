// lucid/_C/backend/cpu/ScatterReduce.h
//
// The element type and the combine step of the CPU scatter-reduce kernels
// (scatter_amax / scatter_amin / scatter_prod).
//
// The kernel loop dispatched on ``F32`` and ``F64`` only, so every integer,
// bool and complex input raised ``NotImplementedError`` on the CPU while
// Metal reduced it.  The reductions are comparisons and a product, which
// every dtype's native type supports exactly — so each dtype runs at its
// own width here, except the 16-bit floats, which the backend widens to
// float32 at the door as it does for the rest of the half family.  Complex
// values have no order: amax / amin refuse them, as the reference does,
// and prod multiplies them as complex numbers.

#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>

#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"

namespace lucid::backend::cpu {

enum class ScatterReduce { Amax, Amin, Prod };

template <typename T>
inline constexpr bool is_std_complex_v = false;
template <typename T>
inline constexpr bool is_std_complex_v<std::complex<T>> = true;

// ``d = d (op) s`` for one element.  bool rides ``uint8_t``: on 0 / 1 the
// max is or, the min and the product are and.
template <typename T>
inline void scatter_combine(ScatterReduce r, T& d, const T& s) {
    if constexpr (is_std_complex_v<T>) {
        d *= s;  // the dispatch below admits complex for Prod only
    } else {
        switch (r) {
        case ScatterReduce::Amax:
            if (s > d)
                d = s;
            break;
        case ScatterReduce::Amin:
            if (s < d)
                d = s;
            break;
        case ScatterReduce::Prod:
            d = static_cast<T>(d * s);
            break;
        }
    }
}

// Call ``fn(T{})`` with the native element type of ``dt``.  The 16-bit
// floats are not accepted: the caller widens them to float32 first.
template <typename Fn>
inline void visit_scatter_reduce_type(Dtype dt, ScatterReduce r, const char* op, Fn&& fn) {
    if (is_complex(dt) && r != ScatterReduce::Prod)
        ErrorBuilder(op).not_implemented("amax / amin have no order on complex values");
    switch (dt) {
    case Dtype::F32:
        fn(float{});
        break;
    case Dtype::F64:
        fn(double{});
        break;
    case Dtype::I64:
        fn(std::int64_t{});
        break;
    case Dtype::I32:
        fn(std::int32_t{});
        break;
    case Dtype::I16:
        fn(std::int16_t{});
        break;
    case Dtype::I8:
        fn(std::int8_t{});
        break;
    case Dtype::Bool:
        fn(std::uint8_t{});
        break;
    case Dtype::C64:
        fn(std::complex<float>{});
        break;
    case Dtype::C128:
        fn(std::complex<double>{});
        break;
    default:
        ErrorBuilder(op).not_implemented("no scatter-reduce kernel for dtype " +
                                         std::string(dtype_name(dt)));
    }
}

}  // namespace lucid::backend::cpu
