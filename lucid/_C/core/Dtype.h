// lucid/_C/core/Dtype.h
//
// Enumeration of element data types supported by the engine, together with
// constexpr helpers for element sizes, canonical names, and dtype predicates.
// Keeping these as constexpr functions lets the compiler evaluate dtype-based
// decisions (e.g. stride calculation) entirely at compile time when the dtype
// is a constant.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace lucid {

// Element type of a tensor's storage buffer.
//
// The underlying uint8_t representation must remain stable — it is baked into
// serialised checkpoints and the Python/C++ ABI boundary.  Append new entries
// at the end; do not reorder.
//
// C64 stores a pair of float32 values (real, imag) and therefore has the same
// item size as F64 but a fundamentally different numeric semantics.
enum class Dtype : std::uint8_t {
    Bool,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    C64,
};

// Returns the size in bytes of a single element of type dt.
// Throws std::logic_error for unknown enumerators.
constexpr std::size_t dtype_size(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return 1;
    case Dtype::I8:
        return 1;
    case Dtype::I16:
        return 2;
    case Dtype::I32:
        return 4;
    case Dtype::I64:
        return 8;
    case Dtype::F16:
        return 2;
    case Dtype::F32:
        return 4;
    case Dtype::F64:
        return 8;
    case Dtype::C64:
        return 8;
    }
    throw std::logic_error("dtype_size: unknown Dtype");
}

// Returns the canonical string name used in error messages and Python repr.
// Throws std::logic_error for unknown enumerators.
constexpr std::string_view dtype_name(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return "bool";
    case Dtype::I8:
        return "int8";
    case Dtype::I16:
        return "int16";
    case Dtype::I32:
        return "int32";
    case Dtype::I64:
        return "int64";
    case Dtype::F16:
        return "float16";
    case Dtype::F32:
        return "float32";
    case Dtype::F64:
        return "float64";
    case Dtype::C64:
        return "complex64";
    }
    throw std::logic_error("dtype_name: unknown Dtype");
}

// Returns true for F16, F32, and F64.  BF16 is not yet in the enum; if it is
// added, this predicate must be updated.
constexpr bool is_floating_point(Dtype dt) {
    return dt == Dtype::F16 || dt == Dtype::F32 || dt == Dtype::F64;
}

// Returns true for the four signed integer types (Bool is excluded because it
// has no arithmetic meaning in the usual integer sense).
constexpr bool is_integral(Dtype dt) {
    return dt == Dtype::I8 || dt == Dtype::I16 || dt == Dtype::I32 || dt == Dtype::I64;
}

}  // namespace lucid
