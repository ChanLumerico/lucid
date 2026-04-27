#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace lucid {

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

constexpr std::size_t dtype_size(Dtype dt) {
    switch (dt) {
        case Dtype::Bool: return 1;
        case Dtype::I8:   return 1;
        case Dtype::I16:  return 2;
        case Dtype::I32:  return 4;
        case Dtype::I64:  return 8;
        case Dtype::F16:  return 2;
        case Dtype::F32:  return 4;
        case Dtype::F64:  return 8;
        case Dtype::C64:  return 8;
    }
    throw std::logic_error("dtype_size: unknown Dtype");
}

constexpr std::string_view dtype_name(Dtype dt) {
    switch (dt) {
        case Dtype::Bool: return "bool";
        case Dtype::I8:   return "int8";
        case Dtype::I16:  return "int16";
        case Dtype::I32:  return "int32";
        case Dtype::I64:  return "int64";
        case Dtype::F16:  return "float16";
        case Dtype::F32:  return "float32";
        case Dtype::F64:  return "float64";
        case Dtype::C64:  return "complex64";
    }
    throw std::logic_error("dtype_name: unknown Dtype");
}

constexpr bool is_floating_point(Dtype dt) {
    return dt == Dtype::F16 || dt == Dtype::F32 || dt == Dtype::F64;
}

constexpr bool is_integral(Dtype dt) {
    return dt == Dtype::I8 || dt == Dtype::I16 ||
           dt == Dtype::I32 || dt == Dtype::I64;
}

}  // namespace lucid
