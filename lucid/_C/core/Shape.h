#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lucid {

using Shape = std::vector<std::int64_t>;
using Stride = std::vector<std::int64_t>;

inline std::size_t shape_numel(const Shape& shape) {
    if (shape.empty())
        return 1;
    std::size_t n = 1;
    for (auto d : shape) {
        if (d < 0)
            return 0;
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

inline Stride contiguous_stride(const Shape& shape, std::size_t elem_size) {
    Stride s(shape.size());
    if (shape.empty())
        return s;
    std::int64_t acc = static_cast<std::int64_t>(elem_size);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
        s[static_cast<std::size_t>(i)] = acc;
        acc *= shape[static_cast<std::size_t>(i)];
    }
    return s;
}

}  // namespace lucid
