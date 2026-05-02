#pragma once

#include <cstddef>
#include <utility>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"

namespace lucid {

struct TensorMeta {
    Shape shape;
    Stride stride;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    TensorMeta() = default;

    TensorMeta(Shape shape_in, Dtype dtype_in, Device device_in)
        : shape(std::move(shape_in)),
          stride(contiguous_stride(shape, dtype_size(dtype_in))),
          dtype(dtype_in),
          device(device_in) {}

    TensorMeta(Shape shape_in, Stride stride_in, Dtype dtype_in, Device device_in)
        : shape(std::move(shape_in)),
          stride(std::move(stride_in)),
          dtype(dtype_in),
          device(device_in) {}

    std::size_t numel() const noexcept { return shape_numel(shape); }

    std::size_t nbytes() const noexcept { return numel() * dtype_size(dtype); }

    bool is_contiguous() const noexcept {
        return stride == contiguous_stride(shape, dtype_size(dtype));
    }
};

}  // namespace lucid
