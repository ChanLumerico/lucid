// lucid/_C/tensor/TensorMeta.h
//
// Lightweight value type that bundles a tensor's shape, stride, dtype, and
// device without owning any Storage. TensorMeta is used in shape-inference
// code paths (e.g., broadcast rules, view/reshape planning) where the data
// itself is not needed. Decoupling metadata from storage allows shape
// computations to be performed without triggering allocation or device
// synchronization.
//
// Strides follow the byte-offset convention: stride[i] is the number of
// bytes to advance in memory to move one step along dimension i. A tensor
// is contiguous when its strides match the reference contiguous_stride for
// its shape and element size.

#pragma once

#include <cstddef>
#include <utility>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"

namespace lucid {

// Pure metadata descriptor for a tensor; does not own storage.
//
// The two-argument constructor synthesizes contiguous strides from shape
// and dtype_size, which is the common case when creating output tensors.
// The three-argument constructor accepts explicit strides for non-contiguous
// views (slices, transposes). All constructors are deliberately non-explicit
// to allow aggregate-style initialization in shape-inference helpers.
struct TensorMeta {
    Shape shape;
    Stride stride;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    TensorMeta() = default;

    // Construct a contiguous descriptor; stride is computed from shape and dtype.
    TensorMeta(Shape shape_in, Dtype dtype_in, Device device_in)
        : shape(std::move(shape_in)),
          stride(contiguous_stride(shape, dtype_size(dtype_in))),
          dtype(dtype_in),
          device(device_in) {}

    // Construct with explicit strides (e.g., after a slice or transpose).
    TensorMeta(Shape shape_in, Stride stride_in, Dtype dtype_in, Device device_in)
        : shape(std::move(shape_in)),
          stride(std::move(stride_in)),
          dtype(dtype_in),
          device(device_in) {}

    // Total number of logical elements (product of all dimension sizes).
    std::size_t numel() const noexcept { return shape_numel(shape); }

    // Total bytes in a contiguous layout for this shape and dtype.
    std::size_t nbytes() const noexcept { return numel() * dtype_size(dtype); }

    // Return true when the actual strides match the reference contiguous
    // strides for this shape and element size.
    bool is_contiguous() const noexcept {
        return stride == contiguous_stride(shape, dtype_size(dtype));
    }
};

}  // namespace lucid
