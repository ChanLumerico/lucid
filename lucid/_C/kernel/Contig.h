// lucid/_C/kernel/Contig.h
//
// Forward declaration of contiguous_op, a helper used by all CRTP kernel
// forward() methods to guarantee that CPU inputs have a contiguous memory
// layout before being handed to a backend compute function.
//
// Non-contiguous tensors arise from slices and transposes that set
// non-unit strides without copying data. CPU backend kernels rely on
// stride-1 row-major layout, so contiguous_op materializes a fresh copy
// whenever the tensor's actual strides differ from the contiguous
// reference strides. GPU (MLX) kernels handle non-contiguous inputs
// natively via MLX's lazy evaluation, so contiguous_op is a no-op on
// the GPU path.

#pragma once

#include <memory>

namespace lucid {

class TensorImpl;
using TensorImplPtr = std::shared_ptr<TensorImpl>;

// Return a contiguous view (or copy) of a. If a is already contiguous
// the same pointer is returned with no allocation. Otherwise a new
// TensorImpl is allocated with a contiguous CpuStorage whose contents
// are copied from a respecting a's strides.
TensorImplPtr contiguous_op(const TensorImplPtr& a);

}  // namespace lucid
