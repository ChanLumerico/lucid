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

// Return a dense, row-major copy of ``a`` whenever its strides diverge
// from the canonical contiguous layout; return ``a`` unchanged when it
// is already contiguous.
//
// This is the public entry point every CRTP kernel ``forward()`` calls
// on CPU inputs before handing storage to a backend compute routine.
// Slices and transposes produce non-unit strides without copying data,
// but the Accelerate (CPU) kernels assume stride-1 row-major access, so
// materialising a contiguous copy is required to avoid garbage reads.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor.  Must be non-null.  May live on either device.
//
// Returns
// -------
// TensorImplPtr
//     Either ``a`` itself (when ``a->is_contiguous()`` is true) or a
//     freshly allocated :class:`TensorImpl` with the same shape, dtype,
//     and device but a contiguous storage whose elements are copied
//     from ``a`` honouring its strides and ``storage_offset``.
//
// Notes
// -----
// **Autograd.**  Implemented via ``ContiguousBackward`` so the operation
// participates in the autograd graph; the backward is the identity on
// the incoming gradient (a clone to give the upstream node an owning
// buffer).  See ``ops/utils/Contiguous.cpp``.
//
// **GPU path.**  On :data:`Device::GPU` the underlying MLX backend
// handles non-contiguous storage natively via lazy evaluation, so the
// dispatched ``contiguous`` is typically a cheap pass-through.  Kernel
// ``forward()`` helpers therefore guard the call with a
// ``device() == CPU && !is_contiguous()`` predicate.
//
// **Aliasing.**  The returned tensor never aliases ``a``'s storage when
// a copy was performed; consumers may freely mutate it without
// violating the version counter on ``a``.
//
// Raises
// ------
// LucidError
//     If ``a`` is null.
//
// See Also
// --------
// :class:`ContiguousBackward` — the underlying op implementation.
// :class:`UnaryKernel`, :class:`BinaryKernel`, :class:`ReduceKernel` —
// CRTP bases that invoke this helper during forward.
TensorImplPtr contiguous_op(const TensorImplPtr& a);

}  // namespace lucid
