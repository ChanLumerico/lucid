// lucid/_C/backend/gpu/mps/MpsBridge.h
//
// MLX ↔ MPSGraph bridge primitives.  Callable from plain C++ (`.cpp` / `.h`)
// even though the underlying implementation is Obj-C++ — all Obj-C types
// (`id<MTLBuffer>`, `id<MTLDevice>`, `id<MTLCommandQueue>`) are erased to
// `void*` at this boundary.  Casts back to Obj-C happen via `__bridge` in
// MpsBridge.mm.  Sole consumer is the `lucid.compile` MPSGraph emitter
// (`compile/MpsBuilder.mm`, `compile/CompiledExecutable.mm`).
//
// Lifetime model:
//   • Process-wide singleton MTLDevice + MTLCommandQueue, lazily created on
//     first call.  Same MTLDevice as MLX's (via mlx::core::metal::device).
//   • `array_to_buffer` evaluates the array (forcing the MLX kernel that
//     produced it to run + complete) and returns a non-owning view of the
//     underlying MTLBuffer.  Caller must NOT release.
//   • `buffer_to_array` takes a fresh MTLBuffer that the caller allocated
//     (refcount == 1 going in) and hands ownership to a new mlx::core::array;
//     the array's deleter releases the buffer when the array dies.

#pragma once

#include <cstddef>
#include <vector>

#include "../../../core/Dtype.h"

namespace mlx::core {
class array;
}  // namespace mlx::core

namespace lucid::gpu::mps {

// Return the process-wide ``MTLDevice`` used by both MLX and MPSGraph.
//
// Lazily initialised on first call.  Reuses MLX's device pointer
// (``mlx::core::metal::device``) so MPS kernels and MLX kernels
// dispatch onto the same hardware queue family.
//
// Returns
// -------
// void*
//     Opaque ``id<MTLDevice>``.  Cast via ``(__bridge id<MTLDevice>)``
//     inside ``.mm`` callers.  Never null on a supported host.
//
// Notes
// -----
// Thread-safe; the underlying init is guarded by a one-shot flag.
void* shared_mtl_device();

// Return the process-wide ``MTLCommandQueue`` used by MPS kernels.
//
// Distinct from MLX's queue so MPS dispatches and MLX dispatches do
// not contend on the same encoder.  Lazily initialised.
//
// Returns
// -------
// void*
//     Opaque ``id<MTLCommandQueue>``.  Cast via
//     ``(__bridge id<MTLCommandQueue>)`` inside ``.mm`` callers.
//
// Notes
// -----
// Thread-safe.
void* shared_mtl_queue();

// Total bytes the Metal device currently has allocated (``MTLDevice
// currentAllocatedSize``).  Unlike MLX's allocator peak this includes the
// compiled MPSGraph executable's internal activations + run_executable output
// buffers — the metric needed to measure a compiled training step's true GPU
// footprint (e.g. the deep-backward memory-pressure investigation).
std::size_t metal_device_allocated_bytes();

// Non-owning view of an MLX-array's underlying ``MTLBuffer``.
//
// Returned by :func:`array_to_buffer` after the array has been
// evaluated + its producing kernel has reached the ``available``
// status.  All fields refer to memory owned by MLX — callers must
// not release ``mtl_buffer`` or outlive the source array.
//
// Attributes
// ----------
// mtl_buffer : void*
//     Opaque ``id<MTLBuffer>`` referencing MLX's allocation.
// offset_bytes : std::size_t
//     Byte offset from the start of ``mtl_buffer`` to the array's
//     logical data slice (``arr.offset() * arr.itemsize()``).  Pass
//     this when constructing an ``MPSGraphTensorData``.
// nbytes : std::size_t
//     Byte length of the array's data slice (logical size, not the
//     full buffer capacity).
//
// Warns
// -----
// Calling release on ``mtl_buffer`` is a double-free.  The caller
// MUST keep the source ``mlx::core::array`` alive while any
// ``BufferView`` referring to it is in flight.
struct BufferView {
    void* mtl_buffer;
    std::size_t offset_bytes;
    std::size_t nbytes;
};

// Materialise an MLX array and view its backing ``MTLBuffer``.
//
// Forces ``arr`` to evaluate (running any pending MLX graph nodes
// that produced it) and blocks until the producing command buffer
// reaches the ``available`` status, then returns a non-owning
// view of the buffer + offset + size.
//
// Parameters
// ----------
// arr : const mlx::core::array&
//     The MLX array whose buffer the caller wishes to bind into an
//     MPSGraph dispatch.
//
// Returns
// -------
// BufferView
//     Triple of (buffer handle, byte offset, byte length).  All
//     fields are valid for the lifetime of ``arr``.
//
// Notes
// -----
// Used by the ``lucid.compile`` MPSGraph emitter to obtain Obj-C
// ``MTLBuffer`` handles for ``MPSGraphTensorData`` construction
// without copying.
BufferView array_to_buffer(const ::mlx::core::array& arr);

// Wrap a caller-allocated ``MTLBuffer`` as a fresh leaf MLX array.
//
// The caller must hold exactly one strong reference to
// ``mtl_buffer`` going in; this function **transfers** that
// reference into the returned ``mlx::core::array`` (and any copies
// of it).  When the final copy dies, the buffer is released exactly
// once via the array's custom deleter.
//
// Parameters
// ----------
// mtl_buffer : void*
//     Opaque ``id<MTLBuffer>`` allocated by the caller, typically via
//     MPSGraph's executable returning a fresh output buffer.
// shape : std::vector<int>
//     Logical MLX shape (int32 dimensions).  The buffer must hold
//     at least ``prod(shape) * sizeof(dt)`` bytes starting at
//     ``offset_bytes``.
// dt : Dtype
//     Lucid dtype of the elements.  Translated to MLX's dtype enum
//     internally.
// offset_bytes : std::size_t, optional
//     Byte offset into ``mtl_buffer`` where the array's data begins.
//     Defaults to 0.
//
// Returns
// -------
// mlx::core::array
//     A leaf array (no upstream graph dependency) backed by
//     ``mtl_buffer``.
//
// Warns
// -----
// Calling this with a buffer that the caller still intends to
// release elsewhere causes a double-free.  Hand off ownership
// cleanly.
::mlx::core::array
buffer_to_array(void* mtl_buffer, std::vector<int> shape, Dtype dt, std::size_t offset_bytes = 0);

// Block until every in-flight MPS command buffer has completed.
//
// Provided primarily for the test suite + callers that need
// CPU-visible results from a kernel before the next op enqueues.
// Routine code paths normally do not need this — MLX consumers of
// an MPS-produced array automatically sync via :func:`array_to_buffer`.
//
// Notes
// -----
// Synchronous from the caller's perspective.
void wait_all();

}  // namespace lucid::gpu::mps
