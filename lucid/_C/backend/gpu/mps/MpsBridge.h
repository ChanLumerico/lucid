// lucid/_C/backend/gpu/mps/MpsBridge.h
//
// MLX ↔ MPSGraph bridge primitives.  Callable from plain C++ (`.cpp` / `.h`)
// even though the underlying implementation is Obj-C++ — all Obj-C types
// (`id<MTLBuffer>`, `id<MTLDevice>`, `id<MTLCommandQueue>`) are erased to
// `void*` at this boundary.  Casts back to Obj-C happen via `__bridge` in
// MpsBridge.mm and MpsKernels.mm.
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

// Process-wide initialised MTLDevice / MTLCommandQueue.  void* values are
// the Obj-C `id` pointers; cast via `(__bridge id<MTLDevice>)` etc. inside
// `.mm` callers.  Both calls are lazy + thread-safe.
void* shared_mtl_device();
void* shared_mtl_queue();

// True if MetalPerformanceShadersGraph is available + the bridge initialised
// successfully.  False return → all `should_dispatch_*` heuristics should
// short-circuit to false and MLX path runs.
bool bridge_available();

// View into an MLX-owned MTLBuffer.  After this call:
//   • The array is evaluated AND completed (status == available).
//   • `mtl_buffer` is the underlying MTL::Buffer*, owned by MLX.
//   • `offset_bytes` is `arr.offset() * arr.itemsize()` — apply when
//     constructing MPSGraphTensorData.
//   • `nbytes` is the logical size of the array's data slice.
//
// Caller MUST NOT release `mtl_buffer`.  The MLX array remains alive for
// the duration the caller holds the BufferView.
struct BufferView {
    void* mtl_buffer;
    std::size_t offset_bytes;
    std::size_t nbytes;
};
BufferView array_to_buffer(const ::mlx::core::array& arr);

// Wrap a caller-allocated MTLBuffer into a fresh leaf mlx::core::array.
// Caller must have one strong reference to the buffer before calling; this
// function transfers that reference.  When the returned array (and all its
// copies) die, the buffer is released exactly once.
//
// `shape` is the logical shape; the underlying buffer must have at least
//   prod(shape) * sizeof(dtype) bytes starting at offset_bytes.
::mlx::core::array buffer_to_array(void* mtl_buffer,
                                   std::vector<int> shape,
                                   Dtype dt,
                                   std::size_t offset_bytes = 0);

// Block until all in-flight MPS command buffers complete.  Called from
// tests + by callers that need to read the result on the CPU.
void wait_all();

}  // namespace lucid::gpu::mps
