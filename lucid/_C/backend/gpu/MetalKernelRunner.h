// lucid/_C/backend/gpu/MetalKernelRunner.h
//
// Compile-and-run interface for user-provided Metal Shading Language (MSL)
// compute kernels.  Used by GpuBackend::run_custom_metal_kernel().
//
// Compile flow (compile_metal_kernel):
//   1. Hashes (source, function_name) to form a cache key.
//   2. On cache miss: compiles the MSL source string via
//      newLibraryWithSource:, looks up the named function, and creates a
//      MTLComputePipelineState.  A dedicated MTLCommandQueue is created per
//      pipeline and cached alongside it.
//   3. Returns a MetalKernel that CFRetain-wraps the pipeline state and
//      command queue as opaque void pointers to avoid Objective-C headers in
//      the public interface.
//
// Launch flow (run_metal_kernel):
//   1. Resolves each input Storage to an id<MTLBuffer> via resolve_storage_to_mtl:
//      SharedStorage → directly bridges the shared MTLBuffer handle (no copy).
//      GpuStorage    → calls eval() then newBufferWithBytes (safe copy needed
//                      because MLX arrays are not page-aligned and cannot be
//                      used with newBufferWithBytesNoCopy).
//      CpuStorage    → newBufferWithBytes (copy to shared memory).
//   2. Allocates the output buffer via make_metal_shared.
//   3. Encodes input buffers, optional scalar constants, and the output buffer
//      into a MTLComputeCommandEncoder in that order.
//   4. Dispatches threadgroups, commits, and waits synchronously for completion.
//   5. Returns the output as a SharedStorage so it is immediately CPU-readable.
//
// Thread safety: g_pipeline_cache is protected by g_cache_mutex.  Individual
// MetalKernel objects are not thread-safe; callers must not share a MetalKernel
// across threads.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::gpu {

// RAII wrapper around a compiled MTLComputePipelineState and its command queue.
// Non-copyable; move-only.  pipeline_state and command_queue are CFRetain'd
// id<MTL*> objects stored as void* to keep the header ObjC-free.
struct MetalKernel {
    void* pipeline_state = nullptr;
    void* command_queue = nullptr;
    std::string name;

    MetalKernel() = default;

    MetalKernel(const MetalKernel&) = delete;
    MetalKernel& operator=(const MetalKernel&) = delete;
    MetalKernel(MetalKernel&& o) noexcept
        : pipeline_state(o.pipeline_state),
          command_queue(o.command_queue),
          name(std::move(o.name)) {
        o.pipeline_state = nullptr;
        o.command_queue = nullptr;
    }
    MetalKernel& operator=(MetalKernel&& o) noexcept {
        if (this != &o) {
            release_();
            pipeline_state = o.pipeline_state;
            command_queue = o.command_queue;
            name = std::move(o.name);
            o.pipeline_state = nullptr;
            o.command_queue = nullptr;
        }
        return *this;
    }
    ~MetalKernel() { release_(); }

    // Returns true if the pipeline compiled successfully.
    bool is_valid() const noexcept { return pipeline_state != nullptr; }

private:
    void release_() noexcept;
};

// Metal dispatch geometry: grid is the threadgroup grid in 3-D;
// threads is the threads-per-threadgroup in 3-D.
struct KernelLaunchConfig {
    std::array<std::size_t, 3> grid = {1, 1, 1};
    std::array<std::size_t, 3> threads = {1, 1, 1};
};

// Scalar constant types that can be passed by value into a kernel.
using KernelConstant = std::variant<int, float, std::size_t>;

// Compiles (or retrieves from cache) the named function from the MSL source
// string.  Returns an invalid MetalKernel (is_valid() == false) if the device
// is unavailable; throws on compilation or pipeline creation failure.
MetalKernel compile_metal_kernel(const std::string& source, const std::string& function_name);

// Launches the kernel and waits synchronously for completion.  Returns the
// output as a SharedStorage backed by a MTLResourceStorageModeShared buffer.
// Input buffers are bound at indices 0..n-1, constants at n..n+k-1, and
// the output buffer at index n+k.
Storage run_metal_kernel(const MetalKernel& kernel,
                         const std::vector<Storage>& inputs,
                         const Shape& output_shape,
                         Dtype output_dtype,
                         const KernelLaunchConfig& config,
                         const std::vector<KernelConstant>& constants = {});

}  // namespace lucid::gpu
