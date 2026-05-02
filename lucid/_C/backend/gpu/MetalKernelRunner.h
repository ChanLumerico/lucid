#pragma once

// =====================================================================
// Lucid C++ engine — MetalKernelRunner (Phase 18)
// =====================================================================
//
// Provides a thin C++ façade over Metal compute pipelines so that
// operations which are not (or not yet) expressible via MLX can fall
// back to a hand-written Metal shader.
//
// Typical usage:
//
//   auto k = lucid::gpu::compile_metal_kernel(source, "my_kernel");
//   Storage result = lucid::gpu::run_metal_kernel(
//       k, {input_storage}, out_shape, Dtype::F32,
//       {{1024,1,1}, {256,1,1}});
//
// The compiled pipeline state is cached keyed by a hash of the source
// string + function name, so repeated calls with the same kernel do not
// re-compile.
//
// Layer: backend/gpu/. Implementation in MetalKernelRunner.mm.

#include <array>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::gpu {

// ---- MetalKernel -----------------------------------------------------------
//
// Compiled pipeline state + dedicated command queue for a single kernel.
// Lifecycle: returned by compile_metal_kernel, destroyed when it goes out of
// scope (the destructor is implemented in .mm and calls CFRelease on the
// opaque ObjC objects).
//
struct MetalKernel {
    void*       pipeline_state = nullptr;  ///< id<MTLComputePipelineState> (retained)
    void*       command_queue  = nullptr;  ///< id<MTLCommandQueue> (retained)
    std::string name;

    MetalKernel() = default;

    // Non-copyable (owns Metal objects); moveable.
    MetalKernel(const MetalKernel&)            = delete;
    MetalKernel& operator=(const MetalKernel&) = delete;
    MetalKernel(MetalKernel&& o) noexcept
        : pipeline_state(o.pipeline_state)
        , command_queue(o.command_queue)
        , name(std::move(o.name)) {
        o.pipeline_state = nullptr;
        o.command_queue  = nullptr;
    }
    MetalKernel& operator=(MetalKernel&& o) noexcept {
        if (this != &o) {
            release_();
            pipeline_state = o.pipeline_state;
            command_queue  = o.command_queue;
            name           = std::move(o.name);
            o.pipeline_state = nullptr;
            o.command_queue  = nullptr;
        }
        return *this;
    }
    ~MetalKernel() { release_(); }

    bool is_valid() const noexcept { return pipeline_state != nullptr; }

private:
    void release_() noexcept;  // implemented in .mm
};

// ---- KernelLaunchConfig ----------------------------------------------------

struct KernelLaunchConfig {
    std::array<std::size_t, 3> grid    = {1, 1, 1};  ///< threadgroups per grid
    std::array<std::size_t, 3> threads = {1, 1, 1};  ///< threads per threadgroup
};

// ---- Constant value type ---------------------------------------------------
//
// Scalar constants pushed via setBytes:atIndex:.
//
using KernelConstant = std::variant<int, float, std::size_t>;

// ---- compile_metal_kernel --------------------------------------------------
//
// Compile `source` (MSL text) and extract the function named
// `function_name`. Returns a valid MetalKernel on success, or an invalid one
// (is_valid() == false) on failure.
//
// Results are cached by (hash(source) ^ hash(function_name)) so repeated
// calls with the same kernel are O(1) after the first compilation.
//
MetalKernel compile_metal_kernel(const std::string& source,
                                 const std::string& function_name);

// ---- run_metal_kernel ------------------------------------------------------
//
// Dispatch `kernel` with the provided inputs.  Each Storage is bound to a
// successive Metal buffer slot (index 0, 1, 2, …).  `constants` are bound
// after the input buffers.  The output buffer is allocated as the final slot.
//
// Storage routing:
//   CpuStorage   → newBufferWithBytes (synchronous copy to GPU, fallback)
//   GpuStorage   → extract internal MTLBuffer from the MLX array
//   SharedStorage→ mtl_handle directly (zero-copy)
//
// Returns a GpuStorage wrapping the output buffer.
//
Storage run_metal_kernel(const MetalKernel&              kernel,
                         const std::vector<Storage>&     inputs,
                         const Shape&                    output_shape,
                         Dtype                           output_dtype,
                         const KernelLaunchConfig&       config,
                         const std::vector<KernelConstant>& constants = {});

}  // namespace lucid::gpu
