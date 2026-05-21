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

// RAII handle for a compiled Metal compute pipeline.
//
// Wraps a ``MTLComputePipelineState`` plus its dedicated
// ``MTLCommandQueue`` (both ``CFRetain``-ed) so that one pipeline can
// be invoked many times without re-compiling MSL source.  The Obj-C
// types are erased to ``void*`` here to keep this header includable
// from plain C++.  Non-copyable, move-only.
//
// Attributes
// ----------
// pipeline_state : void*
//     Opaque ``id<MTLComputePipelineState>``.  ``nullptr`` if
//     compilation failed.
// command_queue : void*
//     Opaque ``id<MTLCommandQueue>``.  Owned independently from any
//     other queue so kernel launches do not contend with MLX's own
//     queue.
// name : std::string
//     The MSL function name compiled into ``pipeline_state``.  Useful
//     for debug logs.
//
// Notes
// -----
// Released in :meth:`release_`.  Move constructors null out the
// source so the destructor only releases once.
//
// Warns
// -----
// Individual ``MetalKernel`` objects are **not** thread-safe.
// Callers must not share a single kernel across threads
// concurrently — clone the handle or guard with a mutex.
//
// See Also
// --------
// :func:`compile_metal_kernel` — factory that produces these.
// :func:`run_metal_kernel` — invokes the compiled pipeline.
struct MetalKernel {
    void* pipeline_state = nullptr;
    void* command_queue = nullptr;
    std::string name;

    // Default-construct an invalid kernel handle.
    //
    // Equivalent to a moved-from state — :meth:`is_valid` returns
    // ``false`` until populated by :func:`compile_metal_kernel`.
    MetalKernel() = default;

    MetalKernel(const MetalKernel&) = delete;
    MetalKernel& operator=(const MetalKernel&) = delete;
    // Move-construct from ``o``, nulling its handles.
    //
    // Steals ``pipeline_state`` / ``command_queue`` / ``name`` and
    // leaves ``o`` in a valid empty state safe for destruction.
    MetalKernel(MetalKernel&& o) noexcept
        : pipeline_state(o.pipeline_state),
          command_queue(o.command_queue),
          name(std::move(o.name)) {
        o.pipeline_state = nullptr;
        o.command_queue = nullptr;
    }
    // Move-assign, releasing any previously held pipeline first.
    //
    // Calls :meth:`release_` on the existing contents before stealing
    // from ``o``.  Self-assignment is a no-op.
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
    // Release the pipeline + queue on destruction.
    ~MetalKernel() { release_(); }

    // Return ``true`` iff compilation produced a usable pipeline.
    //
    // A ``false`` return means either Metal is unavailable on this
    // system or :func:`compile_metal_kernel` failed silently (e.g.
    // device offline).  Hard MSL syntax errors throw rather than
    // returning here.
    //
    // Returns
    // -------
    // bool
    //     ``true`` iff ``pipeline_state != nullptr``.
    bool is_valid() const noexcept { return pipeline_state != nullptr; }

private:
    // Release the retained Metal objects and clear the handle fields.
    //
    // Notes
    // -----
    // Idempotent — safe to call on an already-released kernel.
    void release_() noexcept;
};

// Threadgroup geometry for :func:`run_metal_kernel`.
//
// Mirrors the two ``MTLSize`` arguments passed to
// ``dispatchThreadgroups:threadsPerThreadgroup:`` — the outer grid
// of threadgroups and the per-threadgroup thread count, both in 3-D.
//
// Attributes
// ----------
// grid : std::array<std::size_t, 3>
//     Threadgroup grid extent (x, y, z).  Defaults to ``{1, 1, 1}``.
// threads : std::array<std::size_t, 3>
//     Threads per threadgroup (x, y, z).  Defaults to ``{1, 1, 1}``.
//     Total ``threads`` product must not exceed the device's
//     ``maxTotalThreadsPerThreadgroup``.
//
// Notes
// -----
// Total threads launched = product(grid) * product(threads).  Callers
// are responsible for sizing this to cover ``output_shape``.
struct KernelLaunchConfig {
    std::array<std::size_t, 3> grid = {1, 1, 1};
    std::array<std::size_t, 3> threads = {1, 1, 1};
};

// Scalar constant passable by value to a Metal kernel argument slot.
//
// Variant over the small set of scalar types Lucid currently emits
// into a kernel's argument buffer: 32-bit signed int, 32-bit float,
// and ``std::size_t`` (typically a length / stride).
using KernelConstant = std::variant<int, float, std::size_t>;

// Compile (or fetch from the global cache) a Metal compute pipeline.
//
// Hashes ``(source, function_name)`` to form a cache key, then on
// miss compiles ``source`` via ``newLibraryWithSource:``, looks up
// ``function_name`` inside the resulting library, and creates a
// ``MTLComputePipelineState`` + dedicated ``MTLCommandQueue``.
//
// Parameters
// ----------
// source : const std::string&
//     Complete MSL source text containing at least one ``kernel``
//     function named ``function_name``.
// function_name : const std::string&
//     The MSL kernel function symbol to bind.
//
// Returns
// -------
// MetalKernel
//     A valid kernel handle on success.  An invalid handle
//     (``is_valid() == false``) is returned when Metal is unavailable
//     (no GPU device on the host).
//
// Raises
// ------
// std::runtime_error
//     On MSL compilation failure or pipeline-state creation error.
//
// Notes
// -----
// The pipeline cache is protected by an internal mutex, so concurrent
// callers safely share compiled kernels.  Cached entries live for
// the lifetime of the process.
//
// See Also
// --------
// :func:`run_metal_kernel` — invokes the compiled pipeline.
MetalKernel compile_metal_kernel(const std::string& source, const std::string& function_name);

// Encode + dispatch a compiled kernel; block until it completes.
//
// Allocates an output ``SharedStorage`` of ``output_shape`` /
// ``output_dtype``, binds ``inputs`` at indices ``0..n-1``,
// ``constants`` at ``n..n+k-1``, and the output buffer at ``n+k``,
// then dispatches the threadgroup grid given by ``config`` and waits
// synchronously for the GPU to finish.
//
// Parameters
// ----------
// kernel : const MetalKernel&
//     Compiled pipeline produced by :func:`compile_metal_kernel`.
//     Must satisfy ``kernel.is_valid() == true``.
// inputs : const std::vector<Storage>&
//     Input tensors.  ``SharedStorage`` is bridged in place;
//     ``GpuStorage`` is materialised (``eval()``) and copied into a
//     fresh shared buffer; ``CpuStorage`` is copied verbatim.
// output_shape : const Shape&
//     Logical shape of the output tensor.
// output_dtype : Dtype
//     Element dtype of the output tensor.
// config : const KernelLaunchConfig&
//     Threadgroup grid + per-threadgroup thread geometry.
// constants : const std::vector<KernelConstant>&, optional
//     Scalar constants bound after the input buffers.  Defaults to
//     an empty list.
//
// Returns
// -------
// Storage
//     ``SharedStorage`` backed by a unified-memory buffer; the result
//     is immediately CPU-readable on return (the call waits for
//     completion).
//
// Raises
// ------
// std::runtime_error
//     If the kernel is invalid, an input type cannot be resolved, or
//     the GPU dispatch fails.
//
// Notes
// -----
// Returning ``SharedStorage`` (rather than ``GpuStorage``) trades
// some bandwidth for caller flexibility — the result can flow into
// either CPU or GPU consumers without an extra copy.
//
// See Also
// --------
// :func:`compile_metal_kernel` — produces ``kernel``.
// :func:`make_metal_shared` — backs the returned storage.
Storage run_metal_kernel(const MetalKernel& kernel,
                         const std::vector<Storage>& inputs,
                         const Shape& output_shape,
                         Dtype output_dtype,
                         const KernelLaunchConfig& config,
                         const std::vector<KernelConstant>& constants = {});

}  // namespace lucid::gpu
