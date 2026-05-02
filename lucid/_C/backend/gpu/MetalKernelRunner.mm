// =====================================================================
// Lucid C++ engine — MetalKernelRunner (Phase 18) — Objective-C++ impl.
// =====================================================================

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstddef>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/MemoryStats.h"
#include "MlxBridge.h"       // wrap_mlx_array, to_mlx_shape, to_mlx_dtype
#include "MetalAllocator.h"  // allocate_shared
#include "MetalKernelRunner.h"

namespace lucid::gpu {

// ---------------------------------------------------------------------------
// Module-level shared device (same singleton as MetalAllocator).
// ---------------------------------------------------------------------------

namespace {

id<MTLDevice> runner_device() {
    static id<MTLDevice> dev = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dev = MTLCreateSystemDefaultDevice();
    });
    return dev;
}

// ---------------------------------------------------------------------------
// Pipeline cache: (hash(source + function_name)) → id<MTLComputePipelineState>
// Guarded by a mutex; retain-counted via ObjC ARC.
// ---------------------------------------------------------------------------

struct PipelineEntry {
    id<MTLComputePipelineState> pso  = nil;
    id<MTLCommandQueue>         cq   = nil;
};

std::mutex g_cache_mutex;
std::unordered_map<std::size_t, PipelineEntry> g_pipeline_cache;

std::size_t cache_key(const std::string& source, const std::string& fn_name) {
    std::size_t h = std::hash<std::string>{}(source);
    // Combine with function name hash via FNV-like fold.
    h ^= std::hash<std::string>{}(fn_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------
// MetalKernel::release_
// ---------------------------------------------------------------------------

void MetalKernel::release_() noexcept {
    if (pipeline_state) {
        CFRelease(pipeline_state);
        pipeline_state = nullptr;
    }
    if (command_queue) {
        CFRelease(command_queue);
        command_queue = nullptr;
    }
}

// ---------------------------------------------------------------------------
// compile_metal_kernel
// ---------------------------------------------------------------------------

MetalKernel compile_metal_kernel(const std::string& source,
                                 const std::string& function_name) {
    const std::size_t key = cache_key(source, function_name);

    // Check cache first (read-lock pattern).
    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        auto it = g_pipeline_cache.find(key);
        if (it != g_pipeline_cache.end()) {
            MetalKernel k;
            k.name = function_name;
            // Retain so the MetalKernel owns an independent reference.
            k.pipeline_state = (__bridge_retained void*)it->second.pso;
            k.command_queue  = (__bridge_retained void*)it->second.cq;
            return k;
        }
    }

    id<MTLDevice> dev = runner_device();
    if (!dev)
        return {};  // is_valid() == false

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        ErrorBuilder("compile_metal_kernel")
            .fail(std::string("MTL library compilation failed: ") +
                  (err ? [err.localizedDescription UTF8String] : "unknown"));
    }

    NSString* fn_ns = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> fn = [lib newFunctionWithName:fn_ns];
    if (!fn) {
        ErrorBuilder("compile_metal_kernel")
            .fail("function '" + function_name + "' not found in shader source");
    }

    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        ErrorBuilder("compile_metal_kernel")
            .fail(std::string("pipeline state creation failed: ") +
                  (err ? [err.localizedDescription UTF8String] : "unknown"));
    }

    id<MTLCommandQueue> cq = [dev newCommandQueue];
    if (!cq) {
        ErrorBuilder("compile_metal_kernel").fail("failed to create command queue");
    }

    // Insert into cache.
    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        PipelineEntry entry;
        entry.pso = pso;
        entry.cq  = cq;
        g_pipeline_cache.emplace(key, entry);
    }

    MetalKernel k;
    k.name = function_name;
    k.pipeline_state = (__bridge_retained void*)pso;
    k.command_queue  = (__bridge_retained void*)cq;
    return k;
}

// ---------------------------------------------------------------------------
// Helpers: resolve a Storage to a MTLBuffer suitable for kernel binding.
// ---------------------------------------------------------------------------

namespace {

// Temporary buffers created for CpuStorage uploads are returned alongside
// so the caller can keep them alive until the GPU command completes.
struct BoundBuffer {
    id<MTLBuffer> buf     = nil;
    bool          is_temp = false;  // true → release after waitUntilCompleted
};

BoundBuffer resolve_storage_to_mtl(const Storage& s, id<MTLDevice> dev) {
    if (storage_is_metal_shared(s)) {
        const auto& sh = storage_metal_shared(s);
        // mtl_handle already is the id<MTLBuffer>.
        return {(__bridge id<MTLBuffer>)sh.mtl_handle, /*is_temp=*/false};
    }
    if (storage_is_gpu(s)) {
        const auto& gs = storage_gpu(s);
        if (!gs.arr) {
            ErrorBuilder("run_metal_kernel").fail("null GpuStorage array");
        }
        // Materialise the MLX lazy graph and copy the evaluated data into a
        // Metal-accessible buffer.  We use newBufferWithBytes (copy path) here
        // because:
        //   (a) MLX does not expose an id<MTLBuffer> through its public API.
        //   (b) data<uint8_t>() after eval() returns a CPU-readable pointer that
        //       may not be page-aligned and therefore cannot be wrapped via
        //       newBufferWithBytesNoCopy.
        // For the common hot-path (SharedStorage input), zero-copy applies.
        gs.arr->eval();
        const void* ptr = gs.arr->data<std::uint8_t>();
        if (!ptr) {
            ErrorBuilder("run_metal_kernel").fail("MLX array data pointer is null after eval");
        }
        id<MTLBuffer> tmp = [dev newBufferWithBytes:ptr
                                            length:gs.nbytes
                                           options:MTLResourceStorageModeShared];
        if (!tmp) {
            ErrorBuilder("run_metal_kernel").fail("failed to allocate Metal buffer for GpuStorage");
        }
        return {tmp, /*is_temp=*/true};
    }
    // CpuStorage: copy to a temporary GPU-accessible buffer.
    const auto& cs = storage_cpu(s);
    id<MTLBuffer> tmp = [dev newBufferWithBytes:cs.ptr.get()
                                        length:cs.nbytes
                                       options:MTLResourceStorageModeShared];
    return {tmp, /*is_temp=*/true};
}

}  // namespace

// ---------------------------------------------------------------------------
// run_metal_kernel
// ---------------------------------------------------------------------------

Storage run_metal_kernel(const MetalKernel&              kernel,
                         const std::vector<Storage>&     inputs,
                         const Shape&                    output_shape,
                         Dtype                           output_dtype,
                         const KernelLaunchConfig&       config,
                         const std::vector<KernelConstant>& constants) {
    if (!kernel.is_valid())
        ErrorBuilder("run_metal_kernel").fail("kernel is invalid (compilation failed?)");

    id<MTLDevice> dev = runner_device();
    auto pso = (__bridge id<MTLComputePipelineState>)kernel.pipeline_state;
    auto cq  = (__bridge id<MTLCommandQueue>)kernel.command_queue;

    // Allocate output buffer via MetalAllocator so the result can be consumed
    // as a SharedStorage (zero-copy handoff to the next CPU or GPU op).
    const std::size_t out_numel = shape_numel(output_shape);
    const std::size_t out_bytes = out_numel * dtype_size(output_dtype);
    OwnedMetalBuffer out_owned  = make_metal_shared(out_bytes);
    if (!out_owned.buf.cpu_ptr) {
        ErrorBuilder("run_metal_kernel").fail("failed to allocate output Metal buffer");
    }
    id<MTLBuffer> out_buf =
        (__bridge id<MTLBuffer>)out_owned.buf.mtl_handle;

    // Resolve input Storages to MTLBuffers.
    std::vector<BoundBuffer> bound;
    bound.reserve(inputs.size());
    for (const auto& s : inputs)
        bound.push_back(resolve_storage_to_mtl(s, dev));

    // Encode and dispatch.
    id<MTLCommandBuffer>        cmd  = [cq commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pso];

    // Bind input buffers at indices 0..N-1.
    for (std::size_t i = 0; i < bound.size(); ++i)
        [enc setBuffer:bound[i].buf offset:0 atIndex:static_cast<NSUInteger>(i)];

    // Bind constants after inputs.
    const std::size_t const_base = bound.size();
    for (std::size_t ci = 0; ci < constants.size(); ++ci) {
        std::visit(
            [&](auto v) {
                auto typed = v;
                [enc setBytes:&typed
                       length:sizeof(typed)
                      atIndex:static_cast<NSUInteger>(const_base + ci)];
            },
            constants[ci]);
    }

    // Output buffer at index N + num_constants.
    const NSUInteger out_idx =
        static_cast<NSUInteger>(bound.size() + constants.size());
    [enc setBuffer:out_buf offset:0 atIndex:out_idx];

    // Dispatch.
    MTLSize g  = MTLSizeMake(config.grid[0], config.grid[1], config.grid[2]);
    MTLSize tg = MTLSizeMake(config.threads[0], config.threads[1], config.threads[2]);
    [enc dispatchThreadgroups:g threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Build result SharedStorage.
    SharedStorage result;
    result.cpu_ptr    = out_owned.buf.cpu_ptr;
    result.mtl_handle = out_owned.buf.mtl_handle;
    result.nbytes     = out_bytes;
    result.dtype      = output_dtype;
    result.owner      = std::move(out_owned.owner);

    return Storage{std::move(result)};
}

}  // namespace lucid::gpu
