// lucid/_C/backend/gpu/mps/MpsBridge.mm
//
// Obj-C++ implementation of the MLX ↔ MPSGraph bridge.  See MpsBridge.h.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <atomic>
#include <mutex>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include "MpsBridge.h"

namespace lucid::gpu::mps {

namespace {

std::once_flag g_init_once;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
std::atomic<bool> g_bridge_ok{false};

void init_bridge_once() {
    std::call_once(g_init_once, []() {
        // Apple Silicon has one MTLDevice per system; MTLCreateSystemDefaultDevice
        // returns the same object MLX uses internally.  Going through Apple's
        // API directly avoids depending on MLX's metal-cpp include path.
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return;
        g_queue = [g_device newCommandQueue];
        if (!g_queue) return;
        [g_queue setLabel:@"lucid-mps-dispatch"];
        g_bridge_ok.store(true, std::memory_order_release);
    });
}

}  // namespace

void* shared_mtl_device() {
    init_bridge_once();
    return (__bridge void*)g_device;
}

void* shared_mtl_queue() {
    init_bridge_once();
    return (__bridge void*)g_queue;
}

bool bridge_available() {
    init_bridge_once();
    return g_bridge_ok.load(std::memory_order_acquire);
}

BufferView array_to_buffer(const ::mlx::core::array& arr) {
    // Force materialisation + wait for completion.  We need the kernel that
    // produced `arr` to finish before MPSGraph reads the buffer — MLX's
    // command queue and ours are distinct, so Metal will not insert an
    // implicit fence.  `wait()` is the explicit cross-queue barrier.
    auto& mutable_arr = const_cast<::mlx::core::array&>(arr);
    mutable_arr.eval();
    mutable_arr.wait();

    const auto& buf = arr.buffer();
    // Buffer::ptr() on a const Buffer returns const void*; cast away const
    // because the underlying MTLBuffer is mutable and we hand it to
    // MPSGraph which expects id<MTLBuffer>.
    void* mtl_buffer_raw = const_cast<void*>(buf.ptr());
    const std::size_t offset_bytes = static_cast<std::size_t>(arr.offset()) * arr.itemsize();
    const std::size_t nbytes = arr.nbytes();
    return BufferView{mtl_buffer_raw, offset_bytes, nbytes};
}

::mlx::core::array buffer_to_array(void* mtl_buffer,
                                   std::vector<int> shape,
                                   Dtype dt,
                                   std::size_t offset_bytes) {
    // Wrap as an allocator::Buffer with a deleter that transfers ownership
    // back to ARC.  We don't go through MetalAllocator::make_buffer because
    // that path is for buffers MLX itself allocates — ours came from
    // [MTLBuffer newBufferWithLength:] in the kernel layer.
    ::mlx::core::allocator::Buffer wrapped{mtl_buffer};

    auto deleter = [](::mlx::core::allocator::Buffer b) {
        // `(__bridge_transfer id<MTLBuffer>)` moves the +1 retain into ARC's
        // scope; assigning to nil immediately releases.
        @autoreleasepool {
            id<MTLBuffer> dropped = (__bridge_transfer id<MTLBuffer>)b.ptr();
            (void)dropped;
        }
    };

    ::mlx::core::Shape mlx_shape;
    mlx_shape.reserve(shape.size());
    for (int d : shape) mlx_shape.push_back(d);

    // Map Lucid Dtype → mlx::core::Dtype.  Reuse the existing helper from
    // MlxBridge but without pulling its full header (which would create a
    // dependency cycle).  The mapping is small and stable.
    ::mlx::core::Dtype mlx_dt = ::mlx::core::float32;
    switch (dt) {
        case Dtype::F32: mlx_dt = ::mlx::core::float32; break;
        case Dtype::F16: mlx_dt = ::mlx::core::float16; break;
        case Dtype::F64: mlx_dt = ::mlx::core::float64; break;
        case Dtype::I32: mlx_dt = ::mlx::core::int32; break;
        case Dtype::I64: mlx_dt = ::mlx::core::int64; break;
        default:
            throw std::runtime_error(
                "lucid::gpu::mps::buffer_to_array: unsupported Dtype for MPSGraph");
    }

    ::mlx::core::array out(wrapped, std::move(mlx_shape), mlx_dt, deleter);
    // Offset support: if the caller wants a view starting at offset_bytes,
    // we'd need to slice — for the pilot, require offset_bytes == 0.
    if (offset_bytes != 0) {
        throw std::runtime_error(
            "lucid::gpu::mps::buffer_to_array: nonzero offset_bytes not yet supported");
    }
    return out;
}

void wait_all() {
    init_bridge_once();
    if (!g_bridge_ok.load(std::memory_order_acquire)) return;
    // Issue an empty command buffer + waitUntilCompleted.  This drains
    // the queue without inspecting individual ops.
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
}

}  // namespace lucid::gpu::mps
