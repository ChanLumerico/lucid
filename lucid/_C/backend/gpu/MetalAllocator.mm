// lucid/_C/backend/gpu/MetalAllocator.mm
//
// Implements the MetalBuffer allocation and lifetime management functions
// declared in MetalAllocator.h.
//
// shared_device() is a Meyers-singleton that obtains the system default
// MTLDevice exactly once using dispatch_once.  This is safe for concurrent
// callers and avoids redundant MTLCreateSystemDefaultDevice calls.
//
// allocate_shared: calls newBufferWithLength:options: with
//   MTLResourceStorageModeShared so that the returned pages are simultaneously
//   CPU- and GPU-accessible.  CFRetain is called on the resulting MTLBuffer
//   before bridging to void* to balance the release in deallocate_shared.
//
// deallocate_shared: calls CFRelease on the stored opaque handle to decrement
//   the buffer's retain count.  After release the cpu_ptr pointer must not be
//   accessed by the CPU or the GPU.
//
// wrap_existing: uses newBufferWithBytesNoCopy which creates a zero-copy view
//   of an existing page-aligned allocation.  Ownership of the underlying pages
//   is NOT transferred to the MTLBuffer; the caller is responsible for their
//   lifetime.

#import <Metal/Metal.h>

#include "MetalAllocator.h"

namespace lucid::gpu {

namespace {

// Returns the process-wide default Metal device, created once on first call.
id<MTLDevice> shared_device() {
    static id<MTLDevice> dev = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dev = MTLCreateSystemDefaultDevice();
    });
    return dev;
}

}

MetalBuffer allocate_shared(std::size_t nbytes) {
    if (nbytes == 0)
        return {nullptr, nullptr, 0};

    id<MTLDevice> dev = shared_device();
    if (!dev)
        return {nullptr, nullptr, 0};

    id<MTLBuffer> buf = [dev newBufferWithLength:nbytes
                                        options:MTLResourceStorageModeShared];
    if (!buf)
        return {nullptr, nullptr, 0};

    CFRetain((__bridge CFTypeRef)buf);

    return {
        buf.contents,
        (__bridge void*)buf,
        nbytes,
    };
}

void deallocate_shared(MetalBuffer& buf) noexcept {
    if (buf.mtl_handle) {
        CFRelease(buf.mtl_handle);
        buf.mtl_handle = nullptr;
    }
    buf.cpu_ptr = nullptr;
    buf.nbytes  = 0;
}

MetalBuffer wrap_existing(void* cpu_ptr, std::size_t nbytes) {
    if (!cpu_ptr || nbytes == 0)
        return {nullptr, nullptr, 0};

    id<MTLDevice> dev = shared_device();
    if (!dev)
        return {nullptr, nullptr, 0};

    id<MTLBuffer> buf =
        [dev newBufferWithBytesNoCopy:cpu_ptr
                               length:nbytes
                              options:MTLResourceStorageModeShared
                          deallocator:nil];
    if (!buf)
        return {nullptr, nullptr, 0};

    CFRetain((__bridge CFTypeRef)buf);
    return {cpu_ptr, (__bridge void*)buf, nbytes};
}

OwnedMetalBuffer make_metal_shared(std::size_t nbytes) {
    MetalBuffer raw = allocate_shared(nbytes);
    if (!raw.cpu_ptr)
        return {raw, nullptr};

    auto owner = std::shared_ptr<void>(
        raw.cpu_ptr,
        [raw](void*) mutable { deallocate_shared(raw); });

    return {raw, std::move(owner)};
}

}
