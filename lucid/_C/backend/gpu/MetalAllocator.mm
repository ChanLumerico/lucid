// =====================================================================
// Lucid C++ engine — MetalAllocator (Phase 9.1) — Objective-C++ impl.
// =====================================================================
//
// Compiled as Objective-C++ (.mm) because the Metal API requires ObjC
// message dispatch.  All other engine files (.cpp) include only the C++
// header MetalAllocator.h which exposes plain-C++ entry points.

#import <Metal/Metal.h>

#include "MetalAllocator.h"

namespace lucid::gpu {

// ---------------------------------------------------------------------------
// Module-level singleton device — acquired once, retained forever.
// ---------------------------------------------------------------------------

namespace {

id<MTLDevice> shared_device() {
    static id<MTLDevice> dev = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dev = MTLCreateSystemDefaultDevice();
    });
    return dev;
}

}  // namespace

// ---------------------------------------------------------------------------
// allocate_shared
// ---------------------------------------------------------------------------

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

    // Retain: the caller (via OwnedMetalBuffer / deallocate_shared) owns one ref.
    // ARC would normally handle this, but we're crossing into C++ land, so we
    // retain manually and balance with the CFRelease in deallocate_shared.
    CFRetain((__bridge CFTypeRef)buf);

    return {
        buf.contents,
        (__bridge void*)buf,
        nbytes,
    };
}

// ---------------------------------------------------------------------------
// deallocate_shared
// ---------------------------------------------------------------------------

void deallocate_shared(MetalBuffer& buf) noexcept {
    if (buf.mtl_handle) {
        CFRelease(buf.mtl_handle);
        buf.mtl_handle = nullptr;
    }
    buf.cpu_ptr = nullptr;
    buf.nbytes  = 0;
}

// ---------------------------------------------------------------------------
// wrap_existing
// ---------------------------------------------------------------------------

MetalBuffer wrap_existing(void* cpu_ptr, std::size_t nbytes) {
    if (!cpu_ptr || nbytes == 0)
        return {nullptr, nullptr, 0};

    id<MTLDevice> dev = shared_device();
    if (!dev)
        return {nullptr, nullptr, 0};

    // newBufferWithBytesNoCopy: no copy; the MTLBuffer is a no-copy view over
    // the existing page-aligned CPU allocation. We pass nil as the deallocator
    // because the underlying memory is managed by the caller.
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

// ---------------------------------------------------------------------------
// make_metal_shared
// ---------------------------------------------------------------------------

OwnedMetalBuffer make_metal_shared(std::size_t nbytes) {
    MetalBuffer raw = allocate_shared(nbytes);
    if (!raw.cpu_ptr)
        return {raw, nullptr};

    // shared_ptr<void> with a deleter that releases the Metal buffer.
    auto owner = std::shared_ptr<void>(
        raw.cpu_ptr,
        [raw](void*) mutable { deallocate_shared(raw); });

    return {raw, std::move(owner)};
}

}  // namespace lucid::gpu
