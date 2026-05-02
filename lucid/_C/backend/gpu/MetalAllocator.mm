

#import <Metal/Metal.h>

#include "MetalAllocator.h"

namespace lucid::gpu {

namespace {

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
