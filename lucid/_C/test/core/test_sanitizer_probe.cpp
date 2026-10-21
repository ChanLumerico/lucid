// lucid/_C/test/core/test_sanitizer_probe.cpp
// DS-3 (LCD-262): verifies that a seeded heap out-of-bounds write into
// a CPU storage block is caught and aborted by AddressSanitizer when built
// with -DLUCID_SANITIZE=address,undefined.

#include <gtest/gtest.h>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Storage.h"

using namespace lucid;

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define LUCID_HAS_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define LUCID_HAS_ASAN 1
#endif

#if defined(LUCID_HAS_ASAN)
namespace {

__attribute__((noinline)) void seeded_oob_write() {
    CpuStorage storage{allocate_aligned_bytes(64, Device::CPU), 64, Dtype::F32};
    volatile float* sink = reinterpret_cast<float*>(storage.ptr.get());
    sink[16] = 42.0f;
}

}  // namespace

TEST(SanitizerProbeDeathTest, SeededHeapOutOfBoundsWriteFailsUnderAsan) {
    EXPECT_DEATH(seeded_oob_write(), "AddressSanitizer: heap-buffer-overflow");
}
#else
TEST(SanitizerProbeDeathTest, SanitizerInactiveInReleaseBuild) {
    CpuStorage storage{allocate_aligned_bytes(64, Device::CPU), 64, Dtype::F32};
    EXPECT_NE(storage.ptr.get(), nullptr);
}
#endif
