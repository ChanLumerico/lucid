// lucid/_C/core/Device.h
//
// Enumeration that identifies which physical processing unit owns a tensor's
// storage, along with a constexpr helper for converting the enum to a
// human-readable name.  The two supported targets reflect the Apple Silicon
// architecture: CPU (backed by Apple Accelerate / posix_memalign buffers) and
// GPU (backed by MLX lazy graph nodes that execute on the Metal GPU).

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace lucid {

// Identifies the memory domain that owns a tensor's underlying storage.
//
// CPU: the tensor lives in ordinary virtual memory, accessible directly from
//      any thread via a pointer.  All Accelerate / BLAS / vDSP operations run
//      on CPU tensors.
// GPU: the tensor is represented as an mlx::core::array node in the MLX lazy
//      evaluation graph.  The buffer is GPU-private; reading it from the CPU
//      before evaluation causes a SIGBUS.
enum class Device : std::uint8_t {
    CPU,
    GPU,
};

// Returns the canonical lower-case name for device d ("cpu" or "gpu").
// Throws std::logic_error for unknown enumerators — this can only happen if
// the enum is extended without updating this function.
constexpr std::string_view device_name(Device d) {
    switch (d) {
    case Device::CPU:
        return "cpu";
    case Device::GPU:
        return "gpu";
    }
    throw std::logic_error("device_name: unknown Device");
}

}  // namespace lucid
