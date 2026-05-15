// lucid/_C/backend/Dispatcher.h
//
// Singleton router that maps a Device tag to the concrete IBackend instance
// that services it.  On Apple Silicon, Device::CPU routes to CpuBackend
// (Apple Accelerate) and Device::GPU routes to GpuBackend (MLX / Metal).
// BackendInit.cpp registers both backends at module-load time; after that,
// all tensor operations call Dispatcher::for_device() to obtain a reference.

#pragma once

#include <array>
#include <memory>

#include "../core/Device.h"
#include "IBackend.h"

namespace lucid {
namespace backend {

// Routes compute operations to the backend registered for a given Device.
//
// The singleton owns one IBackend per Device slot (currently CPU and GPU).
// register_backend() transfers ownership via unique_ptr; for_device() returns
// a non-owning reference valid for the lifetime of the process.
//
// Thread safety: register_backend() is called once at startup before any
// worker threads exist; after that for_device() is read-only and safe to
// call concurrently.
class Dispatcher {
public:
    // Returns the backend registered for device `d`.
    // Behaviour is undefined if no backend has been registered for `d`.
    static IBackend& for_device(Device d) noexcept {
        auto& self = instance();
        const auto idx = static_cast<std::size_t>(d);
        return *self.backends_[idx];
    }

    // Registers (or replaces) the backend for device `d`.
    // Called exactly once per Device during process initialisation.
    static void register_backend(Device d, std::unique_ptr<IBackend> be) {
        auto& self = instance();
        self.backends_[static_cast<std::size_t>(d)] = std::move(be);
    }

private:
    // Meyers-singleton: constructed on first use, destroyed at program exit.
    static Dispatcher& instance() {
        static Dispatcher inst;
        return inst;
    }

    // Fixed-size array indexed by Device enum value; kNumDevices must match
    // the number of Device variants (CPU=0, GPU=1).
    static constexpr std::size_t kNumDevices = 2;
    std::array<std::unique_ptr<IBackend>, kNumDevices> backends_;
};

}  // namespace backend
}  // namespace lucid
