#pragma once

// =====================================================================
// Lucid C++ engine — Dispatcher: device → IBackend* router.
// =====================================================================
//
// Phase 4: singleton registry mapping each Device to its IBackend.
// Op kernels replace `if (device == GPU) { mlx::* } else { cpu::* }`
// with `Dispatcher::for_device(d).method(...)`.
//
// Initialization is static: CpuBackend and GpuBackend register
// themselves via their static constructors before any op runs.
//
// Usage:
//   #include "../../backend/Dispatcher.h"
//   ...
//   auto& be = backend::Dispatcher::for_device(device);
//   auto out = be.exp(storage, shape, dtype);
//
// Layer: backend/. Depends on backend/IBackend.h, core/.

#include <array>
#include <memory>

#include "../core/Device.h"
#include "IBackend.h"

namespace lucid {
namespace backend {

class Dispatcher {
public:
    /// Returns the backend registered for `d`. Aborts if none registered.
    static IBackend& for_device(Device d) noexcept {
        auto& self = instance();
        const auto idx = static_cast<std::size_t>(d);
        return *self.backends_[idx];
    }

    /// Register a backend. Called by CpuBackend / GpuBackend at static init.
    static void register_backend(Device d, std::unique_ptr<IBackend> be) {
        auto& self = instance();
        self.backends_[static_cast<std::size_t>(d)] = std::move(be);
    }

private:
    static Dispatcher& instance() {
        static Dispatcher inst;
        return inst;
    }

    // Indexed by Device enum value: [0] = CPU, [1] = GPU.
    static constexpr std::size_t kNumDevices = 2;
    std::array<std::unique_ptr<IBackend>, kNumDevices> backends_;
};

}  // namespace backend
}  // namespace lucid
