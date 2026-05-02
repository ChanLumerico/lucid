#pragma once

#include <array>
#include <memory>

#include "../core/Device.h"
#include "IBackend.h"

namespace lucid {
namespace backend {

class Dispatcher {
public:
    static IBackend& for_device(Device d) noexcept {
        auto& self = instance();
        const auto idx = static_cast<std::size_t>(d);
        return *self.backends_[idx];
    }

    static void register_backend(Device d, std::unique_ptr<IBackend> be) {
        auto& self = instance();
        self.backends_[static_cast<std::size_t>(d)] = std::move(be);
    }

private:
    static Dispatcher& instance() {
        static Dispatcher inst;
        return inst;
    }

    static constexpr std::size_t kNumDevices = 2;
    std::array<std::unique_ptr<IBackend>, kNumDevices> backends_;
};

}  // namespace backend
}  // namespace lucid
