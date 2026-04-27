#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace lucid {

enum class Device : std::uint8_t {
    CPU,
    GPU,
};

constexpr std::string_view device_name(Device d) {
    switch (d) {
        case Device::CPU: return "cpu";
        case Device::GPU: return "gpu";
    }
    throw std::logic_error("device_name: unknown Device");
}

}  // namespace lucid
