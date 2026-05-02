#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::gpu {

struct MetalKernel {
    void* pipeline_state = nullptr;
    void* command_queue = nullptr;
    std::string name;

    MetalKernel() = default;

    MetalKernel(const MetalKernel&) = delete;
    MetalKernel& operator=(const MetalKernel&) = delete;
    MetalKernel(MetalKernel&& o) noexcept
        : pipeline_state(o.pipeline_state),
          command_queue(o.command_queue),
          name(std::move(o.name)) {
        o.pipeline_state = nullptr;
        o.command_queue = nullptr;
    }
    MetalKernel& operator=(MetalKernel&& o) noexcept {
        if (this != &o) {
            release_();
            pipeline_state = o.pipeline_state;
            command_queue = o.command_queue;
            name = std::move(o.name);
            o.pipeline_state = nullptr;
            o.command_queue = nullptr;
        }
        return *this;
    }
    ~MetalKernel() { release_(); }

    bool is_valid() const noexcept { return pipeline_state != nullptr; }

private:
    void release_() noexcept;
};

struct KernelLaunchConfig {
    std::array<std::size_t, 3> grid = {1, 1, 1};
    std::array<std::size_t, 3> threads = {1, 1, 1};
};

using KernelConstant = std::variant<int, float, std::size_t>;

MetalKernel compile_metal_kernel(const std::string& source, const std::string& function_name);

Storage run_metal_kernel(const MetalKernel& kernel,
                         const std::vector<Storage>& inputs,
                         const Shape& output_shape,
                         Dtype output_dtype,
                         const KernelLaunchConfig& config,
                         const std::vector<KernelConstant>& constants = {});

}  // namespace lucid::gpu
