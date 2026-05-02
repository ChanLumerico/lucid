#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {
namespace kernel {

struct KernelPolicy {
    bool saves_inputs = true;
    bool saves_output = false;
    bool has_gradient = true;
    bool deterministic = true;
};

class IKernel {
public:
    virtual ~IKernel() = default;

    virtual std::string_view name() const noexcept = 0;

    virtual Storage compute(const std::vector<Storage>&) {
        throw std::logic_error(std::string(name()) +
                               ": compute() not implemented — use the concrete static forward()");
    }

    virtual std::vector<Storage> apply(Storage grad_out) = 0;
};

}  // namespace kernel
}  // namespace lucid
