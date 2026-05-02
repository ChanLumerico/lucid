#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"
#include "ErrorBuilder.h"
#include "Profiler.h"
#include "Shape.h"

namespace lucid {

class LUCID_API OpScopeFull {
public:
    OpScopeFull(std::string_view name, Device device, Dtype dtype, Shape shape)
        : ctx_(std::string(name)), op_(name, device, dtype, std::move(shape)) {}

    OpScopeFull(const OpScopeFull&) = delete;
    OpScopeFull& operator=(const OpScopeFull&) = delete;

    void set_flops(std::int64_t f) { op_.set_flops(f); }

private:
    ErrorContextGuard ctx_;
    OpScope op_;
};

}  // namespace lucid
