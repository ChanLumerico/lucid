// lucid/_C/core/Scope.h
//
// Combined error-context and profiling RAII scope for op entry points.
//
// OpScopeFull bundles an ErrorContextGuard (which pushes the op name onto the
// thread-local error call-stack) and an OpScope (which starts wall-clock
// timing and memory tracking) into a single object.  Op implementations that
// want both behaviours declare one OpScopeFull at function entry instead of
// managing two separate guards.
//
// The destruction order is significant: the member initialisation order
// (ctx_ before op_) means ErrorContextGuard is destroyed last, so the op name
// remains on the error stack for the entire duration of the OpScope event —
// including the record() call in OpScope's destructor.
//
// Usage:
//   OpScopeFull scope("conv2d", input->device(), input->dtype(), input->shape());
//   scope.set_flops(2LL * N * C * H * W * K);

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

// Composite RAII guard that activates both error-context annotation and
// profiler timing for the lifetime of the enclosing scope.
//
// Non-copyable and non-movable: the guards hold thread-local state whose
// correct nesting depends on stack-discipline construction/destruction order.
class LUCID_API OpScopeFull {
public:
    // Pushes name onto the error-context stack and starts an OpScope for
    // (name, device, dtype, shape).  Both are torn down when this object
    // goes out of scope.
    OpScopeFull(std::string_view name, Device device, Dtype dtype, Shape shape)
        : ctx_(std::string(name)), op_(name, device, dtype, std::move(shape)) {}

    OpScopeFull(const OpScopeFull&) = delete;
    OpScopeFull& operator=(const OpScopeFull&) = delete;

    // Forwards the FLOPs estimate to the underlying OpScope.
    void set_flops(std::int64_t f) { op_.set_flops(f); }

private:
    // Declared before op_ so it is constructed first and destroyed last,
    // keeping the op name on the error stack during the profiler record call.
    ErrorContextGuard ctx_;
    OpScope op_;
};

}  // namespace lucid
