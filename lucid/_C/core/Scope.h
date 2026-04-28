#pragma once

// =====================================================================
// Lucid C++ engine — unified op-entry RAII scope.
// =====================================================================
//
// Every op forward currently opens at least one of:
//   1. `OpScope`            — profiler timing + memory delta recording
//   2. `ErrorContextGuard`  — push the op name onto the thread-local
//                             error-trace stack (Phase 1)
//   3. (planned) MemoryTracker scope marker for high-water-mark resets.
//
// Maintaining three independent guards at the top of every op is
// boilerplate. `OpScopeFull` packs them into one declaration:
//
//     OpScopeFull scope("matmul", device, dtype, out_shape);
//
// Behavior:
//   - The error-context guard is *always* installed (cheap — single
//     std::vector push/pop on a thread-local), so any `ErrorBuilder`
//     throw inside the op picks up the trace.
//   - The profiler `OpScope` is still no-op when no profiler is
//     active on the thread (one TLS load + null check).
//
// Op forwards using the legacy individual `OpScope` continue to work;
// migration to `OpScopeFull` happens incrementally.
//
// Layer: core/.

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

    /// Forwarded to the inner `OpScope` so kernels can record FLOPs.
    void set_flops(std::int64_t f) { op_.set_flops(f); }

private:
    ErrorContextGuard ctx_;
    OpScope op_;
};

}  // namespace lucid
